//! Hermes provider — reads/writes sessions from a SQLite `state.db` database.
//!
//! Hermes (the `hermes-agent` Python runtime + `hermes-desktop` Electron UI)
//! stores every profile's conversations in a single SQLite file:
//!
//! - Default profile: `<hermes_home>/state.db`
//! - Named profile:   `<hermes_home>/profiles/<name>/state.db`
//!
//! where `<hermes_home>` is `%LOCALAPPDATA%\hermes` on Windows (overridable via
//! the `HERMES_HOME` env var, mirroring the desktop's `profileHome`).
//!
//! ## Schema (authoritative source: `hermes_state.py` `SCHEMA_SQL`)
//!
//! Two tables matter for casr:
//!
//! - `sessions(id, source, model, started_at REAL, ended_at REAL, cwd, title,
//!   message_count, tool_call_count, …)` — one row per conversation.
//! - `messages(id, session_id, role, content, tool_call_id, tool_calls,
//!   tool_name, timestamp REAL, finish_reason, reasoning, reasoning_content,
//!   reasoning_details, …, active INTEGER NOT NULL DEFAULT 1)` — one row per
//!   turn. `active=0` marks rewound/compressed messages; casr reads only the
//!   live (`active=1`) view.
//!
//! Multimodal `content` is stored with the prefix `"\x00json:"` followed by a
//! JSON array of content blocks (see `hermes_state._CONTENT_JSON_PREFIX` /
//! `_encode_content`). Plain-text messages store the raw string unchanged.
//!
//! `messages_fts*` full-text tables are populated by `AFTER INSERT` triggers;
//! casr never writes to them directly.
//!
//! ## Resume command
//!
//! `hermes --resume <session-id>` is a best-effort suggestion — confirm the
//! exact flag against the installed `hermes-agent` CLI.

use std::path::{Path, PathBuf};

use anyhow::Context;
use rusqlite::{Connection, OpenFlags};
use tracing::{debug, info, trace, warn};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, flatten_content,
    normalize_role, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// Prefix marking JSON-encoded (multimodal) message content in the `content`
/// column. Mirrors `hermes_state._CONTENT_JSON_PREFIX = "\x00json:"`.
const CONTENT_JSON_PREFIX: &str = "\x00json:";

/// Hermes provider implementation.
pub struct Hermes;

impl Hermes {
    /// Root directory for Hermes data. Respects the `HERMES_HOME` env var
    /// override (required for tests, mirroring `CLAUDE_HOME`/`CURSOR_HOME`).
    fn home_dir() -> Option<PathBuf> {
        if let Ok(home) = std::env::var("HERMES_HOME") {
            return Some(PathBuf::from(home));
        }
        // `dirs::cache_dir()` is `%LOCALAPPDATA%` on Windows, `~/.cache` on
        // Linux, `~/Library/Caches` on macOS — Hermes installs under
        // `%LOCALAPPDATA%\hermes` on Windows.
        if let Some(cache) = dirs::cache_dir() {
            return Some(cache.join("hermes"));
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return Some(PathBuf::from(local).join("hermes"));
        }
        dirs::home_dir().map(|h| h.join(".hermes"))
    }

    /// Resolve the active profile's `state.db` path.
    ///
    /// Reads `<home>/active_profile`; if present and non-empty, the DB lives at
    /// `<home>/profiles/<name>/state.db`, otherwise at `<home>/state.db`.
    /// Mirrors the desktop's `activeStateDbPath()` / `profileHome()`.
    fn state_db() -> Option<PathBuf> {
        let home = Self::home_dir()?;
        let active_profile = home.join("active_profile");
        if active_profile.is_file() {
            if let Ok(name) = std::fs::read_to_string(&active_profile) {
                let name = name.trim();
                if !name.is_empty() {
                    return Some(home.join("profiles").join(name).join("state.db"));
                }
            }
        }
        Some(home.join("state.db"))
    }

    /// Open a Hermes DB read-only with a busy timeout.
    fn open_db(path: &Path) -> anyhow::Result<Connection> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open Hermes DB: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    /// Open a Hermes DB read-write (+ create) with a busy timeout.
    fn open_db_rw(path: &Path) -> anyhow::Result<Connection> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open Hermes DB for writing: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    /// Check that the `sessions` table exists (a Hermes DB marker).
    fn has_sessions_table(conn: &Connection) -> bool {
        conn.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'")
            .and_then(|mut stmt| stmt.exists([]))
            .unwrap_or(false)
    }

    /// Resolve a `(db_path, session_id)` pair from a `read_session` path.
    ///
    /// Supports two shapes (mirroring the Cursor provider's parent-as-file
    /// semantics):
    /// - `<state.db>` (bare DB file) → `session_id = None` (caller reads the
    ///   most-recent session; this is the shape produced by `write_session` and
    ///   by `owns_session`/`list_sessions` per spec).
    /// - `<state.db>/<session_id>` (virtual path) → reads that specific
    ///   session, enabling per-session reads on multi-session DBs.
    fn resolve_db_and_session(path: &Path) -> (PathBuf, Option<String>) {
        if path.is_file() {
            return (path.to_path_buf(), None);
        }
        if let Some(parent) = path.parent()
            && parent.is_file()
        {
            let sid = path
                .file_name()
                .and_then(|f| f.to_str())
                .map(|s| s.to_string());
            return (parent.to_path_buf(), sid);
        }
        (path.to_path_buf(), None)
    }

    /// Core write logic against an explicit DB path. Split out so the in-file
    /// unit tests can exercise it against a temp DB without mutating the
    /// process environment (which is `unsafe` under `forbid(unsafe_code)`).
    fn write_to_db(
        &self,
        db_path: &Path,
        session: &CanonicalSession,
        _opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let target_session_id = uuid::Uuid::new_v4().to_string();
        let now_secs = chrono::Utc::now().timestamp() as f64;

        // Ensure the parent directory exists (open_db_rw creates the file but
        // not its containing directory).
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create Hermes DB directory: {}", parent.display()))?;
        }

        let conn = Self::open_db_rw(db_path)?;

        // Create the schema if this is a fresh DB. We create only the two
        // tables casr uses (FTS tables/triggers are not needed for read-back).
        conn.execute_batch(SCHEMA_SQL)
            .context("failed to initialize Hermes schema")?;

        // Pragmas: WAL + foreign keys, matching hermes_state._execute_write.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        // BEGIN IMMEDIATE with a small jitter retry on "database is locked".
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..MAX_WRITE_RETRIES {
            match Self::run_write_tx(&conn, session, &target_session_id, now_secs) {
                Ok(()) => {
                    last_err = None;
                    break;
                }
                Err(e) => {
                    let msg = e.to_string().to_ascii_lowercase();
                    if (msg.contains("locked") || msg.contains("busy"))
                        && attempt + 1 < MAX_WRITE_RETRIES
                    {
                        warn!(attempt, error = %e, "Hermes DB locked; retrying");
                        std::thread::sleep(std::time::Duration::from_millis(WRITE_RETRY_SLEEP_MS));
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        if let Some(e) = last_err {
            return Err(e);
        }

        info!(
            target_session_id,
            path = %db_path.display(),
            messages = session.messages.len(),
            "Hermes session written"
        );

        Ok(WrittenSession {
            paths: vec![db_path.join(&target_session_id)],
            session_id: target_session_id.clone(),
            resume_command: self.resume_command(&target_session_id),
            backup_path: None,
        })
    }

    /// Run one write transaction: insert the session row + all messages, then
    /// commit. Rolls back on any error.
    fn run_write_tx(
        conn: &Connection,
        session: &CanonicalSession,
        target_session_id: &str,
        now_secs: f64,
    ) -> anyhow::Result<()> {
        conn.execute("BEGIN IMMEDIATE", [])?;

        let tx_result = (|| -> anyhow::Result<()> {
            let model = session.model_name.as_deref();
            let started = session
                .started_at
                .map(|ms| ms as f64 / 1000.0)
                .unwrap_or(now_secs);
            let cwd = session.workspace.as_deref().map(|p| p.to_string_lossy().to_string());
            let title = session.title.as_deref();

            // INSERT OR IGNORE — a collision (astronomically unlikely with a
            // fresh v4 UUID) leaves the existing session row untouched.
            conn.execute(
                "INSERT OR IGNORE INTO sessions \
                 (id, source, model, started_at, cwd, title) \
                 VALUES (?1, 'casr_import', ?2, ?3, ?4, ?5)",
                rusqlite::params![target_session_id, model, started, cwd, title],
            )?;

            let mut tool_call_total: i64 = 0;

            for msg in &session.messages {
                let role = hermes_role_str(&msg.role);
                let content = encode_content(msg);
                let tool_call_id: Option<&str> = msg
                    .tool_results
                    .first()
                    .and_then(|r| r.call_id.as_deref());
                let tool_calls_json = encode_tool_calls(&msg.tool_calls);
                let tool_name = msg.tool_calls.first().map(|t| t.name.as_str());

                let ts = msg
                    .timestamp
                    .map(|ms| ms as f64 / 1000.0)
                    .unwrap_or(now_secs);

                conn.execute(
                    "INSERT INTO messages \
                     (session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![
                        target_session_id,
                        role,
                        content,
                        tool_call_id,
                        tool_calls_json,
                        tool_name,
                        ts,
                    ],
                )?;

                let msg_tools = msg.tool_calls.len() as i64;
                tool_call_total += msg_tools;
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1, \
                     tool_call_count = tool_call_count + ?1 WHERE id = ?2",
                    rusqlite::params![msg_tools, target_session_id],
                )?;
            }

            // Stamp the session start if it was missing (INSERT OR IGNORE may
            // have skipped when started was already set above; harmless here).
            let _ = tool_call_total; // already applied per-message above
            Ok(())
        })();

        match tx_result {
            Ok(()) => {
                conn.execute("COMMIT", [])?;
                Ok(())
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    /// Same logic as [`owns_session`](Self::owns_session) but against an
    /// explicit DB path. Split out so in-file unit tests can exercise it
    /// without env-var mutation (forbidden under `#![forbid(unsafe_code)]`).
    fn owns_session_in_db(db_path: &Path, session_id: &str) -> Option<PathBuf> {
        if !db_path.is_file() {
            return None;
        }
        let conn = Self::open_db(db_path).ok()?;
        if !Self::has_sessions_table(&conn) {
            return None;
        }
        let exists = conn
            .prepare("SELECT 1 FROM sessions WHERE id = ?1")
            .and_then(|mut stmt| stmt.exists(rusqlite::params![session_id]))
            .unwrap_or(false);
        if exists {
            debug!(db = %db_path.display(), session_id, "found Hermes session");
            // Return a VIRTUAL path (db/<session_id>) so that read_session
            // resolves the REQUESTED session via resolve_db_and_session, not
            // the most-recent session. Mirrors cursor.rs.
            Some(db_path.join(session_id))
        } else {
            None
        }
    }
}

impl Provider for Hermes {
    fn name(&self) -> &str {
        "Hermes"
    }

    fn slug(&self) -> &str {
        "hermes"
    }

    fn cli_alias(&self) -> &str {
        "hme"
    }

    fn detect(&self) -> DetectionResult {
        let mut evidence = Vec::new();
        let mut installed = false;

        if which::which("hermes").is_ok() {
            evidence.push("hermes binary found in PATH".to_string());
            installed = true;
        }

        if let Some(db) = Self::state_db()
            && db.is_file()
        {
            evidence.push(format!("{} exists", db.display()));
            installed = true;
        }

        if let Some(home) = Self::home_dir()
            && home.is_dir()
        {
            evidence.push(format!("{} exists", home.display()));
        }

        trace!(provider = "hermes", ?evidence, installed, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        match Self::state_db() {
            Some(db) if db.is_file() => vec![db],
            _ => vec![],
        }
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        let db = Self::state_db()?;
        Self::owns_session_in_db(&db, session_id)
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let db = Self::state_db()?;
        if !db.is_file() {
            return Some(vec![]);
        }
        let conn = match Self::open_db(&db) {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, db = %db.display(), "failed to open Hermes DB for listing");
                return Some(vec![]);
            }
        };
        if !Self::has_sessions_table(&conn) {
            return Some(vec![]);
        }
        let mut stmt = match conn.prepare("SELECT id FROM sessions ORDER BY started_at DESC") {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "failed to list Hermes sessions");
                return Some(vec![]);
            }
        };
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                // Use a virtual path (db/<session_id>) so that read_session
                // can resolve the specific session via resolve_db_and_session.
                Ok((id.clone(), db.join(&id)))
            })
            .ok()?;
        let mut out = Vec::new();
        for row in rows.flatten() {
            out.push(row);
        }
        Some(out)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading Hermes session");

        let (db_path, requested_id) = Self::resolve_db_and_session(path);
        let conn = Self::open_db(&db_path).with_context(|| {
            format!("failed to open Hermes DB: {}", db_path.display())
        })?;

        if !Self::has_sessions_table(&conn) {
            anyhow::bail!("not a Hermes state.db (no sessions table): {}", db_path.display());
        }

        // Load the session row — either the requested id or the most-recent.
        let session_row: Option<SessionRow> = match requested_id.as_deref() {
            Some(id) => query_session_row(&conn, id)?,
            None => query_most_recent_session_row(&conn)?,
        };
        let Some(row) = session_row else {
            anyhow::bail!("no Hermes session found in {}", db_path.display());
        };

        // Load messages for this session (live view only: active = 1).
        let mut messages: Vec<CanonicalMessage> = Vec::new();
        let mut stmt = conn
            .prepare(
                "SELECT id, role, content, tool_call_id, tool_calls, tool_name, timestamp, \
                 finish_reason, reasoning_content, reasoning_details \
                 FROM messages WHERE session_id = ?1 AND active = 1 \
                 ORDER BY timestamp, id",
            )
            .context("failed to prepare Hermes messages query")?;

        let rows = stmt
            .query_map(rusqlite::params![row.id], |r| {
                Ok(MessageRow {
                    role: r.get::<_, String>("role")?,
                    content: r.get::<_, Option<String>>("content")?,
                    tool_call_id: r.get::<_, Option<String>>("tool_call_id")?,
                    tool_calls: r.get::<_, Option<String>>("tool_calls")?,
                    tool_name: r.get::<_, Option<String>>("tool_name")?,
                    timestamp: r.get::<_, f64>("timestamp")?,
                    finish_reason: r.get::<_, Option<String>>("finish_reason")?,
                    reasoning_content: r.get::<_, Option<String>>("reasoning_content")?,
                    reasoning_details: r.get::<_, Option<String>>("reasoning_details")?,
                })
            })
            .context("failed to query Hermes messages")?;

        for msg_row in rows {
            let mr = msg_row.context("failed to read Hermes message row")?;
            let role = normalize_role(&mr.role);
            let content = decode_content(mr.content.as_deref().unwrap_or(""));
            let tool_calls = decode_tool_calls(mr.tool_calls.as_deref());

            let tool_results = if matches!(role, MessageRole::Tool) {
                vec![ToolResult {
                    call_id: mr.tool_call_id.clone(),
                    content: content.clone(),
                    is_error: false,
                }]
            } else {
                Vec::new()
            };

            let mut extra = serde_json::Map::new();
            if let Some(fr) = &mr.finish_reason {
                extra.insert("finish_reason".into(), serde_json::Value::String(fr.clone()));
            }
            if let Some(rc) = &mr.reasoning_content {
                extra.insert("reasoning_content".into(), serde_json::Value::String(rc.clone()));
            }
            if let Some(rd) = &mr.reasoning_details {
                extra.insert("reasoning_details".into(), serde_json::Value::String(rd.clone()));
            }
            if let Some(tn) = &mr.tool_name {
                extra.insert("tool_name".into(), serde_json::Value::String(tn.clone()));
            }
            if let Some(tc) = &mr.tool_call_id {
                extra.insert("tool_call_id".into(), serde_json::Value::String(tc.clone()));
            }

            messages.push(CanonicalMessage {
                idx: 0, // reindexed below
                role,
                content,
                timestamp: Some((mr.timestamp * 1000.0) as i64),
                author: row.model.clone(),
                tool_calls,
                tool_results,
                extra: serde_json::Value::Object(extra),
            });
        }

        reindex_messages(&mut messages);

        let title = messages
            .iter()
            .find(|m| m.role == MessageRole::User)
            .map(|m| truncate_title(&m.content, 100))
            .or_else(|| row.title.clone().filter(|t| !t.is_empty()));

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "source".into(),
            serde_json::Value::String(row.source.clone().unwrap_or_else(|| "hermes".to_string())),
        );
        if let Some(src) = &row.source {
            metadata.insert("hermes_source".into(), serde_json::Value::String(src.clone()));
        }

        debug!(
            session_id = %row.id,
            messages = messages.len(),
            "Hermes session parsed"
        );

        Ok(CanonicalSession {
            session_id: row.id,
            provider_slug: "hermes".to_string(),
            workspace: row.cwd.as_deref().map(PathBuf::from),
            title,
            started_at: row.started_at.map(|s| (s * 1000.0) as i64),
            ended_at: row.ended_at.map(|e| (e * 1000.0) as i64),
            messages,
            metadata: serde_json::Value::Object(metadata),
            source_path: db_path,
            model_name: row.model.clone(),
        })
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let db = Self::state_db()
            .ok_or_else(|| anyhow::anyhow!("cannot determine Hermes state.db path"))?;
        self.write_to_db(&db, session, opts)
    }

    fn resume_command(&self, session_id: &str) -> String {
        format!("hermes --resume {session_id}")
    }
}

// ---------------------------------------------------------------------------
// Schema + row structs
// ---------------------------------------------------------------------------

/// Minimal `sessions`/`messages` DDL (matches `hermes_state.SCHEMA_SQL` for
/// the columns casr reads/writes; FTS tables and triggers are intentionally
/// omitted — they are trigger-driven and irrelevant to read-back).
const SCHEMA_SQL: &str = "\
CREATE TABLE IF NOT EXISTS sessions (\
    id TEXT PRIMARY KEY,\
    source TEXT NOT NULL,\
    user_id TEXT,\
    model TEXT,\
    model_config TEXT,\
    system_prompt TEXT,\
    parent_session_id TEXT,\
    started_at REAL NOT NULL,\
    ended_at REAL,\
    end_reason TEXT,\
    message_count INTEGER DEFAULT 0,\
    tool_call_count INTEGER DEFAULT 0,\
    input_tokens INTEGER DEFAULT 0,\
    output_tokens INTEGER DEFAULT 0,\
    cache_read_tokens INTEGER DEFAULT 0,\
    cache_write_tokens INTEGER DEFAULT 0,\
    reasoning_tokens INTEGER DEFAULT 0,\
    cwd TEXT,\
    title TEXT,\
    archived INTEGER NOT NULL DEFAULT 0,\
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)\
);\
CREATE TABLE IF NOT EXISTS messages (\
    id INTEGER PRIMARY KEY AUTOINCREMENT,\
    session_id TEXT NOT NULL REFERENCES sessions(id),\
    role TEXT NOT NULL,\
    content TEXT,\
    tool_call_id TEXT,\
    tool_calls TEXT,\
    tool_name TEXT,\
    timestamp REAL NOT NULL,\
    token_count INTEGER,\
    finish_reason TEXT,\
    reasoning TEXT,\
    reasoning_content TEXT,\
    reasoning_details TEXT,\
    platform_message_id TEXT,\
    observed INTEGER DEFAULT 0,\
    active INTEGER NOT NULL DEFAULT 1\
);\
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);\
";

const MAX_WRITE_RETRIES: usize = 5;
const WRITE_RETRY_SLEEP_MS: u64 = 25;

#[derive(Debug, Clone)]
struct SessionRow {
    id: String,
    source: Option<String>,
    model: Option<String>,
    started_at: Option<f64>,
    ended_at: Option<f64>,
    cwd: Option<String>,
    title: Option<String>,
}

#[derive(Debug, Clone)]
struct MessageRow {
    role: String,
    content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<String>,
    tool_name: Option<String>,
    timestamp: f64,
    finish_reason: Option<String>,
    reasoning_content: Option<String>,
    reasoning_details: Option<String>,
}

fn query_session_row(conn: &Connection, id: &str) -> anyhow::Result<Option<SessionRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, source, model, started_at, ended_at, cwd, title \
         FROM sessions WHERE id = ?1",
    )?;
    let mut rows = stmt.query_map(rusqlite::params![id], map_session_row)?;
    Ok(rows.next().transpose()?)
}

fn query_most_recent_session_row(conn: &Connection) -> anyhow::Result<Option<SessionRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, source, model, started_at, ended_at, cwd, title \
         FROM sessions ORDER BY started_at DESC LIMIT 1",
    )?;
    let mut rows = stmt.query_map([], map_session_row)?;
    Ok(rows.next().transpose()?)
}

fn map_session_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
    Ok(SessionRow {
        id: row.get("id")?,
        source: row.get("source")?,
        model: row.get("model")?,
        started_at: row.get("started_at")?,
        ended_at: row.get("ended_at")?,
        cwd: row.get("cwd")?,
        title: row.get("title")?,
    })
}

// ---------------------------------------------------------------------------
// Content encoding (mirrors hermes_state._encode_content / _decode_content)
// ---------------------------------------------------------------------------

/// Encode a canonical message's content for the Hermes `content` column.
///
/// Plain-text messages (no tool calls/results) are stored as the raw string,
/// matching `_encode_content` returning scalars unchanged. Messages carrying
/// tool payloads are stored as `"\x00json:" + json(blocks)` so the structure
/// survives a round-trip.
fn encode_content(msg: &CanonicalMessage) -> String {
    if msg.tool_calls.is_empty() && msg.tool_results.is_empty() {
        return msg.content.clone();
    }
    let mut blocks: Vec<serde_json::Value> = Vec::new();
    if !msg.content.is_empty() {
        blocks.push(serde_json::json!({ "type": "text", "text": msg.content }));
    }
    for tc in &msg.tool_calls {
        blocks.push(serde_json::json!({
            "type": "tool_use",
            "id": tc.id.as_deref().unwrap_or(""),
            "name": tc.name,
            "input": tc.arguments,
        }));
    }
    for tr in &msg.tool_results {
        blocks.push(serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tr.call_id.as_deref().unwrap_or(""),
            "content": tr.content,
            "is_error": tr.is_error,
        }));
    }
    format!("{CONTENT_JSON_PREFIX}{}", serde_json::to_string(&blocks).unwrap_or_default())
}

/// Decode a stored `content` value into a flat text string.
///
/// Strings prefixed with `"\x00json:"` are parsed as a JSON array of content
/// blocks and flattened; all other strings are returned unchanged.
fn decode_content(stored: &str) -> String {
    if let Some(json_str) = stored.strip_prefix(CONTENT_JSON_PREFIX) {
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(v) => flatten_content(&v),
            Err(_) => stored.to_string(),
        }
    } else {
        stored.to_string()
    }
}

/// Encode tool calls into the OpenAI-shaped JSON Hermes stores in `tool_calls`.
fn encode_tool_calls(calls: &[ToolCall]) -> Option<String> {
    if calls.is_empty() {
        return None;
    }
    let arr: Vec<serde_json::Value> = calls
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id.as_deref().unwrap_or(""),
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.arguments,
                }
            })
        })
        .collect();
    serde_json::to_string(&arr).ok()
}

/// Decode the OpenAI-shaped `tool_calls` JSON column into canonical tool calls.
fn decode_tool_calls(raw: Option<&str>) -> Vec<ToolCall> {
    let Some(raw) = raw else {
        return vec![];
    };
    let Some(arr) = serde_json::from_str::<Vec<serde_json::Value>>(raw).ok() else {
        return vec![];
    };
    arr.iter()
        .filter_map(|item| {
            let func = item.get("function")?;
            Some(ToolCall {
                id: item.get("id").and_then(|v| v.as_str()).map(String::from),
                name: func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                arguments: func.get("arguments").cloned().unwrap_or(serde_json::Value::Null),
            })
        })
        .collect()
}

/// Map a canonical role to the Hermes role string.
fn hermes_role_str(role: &MessageRole) -> &str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
        MessageRole::System => "system",
        MessageRole::Other(_) => "assistant",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult};
    use crate::providers::{Provider, WriteOptions};
    use std::path::PathBuf;

    fn sample_message(role: MessageRole, content: &str) -> CanonicalMessage {
        CanonicalMessage {
            idx: 0,
            role,
            content: content.to_string(),
            timestamp: Some(1_700_000_000_000),
            author: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            extra: serde_json::Value::Null,
        }
    }

    fn sample_session(messages: Vec<CanonicalMessage>) -> CanonicalSession {
        CanonicalSession {
            session_id: "src-001".to_string(),
            provider_slug: "claude-code".to_string(),
            workspace: Some(PathBuf::from("/data/projects/myapp")),
            title: Some("Test".to_string()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_000_001_000),
            messages,
            metadata: serde_json::json!({}),
            source_path: PathBuf::from("/tmp/src.jsonl"),
            model_name: Some("claude-sonnet-4-5".to_string()),
        }
    }

    // -----------------------------------------------------------------------
    // Content encode/decode round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn content_plain_string_round_trips_unchanged() {
        let msg = sample_message(MessageRole::Assistant, "just text");
        let encoded = encode_content(&msg);
        assert!(
            !encoded.starts_with(CONTENT_JSON_PREFIX),
            "plain text must not be JSON-prefixed"
        );
        assert_eq!(encoded, "just text");
        assert_eq!(decode_content(&encoded), "just text");
    }

    #[test]
    fn content_multimodal_round_trips_through_prefix() {
        let mut msg = sample_message(MessageRole::Assistant, "Plan.");
        msg.tool_calls.push(ToolCall {
            id: Some("t1".to_string()),
            name: "Read".to_string(),
            arguments: serde_json::json!({"file_path": "a.rs"}),
        });
        let encoded = encode_content(&msg);
        assert!(
            encoded.starts_with(CONTENT_JSON_PREFIX),
            "tool-bearing content must be JSON-prefixed"
        );
        // The text portion survives the flatten.
        assert!(decode_content(&encoded).contains("Plan."));
    }

    #[test]
    fn tool_calls_encode_decode_round_trips() {
        let calls = vec![ToolCall {
            id: Some("call_1".to_string()),
            name: "Bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        }];
        let json = encode_tool_calls(&calls).expect("non-empty encodes");
        let back = decode_tool_calls(Some(&json));
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].name, "Bash");
        assert_eq!(back[0].id.as_deref(), Some("call_1"));
    }

    // -----------------------------------------------------------------------
    // Write → read round-trip against a temp DB (no env mutation needed)
    // -----------------------------------------------------------------------

    #[test]
    fn write_then_read_round_trips_text_messages() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("state.db");

        let original = sample_session(vec![
            sample_message(MessageRole::User, "Fix the login bug in auth.rs"),
            sample_message(
                MessageRole::Assistant,
                "I found the issue. The token validation was using an expired key.",
            ),
            sample_message(MessageRole::User, "Great, can you also add a test for it?"),
            sample_message(
                MessageRole::Assistant,
                "Done. I added a test in tests/auth_test.rs.",
            ),
        ]);

        let written = Hermes
            .write_to_db(&db_path, &original, &WriteOptions { force: false })
            .expect("write should succeed");

        assert_eq!(written.paths, vec![db_path.join(&written.session_id)]);
        assert!(!written.session_id.is_empty());
        assert_eq!(written.resume_command, format!("hermes --resume {}", written.session_id));

        let readback = Hermes
            .read_session(&written.paths[0])
            .expect("read-back should succeed");

        assert_eq!(readback.provider_slug, "hermes");
        assert_eq!(readback.session_id, written.session_id);
        assert_eq!(readback.messages.len(), original.messages.len());
        for (i, (orig, rb)) in original.messages.iter().zip(readback.messages.iter()).enumerate() {
            assert_eq!(orig.role, rb.role, "msg {i}: role mismatch");
            assert_eq!(orig.content, rb.content, "msg {i}: content mismatch");
        }
        // Timestamps survive the millis↔seconds round-trip (whole seconds).
        assert_eq!(
            readback.messages[0].timestamp,
            Some(1_700_000_000_000),
            "timestamp should round-trip to whole-second millis"
        );
    }

    #[test]
    fn write_then_read_preserves_tool_messages() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("state.db");

        let mut asst = sample_message(MessageRole::Assistant, "Reading file.");
        asst.tool_calls.push(ToolCall {
            id: Some("tc-9".to_string()),
            name: "Read".to_string(),
            arguments: serde_json::json!({"file_path": "src/main.rs"}),
        });
        let mut tool_msg = sample_message(MessageRole::Tool, "file contents here");
        tool_msg.tool_results.push(ToolResult {
            call_id: Some("tc-9".to_string()),
            content: "file contents here".to_string(),
            is_error: false,
        });

        let original = sample_session(vec![asst, tool_msg]);

        Hermes
            .write_to_db(&db_path, &original, &WriteOptions { force: false })
            .expect("write should succeed");

        let readback = Hermes.read_session(&db_path).expect("read-back should succeed");

        assert_eq!(readback.messages.len(), 2);
        assert_eq!(readback.messages[0].role, MessageRole::Assistant);
        assert_eq!(readback.messages[0].tool_calls.len(), 1);
        assert_eq!(readback.messages[0].tool_calls[0].name, "Read");
        assert_eq!(readback.messages[1].role, MessageRole::Tool);
        assert_eq!(readback.messages[1].tool_results.len(), 1);
        assert_eq!(readback.messages[1].tool_results[0].call_id.as_deref(), Some("tc-9"));
    }

    // -----------------------------------------------------------------------
    // Multi-session: virtual path resolves the REQUESTED session, not most-recent
    // -----------------------------------------------------------------------

    #[test]
    fn virtual_path_resolves_requested_session_not_most_recent() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("state.db");

        // Write session A (first — will be the OLDER session).
        let session_a = sample_session(vec![
            sample_message(MessageRole::User, "Session A: hello from A"),
            sample_message(MessageRole::Assistant, "Session A: reply from A"),
        ]);
        let written_a = Hermes
            .write_to_db(&db_path, &session_a, &WriteOptions { force: false })
            .expect("write A should succeed");

        // Write session B (second — will be the MOST-RECENT session).
        let mut session_b = sample_session(vec![
            sample_message(MessageRole::User, "Session B: completely different content"),
            sample_message(MessageRole::Assistant, "Session B: unique assistant reply"),
        ]);
        session_b.session_id = "src-002".to_string();
        let written_b = Hermes
            .write_to_db(&db_path, &session_b, &WriteOptions { force: false })
            .expect("write B should succeed");

        // owns_session_in_db for A should return a virtual path.
        let path_a = Hermes::owns_session_in_db(&db_path, &written_a.session_id)
            .expect("should find session A");
        // owns_session_in_db for B should return a different virtual path.
        let path_b = Hermes::owns_session_in_db(&db_path, &written_b.session_id)
            .expect("should find session B");

        assert_ne!(path_a, path_b, "virtual paths must differ");

        // Read session A via its virtual path — must return A, not B (the
        // most-recent session).  This is the critical assertion: without the
        // virtual-path fix, read_session on a bare db would return B.
        let readback_a = Hermes
            .read_session(&path_a)
            .expect("read session A should succeed");
        assert_eq!(
            readback_a.session_id, written_a.session_id,
            "must resolve session A, not the most-recent session B"
        );
        assert!(
            readback_a.messages[0].content.contains("Session A"),
            "content should be from session A, got: {}",
            readback_a.messages[0].content
        );

        // Read session B via its virtual path — must return B.
        let readback_b = Hermes
            .read_session(&path_b)
            .expect("read session B should succeed");
        assert_eq!(
            readback_b.session_id, written_b.session_id,
            "must resolve session B"
        );
        assert!(
            readback_b.messages[0].content.contains("Session B"),
            "content should be from session B, got: {}",
            readback_b.messages[0].content
        );
    }
}
