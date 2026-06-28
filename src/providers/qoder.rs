//! Qoder provider — reads/writes JSONL transcripts under `~/.qoder/cache/projects/`.
//!
//! Qoder (the AI coding IDE) persists each conversation as a JSONL transcript:
//!
//! ```text
//! <home>/cache/projects/<project-folder>/conversation-history/<8hex>/<8hex>.jsonl
//! ```
//!
//! where `<8hex>` is the first 8 hex characters of the session UUID and
//! `<project-folder>` is `<workspace-basename>-<8hex-hash>` (Qoder derives the
//! hash from the workspace path).
//!
//! ## Record schema
//!
//! Each line is exactly (verified across real transcripts):
//!
//! ```json
//! {"role":"user"|"assistant","message":{"content":[{"type":"text","text":"…"}]}}
//! ```
//!
//! No timestamps, no tool blocks, and no `tool`/`system` role are stored.
//! Conversions to/from Qoder are therefore lossy for those fields (see
//! `docs/PROVIDERS_HERMES_QODER.md`).
//!
//! ## Project-folder hash
//!
//! `project_folder` reuses an existing `<basename>-*` folder when one exists
//! (so writes land in the folder Qoder already uses for an opened workspace).
//! Otherwise it computes a best-effort `sha256(normalized_path)[:8]` hash —
//! Qoder's own hash algorithm is undocumented, so sessions for never-opened
//! workspaces may not be auto-discovered by the IDE.
//!
//! ## Resume command
//!
//! `qoder --resume <id>` is best-effort — Qoder resumes via the IDE UI, not a
//! CLI flag.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::Context;
use sha2::{Digest, Sha256};
use tracing::{debug, info, trace, warn};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, flatten_content, normalize_role,
    reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// Qoder provider implementation.
pub struct Qoder;

impl Qoder {
    /// Root directory for Qoder data. Respects the `QODER_HOME` env var
    /// override (required for tests, mirroring `CLAUDE_HOME`/`CURSOR_HOME`).
    fn home_dir() -> Option<PathBuf> {
        if let Ok(home) = std::env::var("QODER_HOME") {
            return Some(PathBuf::from(home));
        }
        dirs::home_dir().map(|h| h.join(".qoder"))
    }

    /// `cache/projects` directory where Qoder stores per-workspace transcripts.
    fn projects_dir() -> Option<PathBuf> {
        Self::home_dir().map(|h| h.join("cache").join("projects"))
    }

    /// Derive the Qoder project-folder name for a workspace.
    ///
    /// Reuses the most-recently-modified existing `<basename>-*` folder when
    /// present; otherwise falls back to a best-effort `sha256`-derived hash.
    fn project_folder(workspace: Option<&Path>) -> Option<String> {
        let projects_dir = Self::projects_dir()?;
        Self::project_folder_for(&projects_dir, workspace)
    }

    /// Testable core of [`project_folder`](Self::project_folder) that takes an
    /// explicit projects directory (avoids env mutation under
    /// `forbid(unsafe_code)`).
    fn project_folder_for(projects_dir: &Path, workspace: Option<&Path>) -> Option<String> {
        let basename = workspace
            .and_then(|w| w.file_name())
            .and_then(|n| n.to_str())
            .filter(|s| !s.is_empty())?;
        let prefix = format!("{basename}-");

        // Reuse an existing `<basename>-*` folder, preferring the most-recently
        // modified one (the folder Qoder already uses for this workspace).
        if let Ok(entries) = std::fs::read_dir(projects_dir) {
            let mut best: Option<(std::time::SystemTime, String)> = None;
            for entry in entries.flatten() {
                let name = match entry.file_name().to_str() {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                if !name.starts_with(&prefix) {
                    continue;
                }
                let mtime = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                match &best {
                    Some((prev, _)) if *prev >= mtime => {}
                    _ => best = Some((mtime, name)),
                }
            }
            if let Some((_, name)) = best {
                return Some(name);
            }
        }

        // No existing folder — best-effort hash stand-in. Qoder's real hash
        // algorithm is undocumented, so the IDE may not auto-discover sessions
        // written for never-opened workspaces.
        let hash = workspace_hash(workspace.unwrap_or_else(|| Path::new("")));
        Some(format!("{basename}-{hash}"))
    }

    /// Core write logic against an explicit target path. Split out so the
    /// in-file unit tests can exercise it against a temp file without
    /// mutating the process environment (`unsafe` under `forbid(unsafe_code)`).
    fn write_to_target(
        &self,
        target: &Path,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let full_id = uuid::Uuid::new_v4().to_string();
        let bytes = build_jsonl(session);

        let outcome =
            crate::pipeline::atomic_write(target, &bytes, opts.force, "qoder")?;

        info!(
            session_id = %full_id,
            path = %outcome.target_path.display(),
            messages = session.messages.len(),
            "Qoder session written"
        );

        Ok(WrittenSession {
            paths: vec![outcome.target_path],
            session_id: full_id.clone(),
            resume_command: self.resume_command(&full_id),
            backup_path: outcome.backup_path,
        })
    }
}

impl Provider for Qoder {
    fn name(&self) -> &str {
        "Qoder"
    }

    fn slug(&self) -> &str {
        "qoder"
    }

    fn cli_alias(&self) -> &str {
        "qod"
    }

    fn detect(&self) -> DetectionResult {
        let mut evidence = Vec::new();
        let mut installed = false;

        if let Some(dir) = Self::projects_dir()
            && dir.is_dir()
        {
            evidence.push(format!("{} exists", dir.display()));
            installed = true;
        }

        if let Some(home) = Self::home_dir()
            && home.is_dir()
        {
            evidence.push(format!("{} exists", home.display()));
        }

        trace!(provider = "qoder", ?evidence, installed, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        match Self::projects_dir() {
            Some(dir) if dir.is_dir() => vec![dir],
            _ => vec![],
        }
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        let projects_dir = Self::projects_dir()?;
        if !projects_dir.is_dir() {
            return None;
        }
        let first8 = first8_of(session_id)?;
        // projects_dir/<project>/conversation-history/<first8>/<first8>.jsonl
        for proj in std::fs::read_dir(&projects_dir).ok()?.flatten() {
            if !proj.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let candidate = proj
                .path()
                .join("conversation-history")
                .join(&first8)
                .join(format!("{first8}.jsonl"));
            if candidate.is_file() {
                debug!(path = %candidate.display(), session_id, "found Qoder session");
                return Some(candidate);
            }
        }
        None
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let projects_dir = Self::projects_dir()?;
        if !projects_dir.is_dir() {
            return Some(vec![]);
        }
        let mut out = Vec::new();
        // projects_dir/<project>/conversation-history/<8hex>/<8hex>.jsonl
        for proj in std::fs::read_dir(&projects_dir).ok()?.flatten() {
            let proj_path = proj.path();
            if !proj_path.is_dir() {
                continue;
            }
            let history = proj_path.join("conversation-history");
            if !history.is_dir() {
                continue;
            }
            let Ok(sid_entries) = std::fs::read_dir(&history) else {
                continue;
            };
            for sid_dir in sid_entries.flatten() {
                let sid_path = sid_dir.path();
                if !sid_path.is_dir() {
                    continue;
                }
                let Some(name) = sid_path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let candidate = sid_path.join(format!("{name}.jsonl"));
                if candidate.is_file() {
                    out.push((name.to_string(), candidate));
                }
            }
        }
        Some(out)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading Qoder session");

        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);

        let session_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut messages: Vec<CanonicalMessage> = Vec::new();
        let mut line_num: usize = 0;
        let mut skipped: usize = 0;

        for line_result in reader.lines() {
            line_num += 1;
            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    warn!(line = line_num, error = %e, "skipping unreadable line");
                    skipped += 1;
                    continue;
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    warn!(line = line_num, error = %e, "skipping malformed JSON line");
                    skipped += 1;
                    continue;
                }
            };

            let role_str = entry.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let role = normalize_role(role_str);

            let content_val = entry.pointer("/message/content");
            let content = content_val
                .map(flatten_content)
                .unwrap_or_default();

            // Qoder stores no tool blocks; skip lines with no recoverable text.
            if content.trim().is_empty() {
                trace!(line = line_num, ?role, "skipping empty Qoder line");
                continue;
            }

            messages.push(CanonicalMessage {
                idx: 0, // reindexed below
                role,
                content,
                timestamp: None,
                author: None,
                tool_calls: Vec::new(),
                tool_results: Vec::new(),
                extra: entry,
            });
        }

        reindex_messages(&mut messages);

        let title = messages
            .iter()
            .find(|m| m.role == MessageRole::User)
            .map(|m| truncate_title(&m.content, 100));

        debug!(
            session_id,
            messages = messages.len(),
            skipped,
            "Qoder session parsed"
        );

        Ok(CanonicalSession {
            session_id,
            provider_slug: "qoder".to_string(),
            workspace: None,
            title,
            started_at: None,
            ended_at: None,
            messages,
            metadata: serde_json::json!({ "source": "qoder" }),
            source_path: path.to_path_buf(),
            model_name: None,
        })
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let projects_dir = Self::projects_dir()
            .ok_or_else(|| anyhow::anyhow!("cannot determine Qoder projects directory"))?;

        let full_id = uuid::Uuid::new_v4();
        let first8 = first8_of(&full_id.to_string())
            .ok_or_else(|| anyhow::anyhow!("generated Qoder session id too short"))?;

        let folder = Self::project_folder(session.workspace.as_deref())
            .ok_or_else(|| anyhow::anyhow!("cannot derive Qoder project folder"))?;

        let target = projects_dir
            .join(&folder)
            .join("conversation-history")
            .join(&first8)
            .join(format!("{first8}.jsonl"));

        debug!(
            target = %target.display(),
            folder,
            first8,
            "writing Qoder session"
        );

        self.write_to_target(&target, session, opts)
    }

    fn resume_command(&self, session_id: &str) -> String {
        format!("qoder --resume {session_id}")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// First 8 hex characters of a (hex) session id, if long enough.
fn first8_of(id: &str) -> Option<String> {
    let hex: String = id
        .trim()
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .take(8)
        .collect();
    if hex.len() == 8 {
        Some(hex.to_ascii_lowercase())
    } else {
        None
    }
}

/// Best-effort 8-hex hash of a workspace path (Qoder's real algorithm is
/// undocumented). Uses `sha256` (the available hashing crate) of the
/// lower-cased path string.
fn workspace_hash(workspace: &Path) -> String {
    let normalized = workspace.to_string_lossy().to_ascii_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let result = hasher.finalize();
    result.iter().take(4).map(|b| format!("{:02x}", b)).collect()
}

/// Build the Qoder JSONL byte content for a canonical session.
///
/// Each line is `{"role":"user|assistant","message":{"content":[{"type":"text","text":…}]}}`.
/// `Tool`/`System`/`Other` roles are lossy: dropped when empty, otherwise
/// written as `assistant` with a `[tool] `/`[system] `/`[other] ` prefix
/// (documented limitation — Qoder has no tool/system role).
fn build_jsonl(session: &CanonicalSession) -> Vec<u8> {
    let mut lines: Vec<String> = Vec::with_capacity(session.messages.len());
    for msg in &session.messages {
        let (role_str, text) = match &msg.role {
            MessageRole::User => ("user", msg.content.clone()),
            MessageRole::Assistant => ("assistant", msg.content.clone()),
            MessageRole::Tool => {
                if msg.content.trim().is_empty() {
                    continue;
                }
                ("assistant", format!("[tool] {}", msg.content))
            }
            MessageRole::System => {
                if msg.content.trim().is_empty() {
                    continue;
                }
                ("assistant", format!("[system] {}", msg.content))
            }
            MessageRole::Other(_) => {
                if msg.content.trim().is_empty() {
                    continue;
                }
                ("assistant", format!("[other] {}", msg.content))
            }
        };
        let entry = serde_json::json!({
            "role": role_str,
            "message": {
                "content": [{ "type": "text", "text": text }]
            }
        });
        lines.push(serde_json::to_string(&entry).unwrap_or_default());
    }
    let mut content = lines.join("\n");
    if !content.is_empty() {
        content.push('\n');
    }
    content.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CanonicalMessage, CanonicalSession, MessageRole};
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
    // Pure helpers
    // -----------------------------------------------------------------------

    #[test]
    fn first8_of_uuid_extracts_eight_hex() {
        let id = "2e9d924b-1234-5678-9abc-def012345678";
        assert_eq!(first8_of(id).as_deref(), Some("2e9d924b"));
    }

    #[test]
    fn first8_of_short_id_returns_none() {
        assert_eq!(first8_of("abc"), None);
    }

    #[test]
    fn workspace_hash_is_eight_lowercase_hex() {
        let h = workspace_hash(Path::new("/data/projects/myapp"));
        assert_eq!(h.len(), 8, "hash must be 8 hex chars");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(h, h.to_ascii_lowercase());
    }

    #[test]
    fn project_folder_reuses_existing_basename_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let projects = tmp.path().join("projects");
        std::fs::create_dir_all(projects.join("myapp-deadbeef")).unwrap();
        std::fs::create_dir_all(projects.join("myapp-cafef00d")).unwrap();
        // Touch the second one later so it wins recency.
        std::fs::write(projects.join("myapp-cafef00d").join("marker"), "x").unwrap();

        let folder = Qoder::project_folder_for(&projects, Some(Path::new("/data/projects/myapp")))
            .expect("folder should resolve");
        assert!(
            folder == "myapp-deadbeef" || folder == "myapp-cafef00d",
            "should reuse an existing myapp-* folder, got {folder}"
        );
    }

    #[test]
    fn project_folder_creates_hash_folder_when_none_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let projects = tmp.path().join("projects");
        std::fs::create_dir_all(&projects).unwrap();

        let folder = Qoder::project_folder_for(&projects, Some(Path::new("/data/projects/myapp")))
            .expect("folder should resolve");
        assert!(
            folder.starts_with("myapp-"),
            "should derive a myapp-<hash> folder, got {folder}"
        );
        assert_eq!(folder.len(), "myapp-".len() + 8);
    }

    // -----------------------------------------------------------------------
    // Write → read round-trip against a temp file (no env mutation)
    // -----------------------------------------------------------------------

    #[test]
    fn write_then_read_round_trips_text_messages() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("abc12345").join("abc12345.jsonl");

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

        let written = Qoder
            .write_to_target(&target, &original, &WriteOptions { force: false })
            .expect("write should succeed");

        assert_eq!(written.paths, vec![target.clone()]);
        assert!(!written.session_id.is_empty());
        assert!(written.resume_command.starts_with("qoder --resume "));

        let readback = Qoder
            .read_session(&target)
            .expect("read-back should succeed");

        assert_eq!(readback.provider_slug, "qoder");
        assert_eq!(readback.session_id, "abc12345");
        assert_eq!(readback.messages.len(), original.messages.len());
        for (i, (orig, rb)) in original.messages.iter().zip(readback.messages.iter()).enumerate() {
            assert_eq!(orig.role, rb.role, "msg {i}: role mismatch");
            assert_eq!(orig.content, rb.content, "msg {i}: content mismatch");
        }
        // Qoder stores no timestamps.
        assert!(readback.messages.iter().all(|m| m.timestamp.is_none()));
    }

    #[test]
    fn build_jsonl_drops_empty_tool_and_system_messages() {
        let session = sample_session(vec![
            sample_message(MessageRole::User, "hello"),
            sample_message(MessageRole::Tool, ""),
            sample_message(MessageRole::System, ""),
            sample_message(MessageRole::Assistant, "hi"),
        ]);
        let bytes = build_jsonl(&session);
        let text = String::from_utf8(bytes).unwrap();
        let count = text.lines().filter(|l| !l.trim().is_empty()).count();
        assert_eq!(count, 2, "empty tool/system lines should be dropped");
        // Non-empty system/tool get prefixed.
        let session2 = sample_session(vec![
            sample_message(MessageRole::Tool, "tool output"),
            sample_message(MessageRole::System, "sys msg"),
        ]);
        let text2 = String::from_utf8(build_jsonl(&session2)).unwrap();
        assert!(text2.contains("[tool] tool output"));
        assert!(text2.contains("[system] sys msg"));
    }

    #[test]
    fn read_session_parses_real_record_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("2e9d924b").join("2e9d924b.jsonl");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(
            &target,
            "{\"role\":\"user\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"hello world\"}]}}\n",
        )
        .unwrap();

        let session = Qoder.read_session(&target).expect("read should succeed");
        assert_eq!(session.session_id, "2e9d924b");
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[0].content, "hello world");
        assert_eq!(session.title.as_deref(), Some("hello world"));
    }
}
