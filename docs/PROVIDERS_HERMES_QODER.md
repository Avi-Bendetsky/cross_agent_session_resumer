# Hermes + Qoder Provider Limitations

This document records known limitations of the Hermes and Qoder providers
added to `casr` (cross_agent_session_resumer).

## 1. Qoder project-folder hash is best-effort

Qoder organises transcripts under
`~/.qoder/cache/projects/<basename>-<hash>/conversation-history/...`.

The `<hash>` portion is derived from the workspace path by Qoder's own
(undocumented) algorithm.  The casr Qoder provider handles this in two ways:

- **Reuses an existing folder**: When writing a session for a workspace whose
  basename matches an existing `<basename>-*` folder in `projects/`, casr
  reuses that folder.  This works correctly for workspaces already opened in
  Qoder.
- **Creates a new folder**: When no matching folder exists, casr computes a
  best-effort hash using `sha256(normalized_path)[:8]` (the `md5` crate is not
  a dependency, so SHA-256 is used as a documented stand-in).  Qoder may not
  auto-discover sessions written to such a folder because Qoder's own hash
  algorithm may differ.

## 2. Qoder transcript format is lossy

The Qoder JSONL transcript format stores only:

```json
{"role":"user"|"assistant","message":{"content":[{"type":"text","text":"..."}]}}
```

The following fields are **not stored** and are therefore lost when converting
to/from Qoder:

- **Timestamps** — Qoder transcripts carry no per-message timestamps.
  Conversions from providers that do have timestamps (e.g. Claude Code)
  will lose them on the Qoder side.
- **Tool blocks** — Qoder has no `tool` role or tool-call/tool-result
  structures.  Messages with `Tool` role are dropped if their content is
  empty, or written as `"assistant"` with a `[tool] ` content prefix.
- **System/Other roles** — Same treatment as Tool: dropped if empty, or
  written as `"assistant"` with `[system] `/`[other] ` prefix.
- **Metadata** — No model name, git branch, token usage, or other metadata
  is stored.

## 3. Hermes resume command is best-effort

The `resume_command` for Hermes is `hermes --resume <session-id>`.  This is
a best-effort guess based on common CLI patterns.  The exact resume
invocation should be confirmed against the installed `hermes-agent` CLI.

## 4. Qoder `state.vscdb` UI entries are not populated

Qoder stores UI state (chat tabs, chat views) in a `state.vscdb` SQLite
database under keys like `aicoding.chat.tabs` and `aicoding.chat.views`.
The casr Qoder provider does **not** populate these entries — the JSONL
transcript file is the only durable store that casr reads and writes.
Sessions created by casr will appear in the transcript directory but may
not appear in Qoder's UI sidebar until Qoder itself discovers them.

## 5. Hermes timestamp units

Hermes stores `started_at`, `ended_at`, and per-message `timestamp` as
`REAL` (floating-point) **epoch seconds** in the SQLite database.  The casr
canonical model uses `i64` **epoch milliseconds**.  The Hermes provider
converts on read (`* 1000.0` → `i64`) and on write (`/ 1000.0` → `REAL`).

## 6. Hermes `active` flag filtering

The Hermes `messages` table has an `active INTEGER NOT NULL DEFAULT 1`
column.  Messages with `active = 0` are marked as rewound/compressed.
The casr Hermes provider reads only `active = 1` messages (the live
conversation view), matching the behaviour of the Hermes desktop UI.

## 7. Hermes FTS tables are trigger-driven

The `messages_fts` and related full-text search tables in the Hermes
database are populated by `AFTER INSERT` triggers.  The casr Hermes
provider never writes to FTS tables directly.  Sessions created by casr
will have FTS entries populated automatically by the existing triggers
when the DB is opened by Hermes.

## 8. Hermes multi-session DB and virtual paths

A single Hermes `state.db` contains multiple sessions.  When `casr list`
enumerates sessions, it returns virtual paths of the form
`<db_path>/<session_id>` so that `read_session` can resolve the specific
session.  Passing the raw `state.db` path to `read_session` (or
`casr info`) will read the most recently started session, which is the
correct fallback for single-session DBs.
