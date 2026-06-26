# git_to_doc

> Turn a raw `git diff` into a Conventional Commit message and a Markdown changelog entry — powered by a local Gemma model, no API keys, no cloud.

```bash
python git_to_doc.py changes.diff
```

```
① CONVENTIONAL COMMIT
────────────────────────────────────────────────────────────
feat(auth): add scope-based access control to token validation

Extend TokenManager.generate() to accept an optional list of OAuth-style
scopes stored alongside each token. validate() now accepts a required_scope
parameter and returns None when the token lacks the requested permission.
Also fixes naive datetime.utcnow() calls to use timezone-aware equivalents.

② README CHANGELOG SNIPPET
────────────────────────────────────────────────────────────
### [Unreleased] — 2025-07-12

#### Added
- Scope-based access control: tokens now carry an optional list of permission scopes
- `require_auth` middleware decorator accepts a `scope` argument for route-level enforcement

#### Changed
- `TokenManager.generate()` now returns a metadata dict (`token`, `expires_at`, `scopes`)
- Token generation uses a cryptographic nonce via `secrets.token_hex` for replay protection

#### Fixed
- `middleware.require_auth` now logs auth failures with client IP for auditability
```

---

## What it actually does

Developers write terrible commit messages. They skip changelog updates. `git_to_doc` fixes both in one command by routing your diff through a local [Google Gemma](https://ai.google.dev/gemma) model and returning two structured, ready-to-paste outputs:

- A **Conventional Commit message** (`feat(scope): summary` + body) following the [Conventional Commits](https://www.conventionalcommits.org/) spec
- A **Markdown changelog snippet** in [Keep a Changelog](https://keepachangelog.com/) format, with `Added / Changed / Fixed / Removed` subsections

Everything runs locally via [Ollama](https://ollama.com). Your code never leaves your machine.

---

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) running locally
- A pulled Gemma model (see setup below)
- `rich` (optional — enables pretty terminal output)

---

## Setup

**1. Install Ollama**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download the installer at https://ollama.com
```

**2. Pull a Gemma model**

```bash
ollama pull gemma3:4b        # recommended — best speed/quality balance (~2.5 GB)
ollama pull gemma3:1b        # fastest, lowest memory footprint (~800 MB)
ollama pull gemma3:12b       # highest quality, requires ~8 GB RAM
```

**3. Install optional dependency**

```bash
pip install rich              # optional — enables syntax highlighting and panels
```

**4. Confirm everything works**

```bash
ollama serve                  # keep this running in a separate terminal
python git_to_doc.py sample.diff
```

---

## Usage

```
python git_to_doc.py <diff_file> [options]

positional arguments:
  diff_file          Path to a .diff or .txt file containing a raw git diff

options:
  --model MODEL      Ollama model name  (default: gemma3:4b)
  --output FILE      Save output to a Markdown file instead of printing
  --json             Emit JSON  {commit_message, changelog, stats}  for scripting
  -h, --help         Show this help message
```

### Examples

```bash
# Basic usage — print to terminal
python git_to_doc.py changes.diff

# Use the lightweight 1B model
python git_to_doc.py changes.diff --model gemma3:1b

# Save output to a file
python git_to_doc.py changes.diff --output entry.md

# Machine-readable JSON — pipe into jq, CI scripts, etc.
python git_to_doc.py changes.diff --json | jq .commit_message

# Generate a diff on the fly and pass it straight in
git diff HEAD~1 > /tmp/latest.diff && python git_to_doc.py /tmp/latest.diff

# Staged changes only
git diff --cached > staged.diff && python git_to_doc.py staged.diff
```

---

## Generating diff files

```bash
# Last commit vs current HEAD
git diff HEAD~1 HEAD > my_change.diff

# Everything staged but not yet committed
git diff --cached > staged.diff

# A single file's history
git diff HEAD~1 -- src/auth/token.py > token_change.diff

# Difference between two branches
git diff main..feature/my-branch > feature.diff
```

---

## How it works

```
your_change.diff
      │
      ▼
  load_diff()       — reads file, validates, truncates at 6,000 chars if needed
      │
      ├──► call_gemma(COMMIT_PROMPT)    ──► Ollama API ──► Gemma model
      │                                                          │
      └──► call_gemma(CHANGELOG_PROMPT) ──► Ollama API ──► Gemma model
                  │                               │
                  ▼                               ▼
          commit message                  changelog markdown
                  │                               │
                  └──────────────┬────────────────┘
                                 ▼
                    stdout  /  --output  /  --json
```

Two separate model calls are made intentionally — one focused prompt per output produces better results than a single combined prompt trying to do both jobs.

**Key design decisions:**

| Decision | Why |
|---|---|
| `temperature: 0.2` | Low randomness gives consistent, structured output every run |
| Truncate at 6,000 chars | Keeps diffs within Gemma 4B's context window safely |
| Two prompts, not one | Separation of concerns; each prompt has a single job |
| stdlib only (`urllib`) | Zero required dependencies — works out of the box |
| `--json` flag | Makes the tool composable with CI pipelines and other scripts |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Cannot reach Ollama at localhost:11434` | Run `ollama serve` in a separate terminal |
| `model not found` | Run `ollama pull gemma3:4b` |
| Output is blank or conversational filler | Lower temperature is already set; try `--model gemma3:12b` for better instruction-following |
| Diff truncated warning | Split your diff into per-feature files; only the first 6,000 chars are sent |
| No GPU / low RAM | `gemma3:1b` runs fine on CPU — slower but functional |

---

## Extension ideas

- `--watch` mode — poll `git diff` every 30 seconds and auto-document as you code
- `--append` — append the changelog entry directly to `CHANGELOG.md` in place
- Batch mode — process a directory of `.diff` files in one pass
- GitHub Action — post the generated commit message as a PR comment automatically
- Git alias — wrap the script so `git doc` runs it on your latest diff
