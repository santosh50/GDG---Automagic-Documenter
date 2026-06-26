# Automagic Documenter (`git_to_doc`)

> Turn a raw `git diff` into a Conventional Commit message **and** a Markdown changelog entry — grounded by a deterministic analyzer and written by a local Google Gemma model. No API keys, no cloud, your code never leaves your machine.

```bash
python git_to_doc.py changes.diff
```

```
⊙ STRUCTURAL INSIGHTS
────────────────────────────────────────────────────────────
  • 9 file(s) touched across source, test (+146 / -83).
  • New public API surface: handleSidebarToggle.
  • Deterministic type: feat (model will confirm).

① CONVENTIONAL COMMIT
────────────────────────────────────────────────────────────
feat(admin): introduce navigation sidebar and related CSS/JS

This change introduces a new navigation sidebar within the Django admin
interface. The sidebar is implemented with updated JavaScript and CSS,
enhancing access to different sections of the admin panel. Changes also
include modifications to options.py and updates to base and changelist
templates, plus responsive CSS fixes to prevent horizontal scrolling.

② README CHANGELOG SNIPPET
────────────────────────────────────────────────────────────
### Unreleased

#### Added or Changed
- Added a `sidebar.css` stylesheet for the navigation sidebar
- Added a JavaScript file to handle the sidebar toggle functionality
- Modified `options.py` to include the new CSS and JS references
- Updated `base.html` to conditionally load the sidebar assets
- Updated tests in `tests/admin_views/tests.py` for the new sidebar
```

---

## What it actually does

Developers write terrible commit messages and skip changelog updates. `git_to_doc` fixes both in one command — but unlike a naive "throw the diff at an LLM" tool, it never lets the model see the raw diff cold.

A **deterministic analyzer** parses the diff's *structure* first — files, change kinds, languages, added/removed symbols, dependency changes, breaking-change signals — resolves everything it can on its own, and hands Gemma a clean, fact-dense **structured digest**. This anchoring is what keeps small local models from hallucinating features that aren't there.

You get three outputs:

- **Structural insights** — a deterministic summary of what changed (files, categories, new/removed API, breaking signals)
- A **Conventional Commit message** (`feat(scope): summary` + body) following the [Conventional Commits](https://www.conventionalcommits.org/) spec — written by Gemma, then **repaired by a deterministic validator** so the header is *always* spec-compliant
- A **Markdown changelog snippet** in the [Best-README-Template](https://github.com/othneildrew/Best-README-Template) style (`Added or Changed` / `Removed`), with a fully grounded deterministic fallback if the model under-produces

Optionally, it can also render a **PNG chart** of the change structure (additions vs. deletions by category and by file).

Everything runs locally via [Ollama](https://ollama.com).

---

## How accuracy is enforced

This is the core of the design — accuracy over fluency:

| Stage | Who does it | Why |
|---|---|---|
| Parse diff → files, symbols, categories | Deterministic | Facts, not guesses |
| Classify change type (feat/fix/docs/…) | Deterministic, with a confidence flag | Single-category changes (docs-only, test-only) are resolved without the model |
| Write commit narrative | Gemma | Human judgement helps here |
| Repair commit header | Deterministic | Guarantees valid `<type>(<scope>): <summary>`, ≤72 chars, no fabricated `BREAKING CHANGE` footers |
| Write changelog bullets | Gemma | Descriptive, file-aware prose |
| Fallback changelog | Deterministic | Zero hallucination, never empty or truncated |

The analyzer is **authoritative** on breaking changes and change type — the model is given a strong prior but its drift is corrected after the fact.

---

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) running locally
- A pulled Gemma model (see setup below)
- `rich` *(optional)* — pretty terminal panels
- `matplotlib` *(optional)* — only needed for `--chart`

The core tool is **pure standard library** (`urllib`) — no `pip install` required to run it.

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
ollama pull gemma3:12b       # default — hallucinates far less (~8 GB RAM)
ollama pull gemma3:4b        # faster, smaller (~2.5 GB), good balance
ollama pull gemma3:1b        # fastest, lowest memory (~800 MB) — riskiest accuracy
```

**3. Install optional dependencies**

```bash
pip install rich             # prettier terminal output
pip install matplotlib       # required only for --chart
```

**4. Confirm everything works**

```bash
ollama serve                 # keep running in a separate terminal
python git_to_doc.py sample_diff/django_docs.diff
```

---

## Usage

```
python git_to_doc.py <diff_file> [options]

positional arguments:
  diff_file                  Path to a .diff or .txt file containing a raw git diff

options:
  --model MODEL              Ollama model name (default: gemma3:12b)
  --output FILE              Save full output (insights + commit + changelog) to a .md file
  --json                     Emit machine-readable JSON for scripting
  --version LABEL            Version label for the changelog (default: Unreleased; e.g. v1.2.0)
  --deterministic-changelog  Skip Gemma for the changelog; build it purely from parsed facts
  --chart FILE.png           Render a PNG of the change structure (requires matplotlib)
  -h, --help                 Show this help message
```

### Examples

```bash
# Basic usage — print to terminal
python git_to_doc.py changes.diff

# Use the lightweight model for speed
python git_to_doc.py changes.diff --model gemma3:4b

# Tag the changelog entry with a real version
python git_to_doc.py changes.diff --version v1.2.0

# Save the full report to a file
python git_to_doc.py changes.diff --output entry.md

# Zero-hallucination changelog built purely from parsed facts
python git_to_doc.py changes.diff --deterministic-changelog

# Render a visual breakdown of the change
python git_to_doc.py changes.diff --chart change.png

# Machine-readable JSON — pipe into jq, CI scripts, etc.
python git_to_doc.py changes.diff --json | jq .commit_message

# Generate a diff on the fly and pass it straight in
git diff HEAD~1 > latest.diff && python git_to_doc.py latest.diff
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

Sample diffs from real open-source PRs are included in [`sample_diff/`](sample_diff/) to try the tool out immediately.

---

## How it works

```
your_change.diff
      │
      ▼
  parse_diff()      — raw unified diff → structured FileChange list
      │                (kinds, languages, categories, +/- symbols, binaries)
      ▼
  analyze()         — deterministic insight engine
      │                (scopes, dependency changes, type guess, breaking signal)
      ▼
  build_digest()    — fact-dense summary + budgeted raw diff (≤ 6,000 chars)
      │
      ├──► call_gemma(COMMIT_PROMPT)    ──► Ollama ──► Gemma
      │            │                                     │
      │            ▼                                     ▼
      │     enforce_commit_format()  ◄──────── narrative commit message
      │            │  (deterministic header repair)
      │            ▼
      │     spec-compliant commit
      │
      └──► call_gemma(CHANGELOG_PROMPT) ──► Ollama ──► Gemma
                   │                                     │
                   ▼                                     ▼
            enforce_changelog()  ◄──────────── changelog bullets
                   │  (drops empty sections; falls back to
                   │   render_changelog() if under-produced)
                   ▼
        stdout  /  --output  /  --json  /  --chart
```

Two separate model calls are made intentionally — one focused prompt per output produces better results than a single combined prompt trying to do both jobs.

**Key design decisions:**

| Decision | Why |
|---|---|
| Analyzer runs *before* the model | The model gets grounded facts, not a raw diff — kills hallucination |
| Deterministic header repair | Commit header is *always* valid, no matter what the model emits |
| Deterministic changelog fallback | Output is never empty, truncated, or malformed |
| Analyzer is authoritative on breaking changes | Small models love to invent `BREAKING CHANGE:` footers — these get stripped |
| `temperature: 0.15` + fixed `seed` | Consistent, reproducible output for the same diff |
| Budget raw diff at 6,000 chars | Drops large hunk *bodies* first while always keeping every file's header skeleton |
| stdlib only (`urllib`) | Zero required dependencies — works out of the box |
| `--json` flag | Composable with CI pipelines and other scripts |

---

## Output to file

`--output entry.md` writes a complete report — structural insights, the commit message in a code block, and the changelog snippet — ready to drop into a PR description or `CHANGELOG.md`.

`--json` emits `{ commit_message, changelog, analysis }`, where `analysis` includes the per-file breakdown, totals, inferred scopes, dependency changes, type guess (with confidence), breaking-change flag, and human-readable insights.

---

## Visualizing a change (`change_chart.py`)

`change_chart.py` renders a PNG showing *where* a change landed: diverging additions/deletions bars grouped **by category** and the **heaviest individual files**. It consumes the same `DiffAnalysis` the main tool produces, so it never re-parses anything.

```bash
# Via the main tool
python git_to_doc.py changes.diff --chart change.png

# Standalone
python change_chart.py changes.diff change.png
```

Requires `matplotlib` (`pip install matplotlib`).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Cannot reach Ollama at localhost:11434` | Run `ollama serve` in a separate terminal |
| `model not found` | Run `ollama pull gemma3:12b` (or the model you passed to `--model`) |
| Output is blank or conversational filler | Already mitigated by low temperature + sanitization; try the larger `gemma3:12b` |
| Changelog looks thin or generic | Use `--deterministic-changelog` for a fully grounded, fact-based version |
| `bodies of N large file(s) omitted` note | Expected for huge diffs — only the first 6,000 chars of raw bodies are sent; structure is still complete |
| `--chart requires matplotlib` | `pip install matplotlib` |
| No GPU / low RAM | `gemma3:1b` or `gemma3:4b` run on CPU — slower, and 1b is less accurate |
