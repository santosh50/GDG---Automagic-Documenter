#!/usr/bin/env python3
"""
git_to_doc_v2.py — Automagic Documenter (accuracy engine)
=========================================================
Takes a raw git diff (.diff or .txt) and routes it through a local
Google Gemma model (via Ollama) to produce:
  1. A Conventional Commit message
  2. A Markdown changelog snippet

The model never sees the raw diff cold. First a deterministic analyzer
parses the diff's *structure* — files, change kinds, languages, added/
removed symbols, dependency changes, breaking-change signals — resolves
everything it can on its own, and hands Gemma a clean structured digest.
The model is only asked to do what it is actually good at: write short
prose for the parts that genuinely need judgement.

Usage:
  python git_to_doc_v2.py <path/to/file.diff>
  python git_to_doc_v2.py <path/to/file.diff> --output changelog.md
  python git_to_doc_v2.py <path/to/file.diff> --model gemma3:4b
  python git_to_doc_v2.py <path/to/file.diff> --json

Requirements:
  # Pure standard library — no pip install required.
  # `rich` is optional and only enables prettier terminal output.
  # Ollama running locally: https://ollama.com
  # Pull a Gemma model: ollama pull gemma3:4b
"""

from __future__ import annotations

import sys
import re
import json
import argparse
import textwrap
import urllib.request
import urllib.error
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field, asdict
from collections import Counter

# ── optional: pretty terminal output ────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    RICH = True
    console = Console()
except ImportError:
    RICH = False

# ── Ollama endpoint ──────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"  # gemma3:1b for speed, gemma3:12b for quality

# ── budget: how much raw diff detail we send alongside the structured digest ──
MAX_RAW_DIFF_CHARS = 6_000


# ═════════════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION TABLES
# ═════════════════════════════════════════════════════════════════════════════

# extension → human language label
LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin",
    ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".swift": "swift", ".scala": "scala", ".sh": "shell",
    ".css": "css", ".scss": "css", ".html": "html",
    ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".sql": "sql",
}

DOCS_EXT = {".md", ".rst", ".txt", ".adoc"}

# dependency / lockfile manifests — a change here is almost always build/chore
DEP_FILES = {
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py",
    "setup.cfg", "pipfile", "pipfile.lock", "poetry.lock",
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.mod", "go.sum", "cargo.toml", "cargo.lock", "gemfile", "gemfile.lock",
    "composer.json", "composer.lock", "build.gradle", "pom.xml",
}

BUILD_FILES = {
    "dockerfile", "makefile", "rakefile", ".dockerignore",
    "tox.ini", "noxfile.py", "manifest.in",
}


def classify_file(path: str) -> tuple[str, str]:
    """Return (language, category) for a path.

    category ∈ {docs, test, ci, build, config, source, other}
    """
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    ext = ""
    if "." in name:
        ext = name[name.rfind("."):]

    language = LANG_BY_EXT.get(ext, "other")

    # example / demo / benchmark code — real source, but NOT public API, so a
    # change here (even a deletion) must never count as a feature or a break.
    if any(seg in {"example", "examples", "demo", "demos", "sample", "samples",
                   "benchmark", "benchmarks"} for seg in ("/" + p).split("/")):
        return language, "example"

    # CI configuration
    if "/.github/workflows/" in "/" + p or p.startswith(".github/workflows/"):
        return language, "ci"
    if name in {".gitlab-ci.yml", ".travis.yml", ".circleci", "azure-pipelines.yml"}:
        return language, "ci"
    if "/.circleci/" in "/" + p:
        return language, "ci"

    # dependency / build tooling
    if name in DEP_FILES:
        return language, "build"
    if name in BUILD_FILES:
        return language, "build"

    # tests — match test *directories* or test-named *files*, never a bare
    # substring (so "flask/testing.py" and "docs/testing.rst" are NOT tests).
    segments = ("/" + p).split("/")
    in_test_dir = any(seg in {"test", "tests", "__tests__", "spec"} for seg in segments)
    test_named = (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("_test.go")
        or name.endswith((".test.js", ".spec.js", ".test.jsx", ".spec.jsx"))
        or name.endswith((".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx"))
    )
    if in_test_dir or test_named:
        return language, "test"

    # docs
    if ext in DOCS_EXT or "/docs/" in "/" + p or p.startswith("docs/"):
        return language, "docs"

    # other config
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return language, "config"

    if language != "other":
        return language, "source"

    return language, "other"


# ═════════════════════════════════════════════════════════════════════════════
# 2. SYMBOL EXTRACTION  (what functions / classes appeared or vanished)
# ═════════════════════════════════════════════════════════════════════════════

# Per-language regexes that capture a declaration's *name* from a line of code.
SYMBOL_PATTERNS = {
    "python": [
        re.compile(r"^\s*(?:async\s+)?def\s+([a-zA-Z_]\w*)"),
        re.compile(r"^\s*class\s+([a-zA-Z_]\w*)"),
    ],
    "javascript": [
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$][\w$]*)"),
        re.compile(r"^\s*(?:export\s+)?class\s+([a-zA-Z_$][\w$]*)"),
        re.compile(r"^\s*(?:export\s+)?const\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s*)?\("),
    ],
    "go": [
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
        re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s"),
    ],
    "rust": [
        re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_]\w*)"),
        re.compile(r"^\s*(?:pub\s+)?struct\s+([a-zA-Z_]\w*)"),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected).*\s([a-zA-Z_]\w*)\s*\("),
        re.compile(r"^\s*(?:public\s+)?class\s+([a-zA-Z_]\w*)"),
    ],
}
# typescript reuses the javascript patterns
SYMBOL_PATTERNS["typescript"] = SYMBOL_PATTERNS["javascript"]


def _symbols_in(lines: list[str], language: str) -> set[str]:
    pats = SYMBOL_PATTERNS.get(language)
    if not pats:
        return set()
    found: set[str] = set()
    for ln in lines:
        for pat in pats:
            m = pat.match(ln)
            if m:
                found.add(m.group(1))
    return found


# ═════════════════════════════════════════════════════════════════════════════
# 3. DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FileChange:
    path: str
    kind: str               # added | deleted | modified | renamed
    language: str
    category: str           # docs | test | ci | build | config | source | example | other
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    old_path: str | None = None
    added_symbols: list[str] = field(default_factory=list)
    removed_symbols: list[str] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)   # full block, for budgeting
    header_lines: list[str] = field(default_factory=list)  # block minus hunk bodies


@dataclass
class DiffAnalysis:
    files: list[FileChange]
    total_additions: int
    total_deletions: int
    dep_changes: list[str]
    scopes: list[str]
    type_guess: str
    type_confident: bool
    breaking: bool
    insights: list[str]


# ═════════════════════════════════════════════════════════════════════════════
# 4. PARSER  — raw unified diff → list[FileChange]
# ═════════════════════════════════════════════════════════════════════════════

# Accept both prefixed (`a/x b/x`, the default) and unprefixed (`--no-prefix`,
# `diff.noprefix=true`) headers so we parse any unified git diff.
_DIFF_GIT = re.compile(r"^diff --git (.+?) (.+)$")
_HUNK = re.compile(r"^@@ .* @@")


def _strip_diff_prefix(path: str, side: str) -> str:
    """Drop a leading a/ or b/ (or matching quoted form) from a diff path."""
    path = path.strip().strip('"')
    if path.startswith(f"{side}/"):
        return path[2:]
    return path


def parse_diff(text: str) -> list[FileChange]:
    files: list[FileChange] = []
    cur: FileChange | None = None
    added_code: list[str] = []
    removed_code: list[str] = []

    def finalize(fc: FileChange):
        if fc is None:
            return
        fc.added_symbols = sorted(_symbols_in(added_code, fc.language))
        fc.removed_symbols = sorted(_symbols_in(removed_code, fc.language))
        # symbols that exist on both sides are unchanged signatures, drop them
        common = set(fc.added_symbols) & set(fc.removed_symbols)
        fc.added_symbols = [s for s in fc.added_symbols if s not in common]
        fc.removed_symbols = [s for s in fc.removed_symbols if s not in common]
        files.append(fc)

    for line in text.splitlines():
        m = _DIFF_GIT.match(line)
        if m:
            if cur is not None:
                finalize(cur)
            added_code, removed_code = [], []
            new_path = _strip_diff_prefix(m.group(2), "b")
            old_path = _strip_diff_prefix(m.group(1), "a")
            lang, cat = classify_file(new_path)
            cur = FileChange(
                path=new_path, kind="modified", language=lang, category=cat,
                old_path=old_path, raw_lines=[line], header_lines=[line],
            )
            continue

        if cur is None:
            continue  # preamble before first file

        cur.raw_lines.append(line)

        # structural metadata lines belong in the header skeleton
        if line.startswith("new file mode"):
            cur.kind = "added"
            cur.header_lines.append(line)
        elif line.startswith("deleted file mode"):
            cur.kind = "deleted"
            cur.header_lines.append(line)
        elif line.startswith("rename from"):
            cur.kind = "renamed"
            cur.header_lines.append(line)
        elif line.startswith("rename to"):
            cur.header_lines.append(line)
        elif line.startswith("Binary files"):
            cur.binary = True
            cur.header_lines.append(line)
        elif line.startswith("index ") or line.startswith("--- ") or line.startswith("+++ "):
            cur.header_lines.append(line)
        elif _HUNK.match(line):
            cur.header_lines.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            cur.additions += 1
            added_code.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            cur.deletions += 1
            removed_code.append(line[1:])

    if cur is not None:
        finalize(cur)
    return files


# ═════════════════════════════════════════════════════════════════════════════
# 5. ANALYZER  — deterministic insight engine
# ═════════════════════════════════════════════════════════════════════════════

_SKIP_SCOPE_STEMS = {
    "changes", "changelog", "authors", "contributing",
    "readme", "license", "notice", "manifest",
}


def _infer_scope(files: list[FileChange]) -> list[str]:
    """Candidate scopes drawn from real paths, most common first.

    Prefer source/test files — a commit scope should name the code that
    changed, not the changelog or AUTHORS file that came along with it.
    """
    candidates = [f for f in files if f.category in {"source", "test"}] or files
    counter: Counter[str] = Counter()
    for fc in candidates:
        parts = fc.path.split("/")
        # prefer a meaningful directory; skip leading src/lib/app shells
        meaningful = [p for p in parts[:-1] if p not in {"src", "lib", "app", "."}]
        if meaningful:
            counter[meaningful[-1]] += 1
        else:
            stem = parts[-1].rsplit(".", 1)[0]
            if stem.lower() in _SKIP_SCOPE_STEMS:
                continue
            counter[stem] += 1
    return [name for name, _ in counter.most_common(3)]


def _detect_dependency_changes(files: list[FileChange]) -> list[str]:
    out = []
    for fc in files:
        name = fc.path.lower().rsplit("/", 1)[-1]
        if name in DEP_FILES:
            out.append(fc.path)
    return out


def analyze(files: list[FileChange]) -> DiffAnalysis:
    total_add = sum(f.additions for f in files)
    total_del = sum(f.deletions for f in files)
    cats = {f.category for f in files}
    dep_changes = _detect_dependency_changes(files)
    scopes = _infer_scope(files)

    # Type / breaking signals are driven by SOURCE files only — symbols that
    # appear in tests, docs or configs are not part of the public API surface
    # and must not push the change toward feat/breaking.
    src_files = [f for f in files if f.category == "source"]
    new_symbols = [s for f in src_files for s in f.added_symbols]
    gone_symbols = [s for f in src_files for s in f.removed_symbols]
    added_files = [f for f in files if f.kind == "added"]
    deleted_files = [f for f in files if f.kind == "deleted"]
    deleted_source = [f for f in deleted_files if f.category == "source"]

    # ── deterministic type resolution ───────────────────────────────────────
    type_guess, confident = _resolve_type(
        cats, files, new_symbols, gone_symbols, added_files
    )

    # ── breaking-change heuristic ────────────────────────────────────────────
    # Only a removed *source* file or a removed *public source* symbol breaks
    # callers. Deleting an image, a test, or a doc never does.
    breaking = bool(deleted_source) or bool(
        [s for s in gone_symbols if not s.startswith("_")]
    )

    # ── human-readable insights (surfaced in the UI, not just fed to model) ──
    insights: list[str] = []
    insights.append(
        f"{len(files)} file(s) touched across "
        f"{', '.join(sorted(cats)) or 'unknown'} "
        f"(+{total_add} / -{total_del})."
    )
    if added_files:
        insights.append(f"{len(added_files)} new file(s) added.")
    if deleted_files:
        insights.append(f"{len(deleted_files)} file(s) removed.")
    if new_symbols:
        shown = ", ".join(new_symbols[:6]) + ("…" if len(new_symbols) > 6 else "")
        insights.append(f"New public API surface: {shown}.")
    if gone_symbols:
        shown = ", ".join(gone_symbols[:6]) + ("…" if len(gone_symbols) > 6 else "")
        insights.append(f"Removed/renamed symbols: {shown}.")
    if dep_changes:
        insights.append(f"Dependency manifest changed: {', '.join(dep_changes)}.")
    if breaking:
        insights.append("⚠ Possible breaking change (public symbol or file removed).")
    insights.append(
        f"Deterministic type: {type_guess} "
        f"({'confident' if confident else 'model will confirm'})."
    )

    return DiffAnalysis(
        files=files,
        total_additions=total_add,
        total_deletions=total_del,
        dep_changes=dep_changes,
        scopes=scopes,
        type_guess=type_guess,
        type_confident=confident,
        breaking=breaking,
        insights=insights,
    )


def _resolve_type(cats, files, new_symbols, gone_symbols, added_files):
    """Return (type, confident). Confident types skip the model's judgement."""
    non_other = cats - {"other", "config"}

    # single-category changes are confidently classifiable
    if non_other == {"docs"} or cats == {"docs"}:
        return "docs", True
    if non_other == {"test"} or cats == {"test"}:
        return "test", True
    if non_other == {"ci"} or cats == {"ci"}:
        return "ci", True
    if non_other <= {"build"} and non_other:
        return "build", True
    if not non_other and cats <= {"config", "other"}:
        return "chore", True

    # everything below touches real source → needs semantic judgement.
    # we hand the model a strong prior but mark it non-confident.
    if added_files and new_symbols and not gone_symbols:
        return "feat", False           # new files + new API, nothing removed
    if new_symbols and not gone_symbols:
        return "feat", False
    return "fix", False                # conservative default for edits-in-place


# ═════════════════════════════════════════════════════════════════════════════
# 6. DIGEST BUILDER  — what the model actually reads
# ═════════════════════════════════════════════════════════════════════════════

def _budget_raw_diff(files: list[FileChange], limit: int) -> tuple[str, int, int]:
    """Fit raw diff under `limit` chars by dropping hunk *bodies* from the
    largest files first, while ALWAYS keeping every file's header skeleton.
    Returns (text, files_trimmed, lines_dropped)."""
    full = {id(f): "\n".join(f.raw_lines) for f in files}
    headers = {id(f): "\n".join(f.header_lines) for f in files}

    chosen = dict(full)
    total = sum(len(v) for v in chosen.values())
    trimmed = 0
    lines_dropped = 0

    # drop bodies from biggest files until we fit
    for f in sorted(files, key=lambda x: len(full[id(x)]), reverse=True):
        if total <= limit:
            break
        if chosen[id(f)] == headers[id(f)]:
            continue
        lines_dropped += len(f.raw_lines) - len(f.header_lines)
        total -= len(chosen[id(f)]) - len(headers[id(f)])
        chosen[id(f)] = headers[id(f)] + "\n    … [body omitted to fit context]"
        trimmed += 1

    text = "\n".join(chosen[id(f)] for f in files)
    if len(text) > limit:                      # absolute last-resort hard cut
        text = text[:limit] + "\n… [truncated]"
    return text, trimmed, lines_dropped


def build_digest(analysis: DiffAnalysis) -> str:
    """A clean structured summary + budgeted raw detail."""
    lines = ["=== STRUCTURED SUMMARY ==="]

    # CHANGE PROFILE — a fact-dense headline the model cannot ignore. This is
    # the deterministic anchor that keeps small models from free-associating.
    cat_counts = Counter(f.category for f in analysis.files)
    kind_counts = Counter(f.kind for f in analysis.files)
    profile = ", ".join(f"{n} {c}" for c, n in cat_counts.most_common())
    kinds = ", ".join(f"{n} {k}" for k, n in kind_counts.most_common())
    lines.append(
        f"CHANGE PROFILE: {len(analysis.files)} file(s) [{kinds}] "
        f"across categories: {profile}. "
        f"Net +{analysis.total_additions}/-{analysis.total_deletions} lines."
    )
    lines.append("")
    lines.append("Files changed:")
    for f in analysis.files:
        sym = ""
        if f.added_symbols:
            sym += f"  +symbols: {', '.join(f.added_symbols[:5])}"
        if f.removed_symbols:
            sym += f"  -symbols: {', '.join(f.removed_symbols[:5])}"
        bina = " [binary]" if f.binary else ""
        lines.append(
            f"  [{f.kind[:3].upper()}] {f.path} "
            f"({f.language}/{f.category}, +{f.additions} -{f.deletions}){bina}{sym}"
        )
    if analysis.dep_changes:
        lines.append(f"Dependency changes: {', '.join(analysis.dep_changes)}")
    lines.append(f"Candidate scopes: {', '.join(analysis.scopes) or 'n/a'}")
    lines.append(
        f"Deterministic type hint: {analysis.type_guess} "
        f"({'confident' if analysis.type_confident else 'unconfirmed'})"
    )
    if analysis.breaking:
        lines.append("Breaking-change signal: YES")

    raw, trimmed, dropped = _budget_raw_diff(analysis.files, MAX_RAW_DIFF_CHARS)
    lines.append("\n=== RAW DIFF (detail) ===")
    if trimmed:
        lines.append(
            f"[note: bodies of {trimmed} large file(s) omitted, "
            f"~{dropped} lines, to fit context — structure above is complete]"
        )
    lines.append(raw)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# 7. GEMMA via OLLAMA  (guarded, retried, sanitized)
# ═════════════════════════════════════════════════════════════════════════════

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?```\s*$")
_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is)|sure[,!]?|certainly[,!]?|below is)[^\n]*\n",
    re.IGNORECASE,
)


def sanitize(text: str) -> str:
    """Strip the junk small models add despite being told not to."""
    text = text.strip()
    # remove leading/trailing code fences (possibly repeated)
    prev = None
    while prev != text:
        prev = text
        text = _FENCE.sub("", text).strip()
    text = _PREAMBLE.sub("", text).strip()
    # strip wrapping quotes around a single-line answer
    if len(text) > 1 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def call_gemma(prompt: str, model: str, retries: int = 1) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low → deterministic, structured output
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "seed": 42,           # fixed seed → reproducible runs for the same diff
            "num_predict": 768,
        },
    }).encode()

    last_err = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
        except urllib.error.URLError:
            print(f"\n❌  Cannot reach Ollama at {OLLAMA_URL}", file=sys.stderr)
            print("    Start it with:  ollama serve", file=sys.stderr)
            print(f"    Pull the model: ollama pull {model}\n", file=sys.stderr)
            sys.exit(1)

        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            last_err = "Ollama returned a non-JSON response"
            continue

        out = sanitize(body.get("response", ""))
        if out:
            return out
        last_err = "model returned an empty response"

    print(f"⚠️   {last_err} after {retries + 1} attempt(s).", file=sys.stderr)
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# 8. PROMPTS  — note: model reads the DIGEST, never the raw diff alone
# ═════════════════════════════════════════════════════════════════════════════

COMMIT_PROMPT = """You are an expert software engineer writing ONE Conventional Commit message.

A deterministic analyzer has already parsed the change. The digest below is the
ONLY source of truth. Do not use any prior knowledge about this project.

GROUNDING RULES (most important):
- Describe ONLY what the digest shows. Never mention a file, function, feature,
  fix, or dependency that is not listed in the digest.
- If you are unsure what the code does, describe the change literally from the
  file list (e.g. "update N modules", "add <file>") rather than guessing intent.
- Do NOT add a "BREAKING CHANGE:" footer unless the digest contains the exact
  line "Breaking-change signal: YES".

FORMAT RULES:
- Header: <type>(<scope>): <summary in imperative mood, no period, <=72 chars>
- Valid types: feat, fix, docs, style, refactor, perf, test, chore, ci, build
- Use this type unless the digest clearly contradicts it: {type_hint}
- Choose <scope> from these candidates: {scopes}
- After the header, ONE blank line, then 2-4 sentences on WHAT changed and WHY,
  grounded entirely in the digest.
- Output ONLY the commit message. No code fences, no preamble.

{digest}
"""

CHANGELOG_PROMPT = """You are a technical writer producing a Markdown changelog entry.

A deterministic analyzer has parsed the change; the commit type is: {ctype}
The digest below is the ONLY source of truth. Do not use prior knowledge about
this project — if it is not in the digest, it did not happen.

GROUNDING RULES (most important):
- Every bullet MUST correspond to a file or symbol that actually appears in the
  digest. Never invent features, bug fixes, or "security" items.
- When many files changed the same way, write ONE summarising bullet
  (e.g. "Add type hints across 19 modules") instead of fabricating details.
- Only emit a #### Security bullet if the digest explicitly shows a security fix.

FORMAT RULES:
- First line exactly: ### [Unreleased] — {today}
- Then bullet lists under only the subsections that apply:
  #### Added  #### Changed  #### Fixed  #### Removed  #### Security
- Map intent honestly: feat->Added, fix->Fixed, refactor/perf->Changed, removals->Removed.
- Each bullet: present tense, plain English, <=100 chars.
- Output ONLY the Markdown. No code fences, no preamble.

{digest}
"""

VALID_TYPES = {
    "feat", "fix", "docs", "style", "refactor",
    "perf", "test", "chore", "ci", "build",
}
_HEADER_RE = re.compile(r"^([a-z]+)(?:\(([^)]+)\))?(!)?:\s*(.+)$")


def parse_commit_type(message: str) -> str | None:
    for line in message.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m and m.group(1) in VALID_TYPES:
            return m.group(1)
    return None


def _shorten_summary(summary: str, max_len: int) -> str:
    """Trim a summary to max_len chars on a word boundary, no trailing period."""
    summary = summary.strip().rstrip(".").strip()
    if len(summary) <= max_len:
        return summary
    cut = summary[:max_len]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(",;: ").rstrip()


def enforce_commit_format(message: str, analysis: "DiffAnalysis") -> str:
    """Guarantee a spec-compliant Conventional Commit header.

    The model usually gets this right, but small models drift: wrong/missing
    type, capitalised type, a scope that isn't real, an over-long header, a
    trailing period. We repair the *header* deterministically and keep the
    model's body verbatim — so the output is always valid no matter what.
    """
    HEADER_MAX = 72
    lines = message.splitlines()

    # locate the model's header line (first line that parses as one)
    idx = next((i for i, l in enumerate(lines) if _HEADER_RE.match(l.strip())), None)

    if idx is None:
        # no usable header — synthesise one from the analysis + first prose line
        ctype = analysis.type_guess
        scope = analysis.scopes[0] if analysis.scopes else ""
        body_first = next((l.strip() for l in lines if l.strip()), "")
        # drop a stray leading "Type:" the model may have emitted in prose
        body_first = re.sub(r"^[A-Za-z]+:\s*", "", body_first)
        summary = _shorten_summary(body_first or f"update {scope or 'code'}",
                                   HEADER_MAX - len(ctype) - len(scope) - 4)
        header = f"{ctype}({scope}): {summary}" if scope else f"{ctype}: {summary}"
        return header

    m = _HEADER_RE.match(lines[idx].strip())
    ctype, scope, bang, summary = m.group(1), m.group(2), m.group(3), m.group(4)

    # 1. type: lowercase; if invalid, fall back to the deterministic guess
    ctype = ctype.lower()
    if ctype not in VALID_TYPES:
        ctype = analysis.type_guess

    # 2. scope: lowercase, no spaces; accept the model's scope, else best candidate
    if scope:
        scope = re.sub(r"\s+", "", scope.lower())
    elif analysis.scopes:
        scope = analysis.scopes[0]

    # 3. breaking marker: the analyzer is AUTHORITATIVE. Small models love to
    #    invent "BREAKING CHANGE" footers — if no breaking signal was detected,
    #    strip both the ! marker and any fabricated footer from the body.
    bang = "!" if analysis.breaking else ""
    if not analysis.breaking:
        lines = [l for l in lines if not l.strip().upper().startswith("BREAKING CHANGE")]

    # 4. summary: no trailing period, fit the whole header under HEADER_MAX
    prefix = f"{ctype}({scope}){bang}: " if scope else f"{ctype}{bang}: "
    summary = _shorten_summary(summary, HEADER_MAX - len(prefix))
    header = prefix + summary

    lines[idx] = header
    return "\n".join(lines).strip()


def generate(analysis: DiffAnalysis, model: str, today: str) -> tuple[str, str]:
    digest = build_digest(analysis)
    scopes = ", ".join(analysis.scopes) or "(infer from paths)"

    # commit message — model confirms/overrides the deterministic type
    commit = call_gemma(
        COMMIT_PROMPT.format(type_hint=analysis.type_guess, scopes=scopes, digest=digest),
        model,
    )
    if not commit:  # total model failure → deterministic fallback header
        scope = analysis.scopes[0] if analysis.scopes else "core"
        commit = f"{analysis.type_guess}({scope}): update {scope}"

    # deterministic repair: guarantee a spec-compliant header no matter what
    commit = enforce_commit_format(commit, analysis)

    # single source of truth: the type that actually shipped in the header
    resolved = parse_commit_type(commit) or analysis.type_guess

    # changelog is told the resolved type so the two outputs never disagree
    changelog = call_gemma(
        CHANGELOG_PROMPT.format(ctype=resolved, today=today, digest=digest),
        model,
    )
    if not changelog:
        bucket = {"feat": "Added", "fix": "Fixed"}.get(resolved, "Changed")
        changelog = f"### [Unreleased] — {today}\n\n#### {bucket}\n- {commit.splitlines()[0]}"

    changelog = enforce_changelog_header(changelog, today)
    return commit, changelog


def enforce_changelog_header(changelog: str, today: str) -> str:
    """Guarantee the entry opens with the exact Keep-a-Changelog header line.

    Small models drop or reword the header; we normalise it deterministically
    so the snippet is always pasteable into a real CHANGELOG/README.
    """
    want = f"### [Unreleased] — {today}"
    lines = changelog.splitlines()
    # find any existing line that mentions Unreleased and replace it in place
    for i, l in enumerate(lines):
        if "unreleased" in l.lower():
            lines[i] = want
            return "\n".join(lines).strip()
    return f"{want}\n\n" + changelog.strip()


# ═════════════════════════════════════════════════════════════════════════════
# 9. OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def print_results(commit_msg, changelog, analysis: DiffAnalysis, use_json: bool):
    if use_json:
        files = []
        for f in analysis.files:
            d = asdict(f)
            d.pop("raw_lines", None)
            d.pop("header_lines", None)
            files.append(d)
        print(json.dumps({
            "commit_message": commit_msg,
            "changelog": changelog,
            "analysis": {
                "files": files,
                "total_additions": analysis.total_additions,
                "total_deletions": analysis.total_deletions,
                "dep_changes": analysis.dep_changes,
                "scopes": analysis.scopes,
                "type_guess": analysis.type_guess,
                "type_confident": analysis.type_confident,
                "breaking": analysis.breaking,
                "insights": analysis.insights,
            },
        }, indent=2))
        return

    sep = "─" * 60
    if RICH:
        console.print()
        console.print(Panel(
            "\n".join(f"• {i}" for i in analysis.insights),
            title="[bold magenta]⊙ Structural Insights[/bold magenta]",
            border_style="magenta",
        ))
        console.print(Panel(
            Syntax(commit_msg, "text", theme="monokai", word_wrap=True),
            title="[bold yellow]① Conventional Commit[/bold yellow]",
            border_style="yellow",
        ))
        console.print(Panel(
            Markdown(changelog),
            title="[bold cyan]② README Changelog Snippet[/bold cyan]",
            border_style="cyan",
        ))
    else:
        print(f"\n{sep}\n⊙ STRUCTURAL INSIGHTS\n{sep}")
        for i in analysis.insights:
            print(f"  • {i}")
        print(f"\n{sep}\n① CONVENTIONAL COMMIT\n{sep}")
        print(commit_msg)
        print(f"\n{sep}\n② README CHANGELOG SNIPPET\n{sep}")
        print(changelog)
        print(sep)


def save_output(commit_msg, changelog, analysis, out_path: Path):
    insights = "\n".join(f"- {i}" for i in analysis.insights)
    content = textwrap.dedent(f"""\
        <!-- Generated by git_to_doc_v2.py -->

        ## Structural Insights

        {insights}

        ## Commit Message

        ```
        {commit_msg}
        ```

        ## Changelog Entry

        {changelog}
    """)
    out_path.write_text(content, encoding="utf-8")
    print(f"\n✅  Saved to {out_path}", file=sys.stderr)


# ═════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def load_diff(path: Path) -> str:
    if not path.exists():
        print(f"❌  File not found: {path}", file=sys.stderr)
        sys.exit(1)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        print("❌  Diff file is empty.", file=sys.stderr)
        sys.exit(1)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="git_to_doc — turn a git diff into a Conventional Commit + changelog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python git_to_doc_v2.py changes.diff
              python git_to_doc_v2.py pr_42.txt --model gemma3:1b
              python git_to_doc_v2.py changes.diff --output entry.md
              python git_to_doc_v2.py changes.diff --json
        """),
    )
    parser.add_argument("diff_file", type=Path, help="Path to .diff or .txt file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional: save output to this .md file")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of formatted text")
    args = parser.parse_args()

    raw = load_diff(args.diff_file)
    files = parse_diff(raw)
    if not files:
        print("❌  No file changes found — is this a valid git diff?", file=sys.stderr)
        sys.exit(1)

    analysis = analyze(files)

    if not args.json:
        print(f"🔍  Analysing {args.diff_file.name} with {args.model} …", file=sys.stderr)

    today = date.today().isoformat()
    commit_msg, changelog = generate(analysis, args.model, today)

    print_results(commit_msg, changelog, analysis, args.json)
    if args.output:
        save_output(commit_msg, changelog, analysis, args.output)


if __name__ == "__main__":
    main()
