#!/usr/bin/env python3
"""
git_to_doc.py — Automagic Documenter
=====================================
Takes a raw git diff (.diff or .txt) and routes it through a local
Google Gemma model (via Ollama) to produce:
  1. A Conventional Commit message
  2. A Markdown README changelog snippet

Usage:
  python git_to_doc.py <path/to/file.diff>
  python git_to_doc.py <path/to/file.diff> --output changelog.md
  python git_to_doc.py <path/to/file.diff> --model gemma3:4b
  python git_to_doc.py <path/to/file.diff> --json

Requirements:
  pip install requests
  # Ollama running locally: https://ollama.com
  # Pull a Gemma model: ollama pull gemma3:4b
"""

import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import date

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
DEFAULT_MODEL = "gemma3:4b"  # swap to gemma3:1b for speed, gemma3:12b for quality

# ── token budget: keep diffs under 6 k chars to stay within context window ──
MAX_DIFF_CHARS = 6_000


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

COMMIT_PROMPT = """You are an expert software engineer. Analyse the git diff below and write ONE Conventional Commit message.

Rules:
- Format: <type>(<scope>): <short summary in present tense, ≤72 chars>
- Valid types: feat, fix, docs, style, refactor, perf, test, chore, ci
- Scope: the primary module, file, or subsystem changed (lowercase, no spaces)
- Summary: imperative mood, no period, no capital letter after colon
- After the header, add a blank line, then a 2–4 sentence body explaining WHY
- If there are breaking changes add a footer: BREAKING CHANGE: <description>
- Output ONLY the commit message — no extra commentary, no code fences.

Git diff:
{diff}
"""

CHANGELOG_PROMPT = """You are a technical writer. Using the git diff below, write a Markdown changelog entry.

Rules:
- Start with: ### [Unreleased] — {today}
- Use bullet lists under these subsections (only include non-empty ones):
  #### Added, #### Changed, #### Fixed, #### Removed, #### Security
- Each bullet: present tense, plain English, ≤100 chars, no jargon
- End with a blank line
- Output ONLY the Markdown — no preamble, no code fences.

Git diff:
{diff}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gemma via Ollama
# ─────────────────────────────────────────────────────────────────────────────

def call_gemma(prompt: str, model: str) -> str:
    """Send a prompt to a local Ollama Gemma model and return the response text."""
    import urllib.request, urllib.error

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,      # low temp → deterministic, structured output
            "num_predict": 512,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body.get("response", "").strip()
    except urllib.error.URLError as exc:
        print(f"\n❌  Cannot reach Ollama at {OLLAMA_URL}", file=sys.stderr)
        print("    Make sure Ollama is running:  ollama serve", file=sys.stderr)
        print(f"    And the model is pulled:      ollama pull {model}\n", file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_diff(path: Path) -> str:
    """Load and validate a diff file; truncate gracefully if too large."""
    if not path.exists():
        print(f"❌  File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"❌  Could not read file: {exc}", file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("❌  Diff file is empty.", file=sys.stderr)
        sys.exit(1)

    if len(text) > MAX_DIFF_CHARS:
        print(
            f"⚠️   Diff is large ({len(text):,} chars); truncating to "
            f"{MAX_DIFF_CHARS:,} chars to fit the model context window.",
            file=sys.stderr,
        )
        text = text[:MAX_DIFF_CHARS] + "\n... [truncated]"

    return text


def extract_diff_stats(diff: str) -> dict:
    """Quick scan: count files changed, additions, deletions."""
    files, adds, dels = set(), 0, 0
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) == 2:
                files.add(parts[1].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            dels += 1
    return {"files": len(files), "additions": adds, "deletions": dels}


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_results(commit_msg: str, changelog: str, stats: dict, use_json: bool):
    if use_json:
        output = {
            "commit_message": commit_msg,
            "changelog": changelog,
            "stats": stats,
        }
        print(json.dumps(output, indent=2))
        return

    separator = "─" * 60

    if RICH:
        console.print()
        console.print(Panel.fit(
            f"[dim]{stats['files']} file(s) · "
            f"[green]+{stats['additions']}[/green] · "
            f"[red]-{stats['deletions']}[/red]",
            title="[bold]Diff stats[/bold]",
        ))
        console.print()
        console.print(Panel(
            Syntax(commit_msg, "text", theme="monokai", word_wrap=True),
            title="[bold yellow]① Conventional Commit[/bold yellow]",
            border_style="yellow",
        ))
        console.print()
        console.print(Panel(
            Markdown(changelog),
            title="[bold cyan]② README Changelog Snippet[/bold cyan]",
            border_style="cyan",
        ))
    else:
        print(f"\n{separator}")
        print(f"  Diff stats: {stats['files']} file(s)  "
              f"+{stats['additions']} additions  -{stats['deletions']} deletions")
        print(f"{separator}\n")
        print("① CONVENTIONAL COMMIT")
        print(separator)
        print(commit_msg)
        print(f"\n{separator}\n")
        print("② README CHANGELOG SNIPPET")
        print(separator)
        print(changelog)
        print(separator)


def save_output(commit_msg: str, changelog: str, out_path: Path):
    content = textwrap.dedent(f"""\
        <!-- Generated by git_to_doc.py -->

        ## Commit Message

        ```
        {commit_msg}
        ```

        ## Changelog Entry

        {changelog}
    """)
    out_path.write_text(content, encoding="utf-8")
    print(f"\n✅  Saved to {out_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="git_to_doc — turn a git diff into a Conventional Commit + changelog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python git_to_doc.py changes.diff
              python git_to_doc.py pr_42.txt --model gemma3:1b
              python git_to_doc.py changes.diff --output CHANGELOG_entry.md
              python git_to_doc.py changes.diff --json
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

    # 1. Load diff
    diff = load_diff(args.diff_file)
    stats = extract_diff_stats(diff)

    if not args.json:
        print(f"🔍  Analysing {args.diff_file.name} with {args.model} …", file=sys.stderr)

    today = date.today().isoformat()

    # 2. Generate commit message
    commit_prompt = COMMIT_PROMPT.format(diff=diff)
    commit_msg = call_gemma(commit_prompt, args.model)

    # 3. Generate changelog entry
    changelog_prompt = CHANGELOG_PROMPT.format(diff=diff, today=today)
    changelog = call_gemma(changelog_prompt, args.model)

    # 4. Print / save
    print_results(commit_msg, changelog, stats, args.json)

    if args.output:
        save_output(commit_msg, changelog, args.output)


if __name__ == "__main__":
    main()
