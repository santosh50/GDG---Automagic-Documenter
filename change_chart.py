#!/usr/bin/env python3
"""
change_chart.py — visual structure of a git diff
=================================================
Renders a PNG that shows *where* a change landed: additions vs deletions
grouped by category, and the heaviest individual files. It consumes the
DiffAnalysis produced by git_to_doc_v2.py, so it never re-parses anything.

Usage (standalone):
  python change_chart.py <path/to/file.diff> <out.png>

Usage (from git_to_doc_v2.py):
  python git_to_doc_v2.py changes.diff --chart changes.png

Requirements:
  pip install matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path

# Diverging palette: additions green, deletions red (GitHub-ish).
_ADD = "#2ea043"
_DEL = "#cf222e"
_GRID = "#d0d7de"


def _require_matplotlib():
    """Import matplotlib lazily with a friendly message if it is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")            # headless: no display needed
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except ImportError:
        print(
            "❌  matplotlib is required for --chart.\n"
            "    Install it with:  pip install matplotlib",
            file=sys.stderr,
        )
        raise


def _category_totals(analysis):
    """{category: (additions, deletions, file_count)} sorted by total churn."""
    agg: dict[str, list[int]] = {}
    for f in analysis.files:
        a, d, n = agg.setdefault(f.category, [0, 0, 0])
        agg[f.category] = [a + f.additions, d + f.deletions, n + 1]
    return dict(sorted(agg.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True))


def _top_files(analysis, limit: int = 8):
    """The `limit` files with the most total churn (additions + deletions)."""
    ranked = sorted(
        analysis.files,
        key=lambda f: f.additions + f.deletions,
        reverse=True,
    )
    return ranked[:limit]


def _short(path: str, width: int = 28) -> str:
    """Trim a path from the left so the file name always stays visible."""
    return path if len(path) <= width else "…" + path[-(width - 1):]


def _draw_diverging(ax, labels, adds, dels, title):
    """Horizontal diverging bars: deletions to the left, additions to the right."""
    y = range(len(labels))
    ax.barh(y, adds, color=_ADD, label="additions")
    ax.barh(y, [-d for d in dels], color=_DEL, label="deletions")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()                      # biggest at the top
    ax.axvline(0, color="#57606a", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    # annotate each bar end with its count
    for i, (a, d) in enumerate(zip(adds, dels)):
        if a:
            ax.text(a, i, f" +{a}", va="center", ha="left", fontsize=8, color=_ADD)
        if d:
            ax.text(-d, i, f"-{d} ", va="center", ha="right", fontsize=8, color=_DEL)


def render_change_chart(analysis, out_path, title: str | None = None) -> Path:
    """Render the change structure to a PNG and return its path."""
    plt = _require_matplotlib()
    out_path = Path(out_path)

    cats = _category_totals(analysis)
    cat_labels = [f"{c} ({n})" for c, (_, _, n) in cats.items()]
    cat_adds = [a for (a, _, _) in cats.values()]
    cat_dels = [d for (_, d, _) in cats.values()]

    files = _top_files(analysis)
    file_labels = [_short(f.path) for f in files]
    file_adds = [f.additions for f in files]
    file_dels = [f.deletions for f in files]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, max(3.5, 0.45 * max(len(cats), len(files)) + 1.5))
    )

    heading = title or "Change structure"
    fig.suptitle(
        f"{heading}  —  {len(analysis.files)} files, "
        f"+{analysis.total_additions} / -{analysis.total_deletions}",
        fontsize=13, fontweight="bold",
    )

    _draw_diverging(ax1, cat_labels, cat_adds, cat_dels, "By category")
    _draw_diverging(ax2, file_labels, file_adds, file_dels, "Heaviest files")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── standalone entry point ───────────────────────────────────────────────────
def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python change_chart.py <file.diff> <out.png>", file=sys.stderr)
        return 2

    # reuse the parser/analyzer from the main tool — no duplicate logic
    import git_to_doc_v2 as core

    diff_path, out_path = Path(argv[0]), Path(argv[1])
    files = core.parse_diff(core.load_diff(diff_path))
    if not files:
        print("❌  No file changes found — is this a valid git diff?", file=sys.stderr)
        return 1
    analysis = core.analyze(files)
    render_change_chart(analysis, out_path, title=diff_path.name)
    print(f"✅  Saved chart to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
