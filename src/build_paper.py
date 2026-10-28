"""Compile the paper and copy the PDF to docs/.

Usage:  python src/build_paper.py

Compiles paper/stent-2026-value-weighted-targeting.tex with tectonic (preferred) or latexmk, fails on
undefined references, citations or macros, and copies the result to
docs/stent-2026-value-weighted-targeting.pdf.  The LaTeX source lives in the private paper/ folder,
which is absent from the public repository; without it this script exits with a message and the
tracked PDF in docs/ is the only artefact.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO / "paper"
DOCS_DIR = REPO / "docs"
STEM = "stent-2026-value-weighted-targeting"
TEX = PAPER_DIR / f"{STEM}.tex"
PDF_BUILT = PAPER_DIR / f"{STEM}.pdf"
PDF_PUBLISHED = DOCS_DIR / f"{STEM}.pdf"

PROBLEM = re.compile(
    r"undefined (control sequence|reference|citation)|Reference .* undefined|Citation .* undefined"
    r"|There were undefined|Undefined control sequence|multiply[- ]defined|Missing \$ inserted",
    re.IGNORECASE,
)


def compile_command():
    """Return the command that builds TEX, or None if no LaTeX toolchain is installed."""
    if shutil.which("tectonic"):
        return ["tectonic", "--keep-logs", "--outdir", str(PAPER_DIR), str(TEX)]
    if shutil.which("latexmk"):
        return [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={PAPER_DIR}",
            str(TEX),
        ]
    return None


def main():
    if not TEX.exists():
        print(
            "paper/ is missing: the LaTeX source is private and is not part of the public repository.\n"
            f"The compiled paper is docs/{STEM}.pdf.",
            file=sys.stderr,
        )
        return 2
    cmd = compile_command()
    if cmd is None:
        print("No LaTeX toolchain found: install tectonic or latexmk.", file=sys.stderr)
        return 2
    proc = subprocess.run(cmd, cwd=PAPER_DIR, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    log = PAPER_DIR / f"{STEM}.log"
    if log.exists():
        output += "\n" + log.read_text(errors="replace")
    if proc.returncode != 0:
        print(output[-4000:], file=sys.stderr)
        print(f"build failed: {' '.join(cmd[:1])} exited {proc.returncode}", file=sys.stderr)
        return proc.returncode
    problems = sorted({m.group(0) for line in output.splitlines() if (m := PROBLEM.search(line))})
    if problems:
        print("build has undefined references or macros: " + "; ".join(problems), file=sys.stderr)
        return 1
    DOCS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(PDF_BUILT, PDF_PUBLISHED)
    print(f"built {PDF_BUILT.relative_to(REPO)} and copied it to {PDF_PUBLISHED.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
