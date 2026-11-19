#!/usr/bin/env python3
"""Build a static website from all Markdown files in this repository.

Every *.md file (except README.md at the root) becomes an HTML page in _site/,
at the same relative path. Images and other files are copied as they are.
The root README.md is shown at the top of the home page, followed by an
automatically generated list of all notes, grouped by folder.

Requires pandoc (https://pandoc.org). Run from the repository root:
    python site/build.py
"""
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = Path(__file__).resolve().parent
OUT = ROOT / "_site"

SITE_TITLE = "2026B Seminar Supplements"          # shown in the header
SKIP_DIRS = {".git", ".github", "site", "_site", "node_modules"}
SKIP_FILES = {"LICENSE", ".gitignore"}
PANDOC_FROM = "markdown+lists_without_preceding_blankline-smart"   # pandoc Markdown, GitHub-compatible


def is_skipped(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]) \
        or rel.name in SKIP_FILES or rel.name.startswith(".")


def first_heading(md_text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", md_text, flags=re.M)
    return m.group(1).strip() if m else fallback


def first_sentence(md_text: str) -> str:
    """First sentence of the first paragraph after the title (used on the home page)."""
    body = re.sub(r"^#\s+.+$", "", md_text, count=1, flags=re.M)
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if para and not para.startswith(("#", "!", "|", "<", "$$", "-", "```")):
            para = re.sub(r"^\*\*[^*]+\*\*\s*", "", para)      # drop a leading bold label
            m = re.search(r"^(.+?[.!?。])(\s|$)", para.replace("\n", " "))
            return m.group(1) if m else para
    return ""


def folder_label(name: str) -> str:
    """'20261125' -> '2026-11-25'; other folder names are kept."""
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else name


def pandoc(src_text: str, out_file: Path, title: str, root_rel: str, toc: bool, resource_dir: Path, date: str = ""):
    cmd = [
        "pandoc", "-f", PANDOC_FROM, "-t", "html5", "--standalone",
        "--template", str(SITE / "template.html"),
        "--lua-filter", str(SITE / "links.lua"),
        "--mathjax",
        "--metadata", f"pagetitle={title}",
        "--metadata", f"sitetitle={SITE_TITLE}",
        "--metadata", f"root={root_rel}",
        "--resource-path", str(resource_dir),
        "-o", str(out_file),
    ]
    if toc:
        cmd += ["--toc", "--toc-depth=3", "--metadata", "isnote=true",
                "--lua-filter", str(SITE / "title.lua")]
    if date:
        cmd += ["--metadata", f"date={date}"]
    subprocess.run(cmd, input=src_text.encode("utf-8"), check=True)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copy(SITE / "style.css", OUT / "style.css")

    notes = []   # (folder, title, href, summary)
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or is_skipped(path):
            continue
        rel = path.relative_to(ROOT)
        if rel == Path("README.md"):
            continue
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            title = first_heading(text, path.stem)
            root_rel = "../" * (len(rel.parts) - 1) or "./"
            folder = rel.parent.as_posix()
            pandoc(text, dest.with_suffix(".html"), title, root_rel, toc=True,
                   resource_dir=path.parent, date=folder_label(folder) if folder != "." else "")
            notes.append((folder, title, rel.with_suffix(".html").as_posix(), first_sentence(text)))
            print("page :", rel)
        else:
            shutil.copy2(path, dest)

    # Home page: README.md (if any) + list of notes, newest folder first
    readme = ROOT / "README.md"
    intro = readme.read_text(encoding="utf-8") if readme.exists() else f"# {SITE_TITLE}\n"
    lines = [intro.rstrip(), "", "## Notes", "", '<ul class="note-list">']
    for folder, title, href, summary in sorted(notes, key=lambda n: (n[0], n[2]), reverse=True):
        date = folder_label(folder) if folder != "." else ""
        lines += ["<li>", f'<div class="note-date">{date}</div>', "<div>",
                  f'<a class="note-title" href="{href}">{title}</a>']
        if summary:   # Markdown inside the div, so math and bold still work
            lines += ['<div class="note-summary">', "", summary, "", "</div>"]
        lines += ["</div>", "</li>"]
    lines.append("</ul>")
    pandoc("\n".join(lines), OUT / "index.html", SITE_TITLE, "./", toc=False, resource_dir=ROOT)
    (OUT / ".nojekyll").touch()
    print(f"done : {len(notes)} pages -> {OUT}")


if __name__ == "__main__":
    main()
