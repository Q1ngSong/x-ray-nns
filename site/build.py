"""Build the public project page and curated offline examples; no model execution."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xray.exporters.html import render_page
from xray.ir import InferenceTrace


def readme_html(repository: str) -> str:
    """Render the repository README and resolve file links for Pages. [基础设施]

    Args:
        repository: Public repository URL without a trailing slash.
    Returns:
        HTML with local image URLs and repository links for source files.
    """
    rendered = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"]).render(
        (ROOT / "README.md").read_text(encoding="utf-8")
    )

    def link(match: re.Match[str]) -> str:
        """Resolve a relative README link while retaining site images and anchors. [基础设施]

        Args:
            match: A rendered href attribute from the trusted project README.
        Returns:
            The original attribute or an escaped GitHub source URL.
        """
        value = match.group(1)
        if urlsplit(value).scheme or value.startswith(("#", "//", "assets/")):
            return match.group(0)
        return 'href="' + escape(repository + "/blob/main/" + value, quote=True) + '"'

    return re.sub(r'href="([^"]+)"', link, rendered)


def build(output: Path, repository: str) -> None:
    """Build a static site from explicitly listed examples and public assets. [主线]

    Args:
        output: New or empty destination; refuse to mix prior files into a release.
        repository: Public source repository URL used in navigation and documentation.
    """
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory.")
    examples = json.loads((ROOT / "site/examples.json").read_text(encoding="utf-8"))
    slugs = [example["slug"] for example in examples]
    if len(set(slugs)) != len(slugs) or any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) for slug in slugs):
        raise ValueError("Example slugs must be unique, lowercase path components.")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "assets/readme", output / "assets/readme", ignore=shutil.ignore_patterns("._*", ".DS_Store"))
    shutil.copyfile(ROOT / "site/style.css", output / "style.css")
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
    shutil.copyfile(ROOT / "xray/exporters/vendor/three.LICENSE", output / "THREE-LICENSE.txt")
    cards = []
    for number, example in enumerate(examples, 1):
        slug = example["slug"]
        source = ROOT / "site/samples" / slug
        destination = output / "examples" / slug
        destination.mkdir(parents=True)
        trace = InferenceTrace.load_json(source / "trace.json")
        scene = json.loads((source / "scene.json").read_text(encoding="utf-8"))
        page = render_page(trace, scene)
        page = page.replace('<header class="topbar">', '<header class="topbar"><a href="../../" style="color:var(--muted);font-size:12px;text-decoration:none;white-space:nowrap" aria-label="返回示例首页">← 首页</a>', 1)
        (destination / "index.html").write_text(page, encoding="utf-8")
        shutil.copytree(source / "assets", destination / "assets", ignore=shutil.ignore_patterns("._*", ".DS_Store"))
        tags = "".join('<span>' + escape(tag) + '</span>' for tag in example["tags"])
        cards.append(f'''<a class="case-card" href="examples/{slug}/" aria-label="探索 {escape(example['title'])} 示例">
  <div class="case-preview"><img src="{escape(example['preview'], quote=True)}" alt="CLIP 视觉和文本编码器的三维回放" width="1600" height="900" loading="lazy"></div>
  <div class="case-body"><div class="case-meta"><span>EXAMPLE {number:02d}</span><span>RECORDED INFERENCE</span></div>
    <h3>{escape(example['title'])}</h3><p class="case-subtitle">{escape(example['subtitle'])}</p>
    <p class="case-description">{escape(example['description'])}</p><div class="case-tags">{tags}</div>
    <div class="case-cta">打开交互示例 <span aria-hidden="true">↗</span></div>
  </div>
</a>''')
    template = (ROOT / "site/index.html").read_text(encoding="utf-8")
    replacements = {"REPOSITORY": escape(repository, quote=True), "EXAMPLE_CARDS": "\n".join(cards), "README": readme_html(repository)}
    page = re.sub(r"__(REPOSITORY|EXAMPLE_CARDS|README)__", lambda match: replacements[match.group(1)], template)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"Built {len(examples)} example(s): {output}")


def main() -> None:
    """Build the site into an explicitly selected directory. [主线]"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new or empty directory, relative to the current working directory")
    parser.add_argument("--repository", default="https://github.com/Q1ngSong/x-ray-nns", help="public GitHub repository URL")
    args = parser.parse_args()
    build(args.output.resolve(), args.repository.rstrip("/"))


if __name__ == "__main__":
    main()
