"""Render the GitHub README's verification cards from docs/verification.toml."""
import argparse
from html import escape
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- verification:start -->"
END = "<!-- verification:end -->"


def render_card(title, caption, percent, fraction):
    track_width = 326
    fill = "#79e0cf" if fraction == 1 else "#ffc49b" if fraction == 0 else "#92c8ff"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="366" height="104" viewBox="0 0 366 104" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{caption} ({percent}%).</desc>
  <rect x=".5" y=".5" width="365" height="103" rx="11" fill="#122737" stroke="#395160"/>
  <g font-family="Inter, Segoe UI, Arial, sans-serif">
    <text x="20" y="34" font-size="17" font-weight="600" fill="#edf5f5">{title}</text>
    <text x="346" y="36" text-anchor="end" font-size="25" font-weight="700" fill="{fill}">{percent}%</text>
    <text x="20" y="62" font-size="14.5" fill="#c6d9df">{caption}</text>
  </g>
  <rect x="20" y="78" width="{track_width}" height="7" rx="3.5" fill="#344c5c"/>
  <rect x="20" y="78" width="{track_width * fraction:.2f}" height="7" rx="3.5" fill="{fill}"/>
</svg>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report stale cards without writing")
    args = parser.parse_args()
    with (ROOT / "docs" / "verification.toml").open("rb") as spec:
        metrics = tomllib.load(spec)
    assets = ROOT / "docs" / "assets"
    output = {}
    links = ['<p align="center">']
    for key, metric in metrics.items():
        count, total = metric["verified"], metric["total"]
        if not isinstance(count, int) or not isinstance(total, int) or not 0 <= count <= total or total == 0:
            raise ValueError(f"Invalid counter for {key}")
        title = escape(metric["title"])
        caption = f'{count} / {total} {escape(metric["unit"])}'
        fraction = count / total
        percent = round(fraction * 100)
        card = assets / f"progress-{key}.svg"
        output[card] = render_card(title, caption, percent, fraction)
        image = card.relative_to(ROOT).as_posix()
        alt = f"{title} — {caption} ({percent}%)"
        links.append(f'  <a href="{escape(metric["evidence"])}"><img src="{image}" width="366" alt="{alt}"></a>')
    links.append("</p>")
    readme = ROOT / "README.md"
    original = readme.read_text(encoding="utf-8")
    before, section = original.split(START)
    _, after = section.split(END)
    output[readme] = before + START + "\n" + "\n".join(links) + "\n" + END + after

    stale = []
    for path, content in output.items():
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        stale.append(path.relative_to(ROOT).as_posix())
        if not args.check:
            path.write_text(content, encoding="utf-8", newline="\n")
    if args.check and stale:
        print("Run python tools/render_progress.py to update " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
