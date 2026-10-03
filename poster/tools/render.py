"""Render the editable A0 poster and make a self-contained HTML copy."""
from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path
import re

import pymupdf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def bundle(source: Path, target: Path) -> None:
    def inline(match):
        quote, value = match.groups()
        if value.startswith("data:"):
            return match.group(0)
        asset = (source.parent / value).resolve()
        if not asset.is_relative_to(source.parent.resolve()) or not asset.is_file():
            raise ValueError("All image sources must be local poster assets")
        mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        return f"src={quote}data:{mime};base64,{encoded}{quote}"

    html = source.read_text(encoding="utf-8")
    html = re.sub(r"\bsrc=([\"'])(.*?)\1", inline, html)
    target.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "poster.html")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    if args.png_dpi < 150:
        parser.error("Use at least 150 DPI for a readable poster")
    source = args.source.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "BeatEdit_Poster_A0.pdf"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={
            "width": round(841 / 25.4 * 96),
            "height": round(1189 / 25.4 * 96),
        })
        page.emulate_media(media="print")
        page.goto(source.as_uri(), wait_until="networkidle")
        page.evaluate("""async () => {
            await document.fonts.ready;
            await Promise.all(Array.from(document.images, image => image.decode()));
        }""")
        page.pdf(path=str(pdf), width=f"{841 / 25.4}in",
                 height=f"{1189 / 25.4}in", print_background=True,
                 margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        browser.close()
    with pymupdf.open(pdf) as document:
        if len(document) != 1:
            raise RuntimeError("Poster must be exactly one page")
        rect = document[0].rect
        if abs(rect.width * 25.4 / 72 - 841) > 1 or abs(rect.height * 25.4 / 72 - 1189) > 1:
            raise RuntimeError("Poster dimensions must be A0 portrait")
        if not args.no_png:
            document[0].get_pixmap(
                matrix=pymupdf.Matrix(args.png_dpi / 72, args.png_dpi / 72),
                alpha=False,
            ).save(out / f"BeatEdit_Poster_A0_{args.png_dpi}dpi.png")
    bundle(source, out / "poster_standalone.html")
    print("A0 single-page PDF, requested PNG and standalone HTML exported.")


if __name__ == "__main__":
    main()
