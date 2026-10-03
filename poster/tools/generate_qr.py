#!/usr/bin/env python
"""Generate the three poster URL QR codes locally, without a QR web service.

Run with the poster's dedicated interpreter:
  .venv/bin/python tools/generate_qr.py --verify

Dependencies: qrcode[pil]; optional decode verification: zxing-cpp.
Edit DEFAULT_URLS or pass --demo-url / --homepage-url / --prior-work-url.
Only generated QR assets are replaced; the poster HTML is never edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import qrcode
from PIL import Image
from qrcode.image.svg import SvgPathFillImage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URLS = {
    "beatedit_demo_qr": "https://haoyu-gu.github.io/BeatEdit/",
    "haoyu_homepage_qr": "https://haoyu-gu.github.io/",
    "beat_prior_work_qr": "https://arxiv.org/abs/2604.19532",
}


def web_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("Use a complete http(s) URL.")
    return value


def verify_png(path: Path, expected: str) -> None:
    try:
        import zxingcpp
    except ImportError as exc:
        raise RuntimeError("--verify requires zxing-cpp in the same environment.") from exc
    with Image.open(path) as image:
        result = zxingcpp.read_barcode(image)
    if result is None or result.text != expected:
        raise RuntimeError(f"QR round-trip verification failed: {path}")


def generate(stem: str, url: str, out: Path, box_size: int, verify: bool) -> dict:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=4,  # Keep the standard four-module quiet zone.
    )
    qr.add_data(url)
    qr.make(fit=True)
    png = out / f"{stem}.png"
    svg = out / f"{stem}.svg"
    qr.make_image(fill_color="black", back_color="white").save(png, dpi=(300, 300))
    qr.make_image(image_factory=SvgPathFillImage).save(svg)
    if verify:
        verify_png(png, url)
    pixels = (qr.modules_count + 2 * qr.border) * qr.box_size
    return {
        "url": url,
        "png": png.name,
        "svg": svg.name,
        "pixels": [pixels, pixels],
        "quiet_zone_modules": qr.border,
        "error_correction": "M",
        "decoded_and_verified": verify,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-url", type=web_url, default=DEFAULT_URLS["beatedit_demo_qr"])
    parser.add_argument("--homepage-url", type=web_url, default=DEFAULT_URLS["haoyu_homepage_qr"])
    parser.add_argument("--prior-work-url", type=web_url, default=DEFAULT_URLS["beat_prior_work_qr"])
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "qr")
    parser.add_argument("--box-size", type=int, default=24, help="PNG pixels per QR module (default: 24)")
    parser.add_argument("--verify", action="store_true", help="Decode each generated PNG and compare the exact URL")
    args = parser.parse_args()
    if args.box_size < 1:
        parser.error("--box-size must be positive.")
    args.out.mkdir(parents=True, exist_ok=True)
    urls = {"beatedit_demo_qr": args.demo_url, "haoyu_homepage_qr": args.homepage_url,
            "beat_prior_work_qr": args.prior_work_url}
    results = {name: generate(name, url, args.out, args.box_size, args.verify) for name, url in urls.items()}
    (args.out / "qr_manifest.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, info in results.items():
        print(f"{name}: {info['url']} -> PNG + SVG ({info['pixels'][0]}px); verified={args.verify}")


if __name__ == "__main__":
    main()
