"""Resume and verify the six XJTU-SY RAR volumes from the authors' MediaFire mirror."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests


TARGET = Path("data/raw/xjtu-sy")
FOLDER_URL = "https://www.mediafire.com/folder/m3sij67rizpb4/XJTU-SY_Bearing_Datasets"
# Public file metadata from the Data subfolder of the authors' MediaFire share.
PARTS = (
    ("58qg5pjkqq5t26r", 744488960, "c500657353f089a4ab50212ff4ddfc7b982729e92e2b127440c8ad881d80b968"),
    ("sqbsl8ja9c9e84x", 744488960, "4dfa6286a8e9c7cec1e925b347642349f36e27ab3f703c90d1d08977b7b4f61f"),
    ("fj8e74g01vwn5hy", 744488960, "6929f531284f79e0209246b4ee23b23dd8d4faac1c0c3ff8fb131cda4a18bd3a"),
    ("domu7dirov1wjc8", 744488960, "2245825acd29bed3b95e7a77879a3fbadf20f4ad4f99486384c714174621597d"),
    ("n5ozw16n3wqgdaw", 744488960, "e1b0a41a32b865e48ea7981c56f952a960d94335f3496b72eb4a65932f1671bd"),
    ("ts2majip39fcnq1", 722155640, "df1854821a9d481104476379f7bc045ea1e101427c6e36145cd7677dc7c4a684"),
)


class _DownloadLink(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("id") == "downloadButton":
            self.href = attributes.get("href")


def _direct_link(session: requests.Session, number: int) -> str:
    quickkey, _, _ = PARTS[number - 1]
    filename = f"XJTU-SY_Bearing_Datasets.part{number:02d}.rar"
    page = session.get(f"https://www.mediafire.com/file/{quickkey}/{filename}/file", timeout=30)
    page.raise_for_status()
    parser = _DownloadLink()
    parser.feed(page.text)
    link = parser.href
    host = urlparse(link or "").hostname or ""
    if not link or urlparse(link).scheme != "https" or not host.endswith(".mediafire.com"):
        raise RuntimeError(f"Part {number:02d} has no trusted download link")
    return link


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_part(session: requests.Session, number: int, target: Path) -> None:
    _, expected_bytes, expected_hash = PARTS[number - 1]
    path = target / f"XJTU-SY_Bearing_Datasets.part{number:02d}.rar"
    partial = Path(str(path) + ".partial")
    if path.exists():
        if path.stat().st_size != expected_bytes or sha256(path) != expected_hash:
            raise RuntimeError(f"Part {number:02d} exists but fails the author's mirror hash")
        print(f"Part {number:02d}/06 verified locally", flush=True)
        return
    for attempt in range(1, 5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_bytes:
            raise RuntimeError(f"Part {number:02d} partial file exceeds expected size")
        try:
            link = _direct_link(session, number)
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(link, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("MediaFire returned HTML instead of an archive")
                if offset and response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Resume offset mismatch")
                    mode = "ab"
                elif response.status_code == 200:
                    mode = "wb"  # Server ignored Range; restart cleanly.
                    offset = 0
                else:
                    raise RuntimeError(f"Unexpected HTTP {response.status_code} for part {number:02d}")
                with partial.open(mode) as output:
                    next_report = ((offset // 100_000_000) + 1) * 100_000_000
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            if output.tell() >= next_report:
                                print(f"Part {number:02d}: {output.tell() / 1e6:.0f}/{expected_bytes / 1e6:.0f} MB", flush=True)
                                next_report += 100_000_000
            if partial.stat().st_size == expected_bytes:
                digest = sha256(partial)
                if digest != expected_hash:
                    raise RuntimeError(f"Part {number:02d} SHA-256 does not match the author's mirror")
                partial.replace(path)
                print(f"Part {number:02d}/06 verified: {digest}", flush=True)
                return
            print(f"Part {number:02d} incomplete, resuming (attempt {attempt}/4)", flush=True)
        except requests.RequestException as exc:
            print(f"Part {number:02d} network interruption ({type(exc).__name__}); retrying", flush=True)
    raise RuntimeError(f"Part {number:02d} could not be completed after four attempts")


def save_manifest(target: Path) -> dict:
    files = []
    for number, (quickkey, expected_bytes, expected_hash) in enumerate(PARTS, 1):
        path = target / f"XJTU-SY_Bearing_Datasets.part{number:02d}.rar"
        if not path.exists():
            continue
        digest = sha256(path)
        if path.stat().st_size != expected_bytes or digest != expected_hash:
            raise RuntimeError(f"Part {number:02d} failed final integrity check")
        files.append({"part": number, "file": path.name, "mediafire_quickkey": quickkey,
                      "bytes": expected_bytes, "sha256": digest})
    manifest = {"source": FOLDER_URL, "parts_expected": 6,
                "parts_verified_locally": len(files), "files": files,
                "sha256_provenance": "MediaFire Data folder file metadata from the author's linked mirror"}
    (target / "download.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=range(1, 7), action="append",
                        help="Download selected part(s); default is all six")
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    selected = sorted(set(args.part or range(1, 7)))
    remaining = sum(PARTS[n - 1][1] - min(PARTS[n - 1][1],
                    (TARGET / f"XJTU-SY_Bearing_Datasets.part{n:02d}.rar.partial").stat().st_size
                    if (TARGET / f"XJTU-SY_Bearing_Datasets.part{n:02d}.rar.partial").exists() else 0)
                    for n in selected if not (TARGET / f"XJTU-SY_Bearing_Datasets.part{n:02d}.rar").exists())
    if shutil.disk_usage(TARGET).free < remaining + 1_000_000_000:
        raise SystemExit("Insufficient space for selected RAR parts and a 1 GB buffer")
    with requests.Session() as session:
        for number in selected:
            _download_part(session, number, TARGET)
    manifest = save_manifest(TARGET)
    print(json.dumps({"verified_parts": manifest["parts_verified_locally"],
                      "total_bytes": sum(x["bytes"] for x in manifest["files"]),
                      "manifest": str(TARGET / "download.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
