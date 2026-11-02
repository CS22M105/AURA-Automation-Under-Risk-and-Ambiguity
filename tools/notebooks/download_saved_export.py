"""Create a verified download from an existing Drive export, without model loading."""

import hashlib
import json
import zipfile
from pathlib import Path


def create_verified_download(source: str | Path, destination: str | Path) -> Path:
    """Verify every declared file while writing one ZIP; reject incomplete exports."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No saved export manifest in {source}.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or not manifest.get("file_sha256"):
        raise ValueError("The saved export has an unsupported or empty manifest.")
    expected = manifest["file_sha256"]
    missing = []
    for relative in expected:
        path = source / relative
        if source not in path.resolve().parents:
            raise ValueError(f"Invalid export file path: {relative}")
        if not path.is_file():
            missing.append(relative)
    if missing:
        raise FileNotFoundError(
            f"The Drive export itself is incomplete: {len(missing)} of {len(expected)} "
            "files are missing. No ZIP was created. Missing files:\n" + "\n".join(missing)
        )
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("Choose a new ZIP filename; previous downloads are preserved.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    folder_name = source.name + "-verified"
    print(f"Found all {len(expected)} files. Verifying and archiving them…", flush=True)
    try:
        # Use a low compression level to reduce download size without a long
        # compression pass. A temporary suffix prevents downloading a partial ZIP.
        with zipfile.ZipFile(
            partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as archive:
            for index, (relative, wanted) in enumerate(sorted(expected.items()), start=1):
                path = source / relative
                if path.stat().st_size > 50_000_000 or index % 15 == 0:
                    print(f"[{index}/{len(expected)}] {relative}", flush=True)
                digest = hashlib.sha256()
                with (
                    path.open("rb") as incoming,
                    archive.open(f"{folder_name}/{relative}", "w", force_zip64=True) as outgoing,
                ):
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        digest.update(block)
                        outgoing.write(block)
                if digest.hexdigest() != wanted:
                    raise ValueError(
                        f"Saved Drive file differs from its manifest: {relative}. "
                        "No downloadable ZIP was published."
                    )
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError(
                    "The export manifest changed while copying. Retry once it is stable."
                )
            archive.writestr(f"{folder_name}/manifest.json", manifest_bytes)
        partial.rename(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"Verified {len(expected)}/{len(expected)} files. ZIP ready: {destination}", flush=True)
    return destination
