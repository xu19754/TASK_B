"""Build the portable Task B release, excluding independent basket experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from verify_release import verify

ROOT = Path(__file__).resolve().parents[1]
RELEASE_NAME = "TaskB_v3_2026-10-23"


def release_files():
    # An explicit allowlist keeps new experimental files out of future releases.
    names = {
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
        "scripts/run_taskb.sh", "scripts/play_taskb.py", "scripts/check_package.py",
        "scripts/build_release.py", "scripts/verify_release.py",
        "assets/b2z1/SOURCE.json", "assets/b2z1/b2z1_0815.urdf",
        "assets/b2z1/b2z1_0815_with_basket_v3.urdf",
        "tools/vendor/sldprt/LICENSE", "tools/vendor/sldprt/SOURCE.txt",
    }
    for folder in ("taskb_student", "demo", "tests"):
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob("*.py"))
    for name in ("policy.pt", "deployment_manifest.json"):
        names.add("models/locomotion/" + name)
    for name in ("README.md", "basket_properties.json", "零件2.SLDPRT"):
        names.add("assets/b2z1/basket_v3/" + name)
    for pattern in ("*.obj", "*.mtl", "basket_v3/*.obj"):
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "assets/b2z1/meshes").glob(pattern))
    for object_name in ("004_sugar_box", "006_mustard_bottle", "011_banana"):
        names.add(f"assets/objects/{object_name}.usd")
        names.add(f"assets/objects/Materials/Textures/{object_name}_COLOR.png")
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "dist" / RELEASE_NAME)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    archive = output.with_name(output.name + ".zip")
    if output.exists() or archive.exists() or archive.with_suffix(".zip.sha256").exists():
        raise FileExistsError("Release already exists; choose a new --output-dir.")
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Place release output outside the source project.")
    source_manifest = json.loads((ROOT / "assets/b2z1/SOURCE.json").read_text())
    files = release_files()
    for name in files:
        if not (ROOT / name).is_file() or (ROOT / name).is_symlink():
            raise FileNotFoundError(f"Missing regular release input: {name}")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    output.mkdir(parents=True)
    for name in files:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    manifest = {
        "release": RELEASE_NAME,
        "date": "2026-10-23",
        "version": "0.1.2",
        "runtime_urdf": source_manifest["runtime_urdf"],
        "runtime_urdf_sha256": source_manifest["runtime_urdf_sha256"],
        "policy": "models/locomotion/policy.pt",
        "excludes": [
            "reports/", "independent basket viewers, trajectories and analysis scripts",
            "desktop .command launchers", "old basket variants and their meshes",
            "local virtual environments, caches, thumbnails and nested archives",
        ],
        "validation_scope": "File integrity and portable assets; does not certify Isaac/PhysX dynamics with v3",
        "files_sha256": hashes,
    }
    (output / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify(output)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, output.name + "/" + path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Release directory: {output}")
    print(f"ZIP: {archive} ({archive.stat().st_size / 1024**2:.2f} MiB)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
