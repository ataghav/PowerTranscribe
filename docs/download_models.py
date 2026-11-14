#!/usr/bin/env python3
"""Download pinned model snapshots and create a self-contained ZIP64 backup."""
import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_store import REVISIONS, ensure_model, relative_model, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).with_name('powertranscribe-models.zip'))
    args = parser.parse_args()
    paths = []
    for repo in REVISIONS:
        print(f'Preparing {repo}', flush=True)
        path = ensure_model(repo)
        manifest = validate(path, repo, hashes=True)
        paths.append((repo, path, manifest))
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing backups. ZIP64 supports multi-GB weights.
    with args.archive.open('xb') as output:
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for repo, path, manifest in paths:
                for name in [*manifest['files'], 'manifest.json']:
                    archive.write(path/name, (relative_model(repo)/name).as_posix())
    print(f'Created {args.archive.resolve()}. Copy this archive to your personal OneDrive.')


if __name__ == '__main__':
    main()
