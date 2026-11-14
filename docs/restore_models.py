#!/usr/bin/env python3
"""Restore and checksum-verify model snapshots without network access."""
import argparse
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_store import REVISIONS, model_root, relative_model, safe_name, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=root) as temporary:
        staged = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive entries')
            for name in names:
                safe_name(name)
            archive.extractall(staged)
        expected = set()
        for repo in REVISIONS:
            prefix = relative_model(repo)
            manifest = validate(staged/prefix, repo, hashes=True)
            expected.update((prefix/name).as_posix() for name in [*manifest['files'], 'manifest.json'])
        if set(names) != expected:
            raise ValueError('Archive contains unexpected files')
        # Validate all snapshots before installing any. Never overwrite existing models.
        for repo in REVISIONS:
            target = root/relative_model(repo)
            if target.exists():
                validate(target, repo, hashes=True)
        for repo in REVISIONS:
            target = root/relative_model(repo)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                (staged/relative_model(repo)).rename(target)
            print(f'Installed and verified: {target}')


if __name__ == '__main__':
    main()
