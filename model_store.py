"""Pinned, self-contained model snapshots; no Torch import required."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
REVISIONS = json.loads((ROOT/'models.json').read_text())


def model_root():
    return Path(os.environ.get('POWERTRANSCRIBE_MODELS', ROOT/'models')).expanduser().resolve()


def relative_model(repo):
    return Path(repo.replace('/', '--'))/REVISIONS[repo]


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError(f'Unsafe model file path: {name!r}')
    return path


def validate(path, repo, hashes=False):
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['repo'] != repo or manifest['revision'] != REVISIONS[repo] or not manifest['files']:
        raise ValueError(f'Wrong or empty model manifest: {path}')
    for name, info in manifest['files'].items():
        safe_name(name)
        file = path/name
        if not file.is_file() or file.stat().st_size != info['size']:
            raise ValueError(f'Missing/incomplete model file: {file}')
        if hashes and digest(file) != info['sha256']:
            raise ValueError(f'Model checksum mismatch: {file}')
    return manifest


def ensure_model(repo):
    target = model_root()/relative_model(repo)
    if target.exists():
        validate(target, repo)
        return target
    if os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'ON', 'YES', 'TRUE'):
        raise FileNotFoundError(f'Model missing: {target}. Restore with docs/restore_models.py.')
    from huggingface_hub import snapshot_download
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=target.parent) as temporary:
        snapshot = Path(snapshot_download(repo_id=repo, revision=REVISIONS[repo]))
        staged = Path(temporary)/'snapshot'
        # Dereference cache symlinks so this copy survives cache removal.
        shutil.copytree(snapshot, staged, symlinks=False)
        files = {p.relative_to(staged).as_posix(): {'size': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(staged.rglob('*')) if p.is_file()}
        (staged/'manifest.json').write_text(json.dumps(
            dict(repo=repo, revision=REVISIONS[repo], files=files), indent=2)+'\n')
        try:
            staged.rename(target)
        except OSError:
            if not target.exists():
                raise
            validate(target, repo, hashes=True)
    return target
