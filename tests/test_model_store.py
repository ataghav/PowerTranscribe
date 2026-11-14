import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import model_store as models


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def snapshot(self, repo, base):
        path = base/models.relative_model(repo)
        path.mkdir(parents=True)
        file = path/'config.json'
        file.write_text('{}')
        manifest = dict(repo=repo, revision=models.REVISIONS[repo], files={
            'config.json': dict(size=file.stat().st_size, sha256=models.digest(file))})
        (path/'manifest.json').write_text(json.dumps(manifest))
        return path

    def test_existing_snapshot_never_downloads_and_offline_missing_fails(self):
        repo = next(iter(models.REVISIONS))
        path = self.snapshot(repo, self.root)
        with patch.dict(os.environ, POWERTRANSCRIBE_MODELS=str(self.root), HF_HUB_OFFLINE='1'):
            self.assertEqual(models.ensure_model(repo), path)
            with self.assertRaises(FileNotFoundError):
                models.ensure_model(list(models.REVISIONS)[1])
            (path/'config.json').write_text('broken')
            with self.assertRaises(ValueError):
                models.ensure_model(repo)

    def test_backup_restore_round_trip_and_checksum_rejection(self):
        source, target = self.root/'source', self.root/'target'
        for repo in models.REVISIONS:
            self.snapshot(repo, source)
        archive = self.root/'backup.zip'
        env = dict(os.environ, POWERTRANSCRIBE_MODELS=str(source), HF_HUB_OFFLINE='1')
        subprocess.run([sys.executable, str(models.ROOT/'docs/download_models.py'), '--archive', str(archive)],
                       env=env, check=True, capture_output=True)
        env['POWERTRANSCRIBE_MODELS'] = str(target)
        restore = [sys.executable, str(models.ROOT/'docs/restore_models.py'), str(archive)]
        for _ in range(2):  # Reinstallation retains valid files.
            subprocess.run(restore, env=env, check=True, capture_output=True)
        for repo in models.REVISIONS:
            models.validate(target/models.relative_model(repo), repo, hashes=True)
        bad = self.root/'bad.zip'
        with zipfile.ZipFile(archive) as original, zipfile.ZipFile(bad, 'w') as modified:
            for name in original.namelist():
                modified.writestr(name, b'[]' if name.endswith('/config.json') else original.read(name))
        env['POWERTRANSCRIBE_MODELS'] = str(self.root/'bad-target')
        result = subprocess.run([*restore[:-1], str(bad)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum mismatch', result.stderr)
        self.assertFalse(any((self.root/'bad-target').glob('*--*')))

    def test_unsafe_archive_path_rejected(self):
        archive = self.root/'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as file:
            file.writestr('../escaped', 'bad')
        result = subprocess.run(
            [sys.executable, str(models.ROOT/'docs/restore_models.py'), str(archive)],
            env=dict(os.environ, POWERTRANSCRIBE_MODELS=str(self.root/'models')),
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe model file path', result.stderr)
        self.assertFalse((self.root/'escaped').exists())
