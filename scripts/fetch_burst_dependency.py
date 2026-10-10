"""Fetch a fixed external Burst repository without overwriting an existing copy."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'third_party/burst-ann2snn'
URL = 'https://github.com/bic-L/burst-ann2snn.git'
COMMIT = 'f69f670b97ab3fa008ea5156b606cc3d3a735175'


def main():
    if TARGET.exists():
        head = subprocess.check_output(['git', '-C', str(TARGET), 'rev-parse', 'HEAD'], text=True).strip()
        if head != COMMIT:
            raise RuntimeError('Existing Burst checkout has a different commit; use BURST_ROOT to select it explicitly.')
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', URL, str(TARGET)], check=True)
        subprocess.run(['git', '-C', str(TARGET), 'checkout', '--detach', COMMIT], check=True)
    expected = json.loads((ROOT / 'scripts/burst_patch_manifest.json').read_text())
    def matches():
        return all(hashlib.sha256((TARGET / r['path']).read_bytes()).hexdigest() == r['sha256'] for r in expected)
    if not matches():
        patch = str(ROOT / 'scripts/burst_server_compat.patch')
        subprocess.run(['git', '-C', str(TARGET), 'apply', '--check', patch], check=True)
        subprocess.run(['git', '-C', str(TARGET), 'apply', patch], check=True)
    if not matches():
        raise RuntimeError('Patched Burst modules do not match the server snapshot')
    print('Installed and verified:', TARGET, COMMIT, 'with server compatibility patch')


if __name__ == '__main__':
    main()
