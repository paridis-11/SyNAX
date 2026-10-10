"""Check syntax, original source hashes and release contents without training."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for name, record in manifest['files'].items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record['sha256']:
            raise RuntimeError('Copied source changed: ' + name)
    sources = [p for d in ('static', 'event', 'scripts')
               for p in (ROOT / d).rglob('*.py')]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    for path in (ROOT / 'scripts').glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    forbidden = [p for d in ('static', 'event') for p in (ROOT / d).rglob('*')
                 if p.is_file() and (p.suffix in ('.pt', '.pth', '.npz', '.npy', '.safetensors', '.pem', '.key')
                                     or p.name in ('.env', 'credentials', 'id_rsa'))]
    if forbidden:
        raise RuntimeError('Unexpected data or credential file: ' + str(forbidden[0]))
    print('Original source hashes verified:', len(manifest['files']))
    print('Python files parsed:', len(sources))
    print('Launch shell syntax: passed')
    print('Source-only contents: passed')


if __name__ == '__main__':
    main()
