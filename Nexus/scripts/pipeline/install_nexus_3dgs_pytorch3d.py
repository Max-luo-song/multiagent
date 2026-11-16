"""Install a checksum-verified official PyTorch3D conda binary into our venv.

The official py39/CUDA12.1 build targets Torch 2.3.1. Its compatibility with
the reused Torch 2.3.0 is checked by probe_nexus_3dgs_runtime.py using CUDA KNN.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = root/'artifacts/nexus_3dgs/venv'
    if Path(sys.prefix).resolve() != destination.resolve() or sys.version_info[:2] != (3, 9):
        raise ValueError('Run with the isolated project venv Python 3.9')
    basename = 'linux-64/pytorch3d-0.7.8-py39_cu121_pyt231.tar.bz2'
    metadata = next(item for item in json.loads(args.index.read_text())['files'] if item['basename'] == basename)
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    # The official legacy conda entry publishes MD5 but has an empty SHA256.
    # Verify its published checksum and size, and record our own SHA256 too.
    algorithm = 'sha256' if metadata.get('sha256') else 'md5'
    published = hashlib.new(algorithm, args.archive.read_bytes()).hexdigest()
    if published != metadata[algorithm] or args.archive.stat().st_size != metadata['size']:
        raise ValueError('Official PyTorch3D archive checksum mismatch')
    with tarfile.open(args.archive, 'r:bz2') as archive:
        members = []
        for member in archive.getmembers():
            if member.name.startswith('lib/python3.9/site-packages/'):
                if member.issym() or member.islnk() or '..' in Path(member.name).parts:
                    raise ValueError('Unsupported archive member')
                members.append(member)
        if not members:
            raise ValueError('Archive contains no Python package')
        archive.extractall(destination, members=members)
    (root/'artifacts/nexus_3dgs/pytorch3d_install.json').write_text(json.dumps({
        'source': 'https://conda.anaconda.org/pytorch3d/'+basename, 'sha256': digest,
        'upstream_checksum': {algorithm: published},
        'destination': str(destination), 'upstream_build': 'py39_cu121_pyt231',
        'runtime_compatibility': 'must pass probe_nexus_3dgs_runtime.py'}, indent=2)+'\n')
    print('PYTORCH3D_INSTALLED '+str(destination))


if __name__ == '__main__':
    main()
