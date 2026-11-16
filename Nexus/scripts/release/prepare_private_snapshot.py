"""Create an audited source-only Git snapshot without importing old history.

No upload, network request, authentication-file access, or original-file redaction
is performed by this script. Secret matches are removed only in copied text;
reports contain paths, rule names and line numbers, never matched values.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRS = {
    '.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
    '.cache', '.venv', 'venv', 'envs', 'node_modules', 'build', 'dist',
    'site-packages', 'artifacts', 'report', 'data', 'logs', 'log', 'wandb',
    'outputs', 'output', 'ckpt', 'checkpoints', '.aws', '.ssh', '.codex',
    '.agents', '.vscode', '.sii', '.medias', '.idea', '.ipynb_checkpoints',
    '.tox', '.eggs', 'htmlcov', 'test_data', 'testdata', 'pretrained',
}
TEXT_SUFFIXES = {
    '.py', '.pyi', '.sh', '.bash', '.yaml', '.yml', '.json', '.toml',
    '.cfg', '.ini', '.conf', '.md', '.rst', '.txt', '.c', '.h', '.cc',
    '.cpp', '.cu', '.cuh', '.hpp', '.cmake', '.xml', '.html', '.css',
    '.js', '.ts', '.svg', '.csv', '.patch', '.diff', '.proto', '.in',
    '.gin', '.ipynb', '.pyx', '.pxd', '.bib', '.tex', '.sty', '.bat',
}
TEXT_NAMES = {
    'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING', 'NOTICE', 'AUTHORS',
    'CONTRIBUTORS', 'Makefile', 'Dockerfile', 'MANIFEST.in', '.gitignore',
    '.dockerignore', '.gitattributes', '.gitmodules', '.env.example',
    '.env.template', '.clang-format', '.pre-commit-config.yaml',
}
SOURCES = ['nuplan_extent', 'scripts', 'docs', 'nexus_3dgs',
           'Omniscenegraph', 'third_party', '.github']
TOP_FILES = ['LICENSE', 'requirements.txt', 'alf_compat_changes.json', '.gitignore']
RULES = [
    ('private_key', re.compile(r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----')),
    ('openai_style_key', re.compile(r'\bsk-(?:proj-|ant-|svcacct-)?[A-Za-z0-9_-]{20,}')),
    ('github_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})')),
    ('gitlab_token', re.compile(r'\bglpat-[A-Za-z0-9_-]{15,}')),
    ('huggingface_token', re.compile(r'\bhf_[A-Za-z0-9]{25,}')),
    ('aws_access_key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('google_api_key', re.compile(r'\bAIza[A-Za-z0-9_-]{30,}')),
    ('slack_token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}')),
    ('jwt_token', re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}')),
    ('url_password', re.compile(r'(?P<prefix>\b[a-z][a-z0-9+.-]*://[^\s/:@]+:)(?P<value>[^\s/@]+)(?=@)', re.I)),
    ('literal_secret', re.compile(
        r'''(?P<prefix>\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|aws[_-]?secret[_-]?access[_-]?key|password|passwd|secret[_-]?key)\b["']?\s*[:=]\s*)(?P<quote>["'])(?P<value>[^"'\n]{6,})(?P=quote)''', re.I)),
    ('bearer_token', re.compile(r'(?P<prefix>\bBearer\s+)(?P<value>[A-Za-z0-9_.-]{20,})', re.I)),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def placeholder(value):
    v = value.lower()
    return (v.startswith(('redacted', 'your_', 'your-', 'example', 'dummy',
                          'changeme', 'replace_', '${', '{{', '<', '$'))
            or v in {'password', 'secret', 'none', 'null', 'optional',
                     'api_key', 'api-key', 'access_token', 'not-a-secret'}
            or any(x in v for x in ('getenv(', 'environ[', 'environ.get(',
                                    'os.environ', 'self.', 'args.')))


def redact(text):
    findings = []
    for rule, pattern in RULES:
        def replace(match):
            value = match.groupdict().get('value', match.group())
            if placeholder(value):
                return match.group()
            findings.append({'rule': rule, 'line': text.count('\n', 0, match.start()) + 1})
            if rule == 'literal_secret':
                return match['prefix'] + match['quote'] + 'REDACTED_USE_ENVIRONMENT' + match['quote']
            if rule in {'url_password', 'bearer_token'}:
                return match['prefix'] + 'REDACTED_USE_ENVIRONMENT'
            return 'REDACTED_USE_ENVIRONMENT'
        text = pattern.sub(replace, text)
    return text, findings


def excluded_file(path):
    name = path.name.lower()
    return ((name.startswith('.env') and name not in {'.env.example', '.env.template'})
            or name in {'.netrc', '.npmrc', '.pypirc', '.git-credentials', 'known_hosts',
                        'authorized_keys', 'credentials', 'credentials.json', 'secrets.json'}
            or name.startswith(('id_rsa', 'id_ed25519', 'credentials.', 'secrets.'))
            or name.endswith(('.pem', '.key', '.p12', '.pfx', '.keystore', '.log'))
            or name.endswith(('.before-nexus-compat', '.bak', '.orig')))


def git(*args, cwd=ROOT):
    return subprocess.run(['git', '-C', str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def self_check():
    probes = [
        'key=' + 'sk-' + 'Q' * 35,
        'token=' + 'ghp_' + 'Q' * 36,
        'api_key="' + 'A9' * 20 + '"',
        'password: "' + 'B7' * 10 + '"',
        'https://user:' + 'C8' * 10 + '@example.org',
        'Bearer ' + 'D6' * 20,
        '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----\n' + 'SENSITIVE' + '\n-----END OPENSSH PRIVATE KEY-----',
    ]
    for probe in probes:
        cleaned, matches = redact(probe)
        assert matches and cleaned != probe
        assert not redact(cleaned)[1]
    for probe in ['api_key=os.environ.get("KEY")', 'api_key="YOUR_API_KEY"',
                  'secret_key="REDACTED_USE_ENVIRONMENT"']:
        assert not redact(probe)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    self_check()
    target = args.output.resolve()
    assert not target.exists(), 'Use a new output directory; previous freezes are preserved'
    target.mkdir(parents=True)
    audit = {'excluded_by_reason': collections.Counter(), 'redactions': [],
             'omitted_binary_or_oversized_paths': [], 'source_sha256': {}, 'copied_files': 0}

    def copy_file(source, relative):
        if source.is_symlink():
            audit['excluded_by_reason']['symlink'] += 1
            return
        if excluded_file(source):
            audit['excluded_by_reason']['credential_or_backup_name'] += 1
            return
        if source.name not in TEXT_NAMES and source.suffix.lower() not in TEXT_SUFFIXES:
            audit['excluded_by_reason']['non_source_extension'] += 1
            return
        if source.stat().st_size > 3 * 1024 * 1024:
            audit['excluded_by_reason']['over_3_MiB'] += 1
            audit['omitted_binary_or_oversized_paths'].append(str(relative))
            return
        payload = source.read_bytes()
        if b'\0' in payload:
            audit['excluded_by_reason']['binary_content'] += 1
            return
        try:
            text = payload.decode('utf-8')
        except UnicodeDecodeError:
            audit['excluded_by_reason']['non_utf8_content'] += 1
            return
        # Notebooks may contain arbitrary printed credentials or rich outputs.
        if source.suffix == '.ipynb':
            try:
                nb = json.loads(text)
                for cell in nb.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        cell['outputs'] = []
                        cell['execution_count'] = None
                    cell.pop('attachments', None)
                nb['metadata'] = {}
                text = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
            except (ValueError, TypeError):
                audit['excluded_by_reason']['invalid_notebook'] += 1
                return
        cleaned, matches = redact(text)
        if matches:
            audit['redactions'].append({'path': str(relative), 'findings': matches})
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(cleaned)
        if os.access(source, os.X_OK):
            dst.chmod(0o755)
        audit['source_sha256'][str(relative)] = hashlib.sha256(payload).hexdigest()
        audit['copied_files'] += 1

    for relative in TOP_FILES:
        source = ROOT / relative
        if source.is_file():
            copy_file(source, Path(relative))
    copy_file(ROOT / 'README.md', Path('README_UPSTREAM.md'))
    for name in SOURCES:
        for folder, dirs, files in os.walk(ROOT / name, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS
                             and not d.endswith('.egg-info')
                             and not Path(folder, d).is_symlink()
                             and not (Path(folder).name == '.github' and d == 'workflows'))
            for name in sorted(files):
                source = Path(folder, name)
                copy_file(source, source.relative_to(ROOT))

    dependencies = []
    candidates = [ROOT, ROOT / 'Omniscenegraph']
    candidates.extend(p for p in (ROOT / 'third_party').iterdir() if p.is_dir())
    candidates.extend(p for p in (ROOT / 'Omniscenegraph/third_party').iterdir() if p.is_dir())
    for source in candidates:
        if not (source / '.git').exists():
            continue
        status = git('status', '--short', '--untracked-files=no', cwd=source)
        remote = subprocess.run(['git', '-C', str(source), 'config', '--get', 'remote.origin.url'],
                                capture_output=True, text=True).stdout.strip()
        remote = re.sub(r'(https?://)[^/@\s]+@', r'\1REDACTED@', remote)
        remote, matches = redact(remote)
        dependencies.append({'path': '.' if source == ROOT else str(source.relative_to(ROOT)),
                             'commit': git('rev-parse', 'HEAD', cwd=source),
                             'remote': remote, 'tracked_worktree_changes': status.splitlines(),
                             'source_snapshot_used_instead_of_gitlink': True})
    (target / 'evidence').mkdir()
    for relative, name in [
        ('report/nexus_threecar_asset_insert_20261028/summary.json', 'threecar_asset_summary.json'),
        ('report/nexus_threecar_asset_insert_20261028/geometry_audit.json', 'threecar_geometry_audit.json'),
        ('report/nexus_threecar_B_multiview_20261028/summary.json', 'B_multiview_summary.json'),
        ('report/nexus_threecar_B_multiview_20261028/verification.json', 'B_multiview_verification.json'),
    ]:
        if (ROOT / relative).is_file():
            copy_file(ROOT / relative, Path('evidence') / name)
    write(target / 'DEPENDENCIES.lock.json', dependencies)
    (target / 'README.md').write_text('''# Nexus交互与3DGS：冻结代码交付

本仓保存2026-10-30的本地开发代码快照，目标是离线条件交通场景生成与3DGS呈现。
它包含Nexus复现、推理期交互/物理引导、三车跟驰与变道研究，以及外部高斯资产插入和B车四向相机。

- [项目入口与恢复说明](docs/PRIVATE_REPOSITORY_HANDOFF_20261030.md)
- [Baseline缺点与后续优化](docs/BASELINE_LIMITATIONS_20261030.md)
- [原Nexus说明及引用](README_UPSTREAM.md)
- [第三方来源及冻结版本](DEPENDENCIES.lock.json)
- [文件哈希](SOURCE_MANIFEST.json)
- [脱敏检查摘要](SECURITY_AUDIT.json)

源码快照使用新的Git历史，不携带原仓历史、任何嵌套.git目录或原机器的认证配置。
数据、模型权重、高斯资产、完整实验输出、视频、环境和编译产物留在本地，按交接说明单独准备。
第三方源码保留各自LICENSE和NOTICE；查看各目录的许可约定。实验结论和范围见baseline说明及evidence摘要。

此版本高级行为使用Nexus加显式行为搜索/轨迹优化；三车样例通过不等于完整场景安全或严格因果证明。
''')
    # Keep top-level guardrails, but allow necessary third-party source files.
    ignore = (target / '.gitignore').read_text()
    ignore = '\n'.join(line for line in ignore.splitlines()
                       if line not in {'third_party/MTR', 'third_party/nuplan-devkit', 'third_party/CAT'})
    (target / '.gitignore').write_text(ignore + '\n*.so\n*.a\n*.o\n*.whl\n*.zip\n*.tar\n*.gz\n')

    # Re-scan the final copied bytes. A remaining finding blocks all Git actions.
    remaining = []
    for path in target.rglob('*'):
        if path.is_file():
            _, matches = redact(path.read_text())
            if matches:
                remaining.append({'path': str(path.relative_to(target)), 'findings': matches})
    if remaining:
        write(target / 'BLOCKED_SECRET_AUDIT.json', remaining)
        raise RuntimeError('Snapshot contains remaining sensitive matches; no commit was created')
    audit['excluded_by_reason'] = dict(audit['excluded_by_reason'])
    write(target / 'SECURITY_AUDIT.json', {
        'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'rules': [name for name, _ in RULES], 'redactions': audit['redactions'],
        'excluded_by_reason': audit['excluded_by_reason'],
        'copied_source_files': audit['copied_files'],
        'git_history_included': False, 'nested_git_directories_included': False,
        'source_files_modified_by_export': False,
        'notebook_outputs_included': False, 'remaining_detected_secrets': 0,
        'scope': 'rule-based text and file admission checks; not a claim to detect every possible secret',
    })
    write(target / 'SOURCE_ORIGINS.json', audit['source_sha256'])
    manifest = {str(path.relative_to(target)): sha(path) for path in target.rglob('*') if path.is_file()}
    write(target / 'SOURCE_MANIFEST.json', manifest)
    git('init', '-b', 'main', cwd=target)
    git('config', 'user.name', 'keepgoing-xhz', cwd=target)
    git('config', 'user.email', 'keepgoing-xhz@users.noreply.github.com', cwd=target)
    git('config', 'core.quotepath', 'false', cwd=target)
    paths = sorted([*manifest, 'SOURCE_MANIFEST.json'])
    # Explicit audited paths admit vendored source even when an upstream
    # dependency's own ignore rules exclude its entire third_party directory.
    subprocess.run(['git', '-C', str(target), 'add', '--force', '--pathspec-from-file=-',
                    '--pathspec-file-nul'], input='\0'.join(paths).encode() + b'\0',
                   check=True, capture_output=True)
    assert set(git('ls-files', cwd=target).splitlines()) == set(paths)
    assert not any(line.startswith('160000 ') for line in git('ls-files', '--stage', cwd=target).splitlines())
    git('commit', '-m', 'Freeze Nexus interaction and 3DGS baseline (2026-10-30)', cwd=target)
    assert not git('status', '--porcelain', cwd=target)
    print(json.dumps({'status': 'complete', 'output': str(target),
                      'commit': git('rev-parse', 'HEAD', cwd=target),
                      'tracked_files': len(paths), 'redacted_files': len(audit['redactions']),
                      'bytes': sum(path.stat().st_size for path in target.rglob('*')
                                   if path.is_file() and '.git' not in path.parts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
