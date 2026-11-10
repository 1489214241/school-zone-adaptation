"""Validate and package completed server tasks without overwriting prior artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import resume_adaptation as app


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def package(output_root, records, destination):
    state = app.read_json(records / 'execution.json')
    if state['stage'] != 'complete' or state.get('worker_returncodes') != [0, 0]:
        raise RuntimeError('All tasks must finish successfully before final packaging')
    if (app.REPO / state['output_root']).resolve() != output_root.resolve():
        raise RuntimeError('Output root differs from the supervisor completion record')
    tasks = app.read_json(records / 'tasks.json')
    if tasks != app.plan() or len(tasks) != state['completed_tasks']:
        raise RuntimeError('Frozen task plan or completed count differs')
    app.verify_sources(app.read_json(app.REPO / 'SOURCE_MANIFEST.json'))
    files = {}
    rows = []
    for task in tasks:
        expected = app.arm_cfg(app.configuration(task['group']), task['arm'])
        relative = Path(expected['output_dir'])
        config = output_root / relative / 'config.json'
        probability = output_root / relative / 'probs' / f"{task['district']}.npz"
        metadata = output_root / relative / 'tasks' / f"{task['district']}.json"
        record = app.read_json(metadata)
        if app.read_json(config) != expected:
            raise RuntimeError(f'Frozen configuration differs: {config}')
        for key, path in [('config', config), ('probabilities', probability)]:
            if record[key] != app.fingerprint(path):
                raise RuntimeError(f'Fingerprint differs: {path}')
        for key in ('group', 'arm', 'district'):
            if record['task'][key] != task[key]:
                raise RuntimeError(f'Task metadata differs: {metadata}')
        baseline_path = app.REPO / app.configuration(task['group'])['output_dir'] / 'probs' / probability.name
        with np.load(baseline_path, allow_pickle=False) as reference:
            shape = reference['probs'].shape
        with np.load(probability, allow_pickle=False) as data:
            p = data['probs']
            if p.shape != shape or not np.array_equal(data['seeds'], expected['train']['seeds']):
                raise RuntimeError(f'Probability shape or seeds differ: {probability}')
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-5):
                raise RuntimeError(f'Invalid probabilities: {probability}')
        if record['environment']['resume_script_sha256'] != app.fingerprint(Path(app.__file__))['sha256']:
            raise RuntimeError('Executed runner differs from the code being packaged')
        for path in (config, probability, metadata):
            files[path.relative_to(output_root).as_posix()] = path
        rows.append(dict(task=task, fit_seconds=record['fit_seconds'],
                         peak_cuda_reserved_bytes=record['peak_cuda_reserved_bytes'],
                         execution_commit=record['environment']['git_commit']))
    for path in sorted(records.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.md', '.patch'):
            files[path.relative_to(app.REPO).as_posix()] = path
    for name in ('resume_adaptation.py', 'continue_server_experiment.py', 'package_server_results.py'):
        path = app.REPO / 'scripts' / name
        files[path.relative_to(app.REPO).as_posix()] = path
    for name in ('SOURCE_MANIFEST.json', 'experiment_state/completed_districts.json'):
        files[name] = app.REPO / name
    for path in sorted((output_root / 'checks').rglob('*.json')):
        files['server_checks/' + path.relative_to(output_root / 'checks').as_posix()] = path
    pilot_dir = Path(app.arm_cfg(app.configuration('k1_s1'), 'BASE')['output_dir'])
    for name in ('config.json', 'probs/雨花区.npz', 'tasks/雨花区.json'):
        files['pilot_baseline/' + name] = output_root / pilot_dir / name
    for path in sorted((output_root / 'logs').glob('*.log')):
        sanitized = path.read_text().replace(str(app.REPO), '<repository>').replace(str(app.REPO.parent), '<home>')
        files['server_logs/' + path.name] = sanitized.encode('utf-8')
    created = datetime.now(timezone.utc)
    summary = dict(status='complete', created_utc=created.isoformat(), completed_tasks=len(rows),
                   failed_tasks=0, original_completed_tasks=state['original_completed_tasks'],
                   historical_baseline=state['pilot_baseline'],
                   executed_commits=sorted({r['execution_commit'] for r in rows}),
                   total_fit_seconds=sum(r['fit_seconds'] for r in rows),
                   maximum_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in rows),
                   tasks=rows,
                   evaluation='Server completion only; combine with the original 122 tasks and score on the original machine under the frozen pre_v1 protocol.')
    files['server_summary.json'] = json_bytes(summary)
    inventory = {}
    for name, source in files.items():
        inventory[name] = app.fingerprint(source) if isinstance(source, Path) else {
            'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()}
    files['archive_manifest.json'] = json_bytes(inventory)
    bundle = destination / created.strftime('server-results-%Y%m%d-%H%M%S')
    bundle.mkdir(parents=True, exist_ok=False)
    archive = bundle / (bundle.name + '.zip')
    with zipfile.ZipFile(archive, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name, source in sorted(files.items()):
            if isinstance(source, Path):
                zipped.write(source, arcname=name)
            else:
                zipped.writestr(name, source)
    with zipfile.ZipFile(archive) as zipped:
        if sorted(zipped.namelist()) != sorted(files):
            raise RuntimeError('Archive file list differs')
        for name, expected in inventory.items():
            data = zipped.read(name)
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise RuntimeError(f'Archive verification failed: {name}')
        if json.loads(zipped.read('archive_manifest.json')) != inventory:
            raise RuntimeError('Archive inventory differs')
    digest = app.fingerprint(archive)
    with archive.with_suffix('.zip.sha256').open('x') as stream:
        stream.write(f"{digest['sha256']}  {archive.name}\n")
    receipt = dict(archive=str(archive), **digest, verified_payload_files=len(inventory),
                   completed_tasks=len(rows), failed_tasks=0)
    with (bundle / 'receipt.json').open('x') as stream:
        stream.write(json_bytes(receipt).decode('utf-8'))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--records', type=Path, default=app.REPO / 'server_results/20261110-105500')
    parser.add_argument('--destination', type=Path, default=app.REPO.parent / 'school-zone-transfer/packages')
    arguments = parser.parse_args()
    package(arguments.output_root, arguments.records, arguments.destination)
