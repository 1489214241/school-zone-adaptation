"""Continue the authorized server run, stopping at any failed validation gate."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import resume_adaptation as app


RECORDS = app.REPO / 'server_results/20261110-105500'
ROOT = app.REPO / 'outputs/server-20261110-111700'
TRANSFER = app.REPO.parent / 'school-zone-transfer'
CHECK_PID = 1434292


def save(stage, **details):
    value = dict(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 supervisor_pid=os.getpid(), output_root=str(ROOT.relative_to(app.REPO)),
                 expected_tasks=70, original_completed_tasks=122, **details)
    app.write_json(RECORDS / 'execution.json', value, replace=True)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def launch(arguments, gpu, log_name):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    with (ROOT / 'logs' / log_name).open('xb') as stream:
        return subprocess.Popen([sys.executable, str(app.REPO / 'scripts/resume_adaptation.py'),
                                 *arguments], cwd=app.REPO, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)


def completed(tasks):
    result = []
    for task in tasks:
        out = ROOT / app.arm_cfg(app.configuration(task['group']), task['arm'])['output_dir']
        probability = out / 'probs' / f"{task['district']}.npz"
        sidecar = out / 'tasks' / f"{task['district']}.json"
        if not probability.is_file() or not sidecar.is_file():
            continue
        record = app.read_json(sidecar)
        if record['probabilities'] != app.fingerprint(probability):
            raise RuntimeError(f'Probability fingerprint mismatch: {probability}')
        if record['config'] != app.fingerprint(out / 'config.json'):
            raise RuntimeError(f'Configuration fingerprint mismatch: {sidecar}')
        result.append(task)
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / 'logs').mkdir(exist_ok=True)
        save('waiting_data_check')
        while Path(f'/proc/{CHECK_PID}/cmdline').exists():
            try:
                cmdline = Path(f'/proc/{CHECK_PID}/cmdline').read_bytes()
            except FileNotFoundError:
                break
            if b'scripts/resume_adaptation.py\0check' not in cmdline:
                break
            time.sleep(10)
        log = (TRANSFER / 'data-check.log').read_text()
        if 'Traceback' in log or any(f'{group}: built evidence for 8 districts' not in log
                                     for group in app.GROUPS):
            save('failed_data_check')
            return 1
        app.verify_sources(app.read_json(app.REPO / 'SOURCE_MANIFEST.json'))
        tasks = app.plan()
        if tasks != app.read_json(RECORDS / 'tasks.json') or len(tasks) != 70:
            raise RuntimeError('Frozen task plan differs')
        save('pilot', gpu=0, group='k1_s1', district='雨花区')
        pilot = launch(['pilot', '--group', 'k1_s1', '--district', '雨花区',
                        '--output-root', str(ROOT)], 0, 'pilot.log')
        code = pilot.wait()
        if code:
            save('failed_pilot', returncode=code, log=str((ROOT / 'logs/pilot.log').relative_to(app.REPO)))
            return code
        base_meta = ROOT / app.arm_cfg(app.configuration('k1_s1'), 'BASE')['output_dir'] / 'tasks/雨花区.json'
        baseline = app.read_json(base_meta)['baseline_check']
        initial = app.read_json(ROOT / 'checks/k1_s1/雨花区.json')['initial_condition']
        if not initial['passed'] or baseline['max_prob_diff'] >= 1e-5 or baseline['changed_correctness']:
            raise RuntimeError('Pilot gate failed')
        workers = [launch(['run', '--shard', str(shard), '--output-root', str(ROOT)],
                          shard, f'worker-{shard}.log') for shard in (0, 1)]
        save('running', workers=[dict(shard=i, gpu=i, pid=p.pid) for i, p in enumerate(workers)],
             pilot_baseline=baseline, completed_tasks=len(completed(tasks)))
        while any(worker.poll() is None for worker in workers):
            time.sleep(30)
        codes = [worker.returncode for worker in workers]
        done = completed(tasks)
        save('complete' if codes == [0, 0] and len(done) == len(tasks) else 'partial',
             worker_returncodes=codes, completed_tasks=len(done),
             completed=done, pilot_baseline=baseline)
        return 0 if codes == [0, 0] and len(done) == len(tasks) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output_root.resolve()
    try:
        sys.exit(main())
    except Exception as error:
        save('failed_supervisor', error=f'{type(error).__name__}: {error}')
        raise
