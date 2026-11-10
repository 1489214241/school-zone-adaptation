"""按区均衡训练 / 按区概况调节的一轮试验（方案：科研推进/按区调整与均衡训练_20261110/方案.md）。
  python scripts/adaptation_trial.py run outputs/main_pre_v1/configs/k1_s1.json     # 一个 (K, 批)：证据只建一次，各臂依次训练
  python scripts/adaptation_trial.py report pre_v1                                   # 汇总：各臂 − 基线
各臂只改训练/模型开关；输出 <基线 output_dir>__T_<臂>/probs/<区>.npz。基线＝main_table 已跑完的完整模型。
"""
import sys, json, copy, time, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np

ARMS = {'BASE': '不改（自检）', 'B1': '每个区总权重相等', 'B2': '两座城市总权重相等', 'AS': '区概况固定，调节证据块增益',
        'A0': '训练时随机藏参照（不调节）', 'A1': '随机藏参照＋按剩下的已知小区调节'}
TARGET = [(0, 1), (1, 1), (1, 2), (1, 3)]


def arm_cfg(base, arm):
    c = copy.deepcopy(base)
    if arm == 'B1': c['train']['items']['balance'] = 'district'
    if arm == 'B2': c['train']['items']['balance'] = 'city'
    if arm in ('AS', 'A1'): c['model']['condition'] = True
    if arm in ('A0', 'A1'): c['train']['reference_dropout'] = 0.8
    c['output_dir'] = base['output_dir'] + f'__T_{arm}'; c['说明'] = base.get('说明', '') + f'；试验臂 {arm}：{ARMS[arm]}'
    return c


def run(path, arms):
    from zoning import rules, evidence, train, references
    from run import load_districts
    base = json.loads(Path(path).read_text(encoding='utf-8')); t0 = time.time(); D = load_districts(base)
    R = {d: rules.run(D[d], base['rules']['preset'], base['rules'].get('hangzhou_names', 'all'), base['rules'].get('hangzhou_boundary', 'v1'), base['rules'].get('hangzhou_community_rule', False)) for d in D}
    R, _ = references.apply(base.get('references'), D, R)
    X = {d: evidence.build(D[d], R[d], base['evidence']) for d in D}; print('证据', f'{time.time() - t0:.0f}s', flush=True)
    for arm in arms:
        c = arm_cfg(base, arm); out = REPO / c['output_dir']
        if all((out / 'probs' / f'{d}.npz').exists() for d in D): continue
        (out / 'probs').mkdir(parents=True, exist_ok=True); (out / 'config.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
        for d in D:
            res = train.fit(d, X, D, c, c['train']['seeds']); P = np.stack([res[s][1] for s in c['train']['seeds']])
            np.savez_compressed(out / 'probs' / f'{d}.npz', probs=P, seeds=c['train']['seeds'])
        print('完成', arm, f'{time.time() - t0:.0f}s', flush=True)


def report(tag):
    from zoning import rules, evaluate, references
    from run import load_districts
    from main_table import runs
    base = json.loads((REPO / 'configs/default.json').read_text(encoding='utf-8')); D = load_districts(base)
    R0 = {d: rules.run(D[d], base['rules']['preset'], base['rules'].get('hangzhou_names', 'all'), base['rules'].get('hangzhou_boundary', 'v1'), base['rules'].get('hangzhou_community_rule', False)) for d in D}
    ev = base['evaluate']; res = {}
    for name, k, s, c in runs(base, tag):
        if (k, s) not in TARGET: continue
        R, _ = references.apply(c['references'], D, R0)
        def load(odir):
            if not all((REPO / odir / 'probs' / f'{d}.npz').exists() for d in D): return None
            return {d: np.load(REPO / odir / 'probs' / f'{d}.npz')['probs'] for d in D}
        def rows(P): return [r for d in D for r in evaluate.correct(P[d].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))]
        P0 = load(c['output_dir']); r0 = rows(P0); out = {}
        for arm in ARMS:
            P = load(c['output_dir'] + f'__T_{arm}')
            if P is None: continue
            x = evaluate.compare(rows(P), r0, ev)
            if arm == 'BASE': x['max_prob_diff'] = float(max(np.abs(P[d] - P0[d]).max() for d in D))
            out[arm] = x
        res[name] = dict(k=k, seed=s, base=evaluate.summary(r0), arms=out)
    folder = REPO.parent / '科研推进/按区调整与均衡训练_20261110'; (folder / '评分.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    L = [f'# 各臂 − 基线（{tag}）', '', '差值 [95% 区间]（改对/改坏）；模型部分。', '']
    for name, r in res.items():
        L += [f"## {name}（基线 两城 {r['base']['两城']['accuracy']:.2f}、长沙 {r['base']['长沙']['accuracy']:.2f}、杭州 {r['base']['杭州']['accuracy']:.2f}）", '', '| 臂 | 两城 | 长沙 | 杭州 |', '|---|---|---|---|']
        for arm, x in r['arms'].items():
            L.append(f'| {arm} {ARMS[arm]} | ' + ' | '.join(f"{x[c_]['delta']:+.2f} [{x[c_]['ci'][0]:+.2f}, {x[c_]['ci'][1]:+.2f}]（{x[c_]['fixed']}/{x[c_]['broken']}）" for c_ in ['两城', '长沙', '杭州']) + ' |' + (f" 概率最大差 {x['max_prob_diff']:.1e}" if 'max_prob_diff' in x else ''))
        L.append('')
    (folder / '评分.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'report']); ap.add_argument('target'); ap.add_argument('--arms', nargs='*', default=list(ARMS))
    a = ap.parse_args()
    run(a.target, a.arms) if a.mode == 'run' else report(a.target)


if __name__ == '__main__':
    main()
