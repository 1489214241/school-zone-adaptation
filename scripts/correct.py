"""限定范围的证据修正（zoning/correction.py）：冻结一次已跑完的模型，只学带名字证据处的修正。
  python scripts/correct.py configs/name_correction.json
步骤：
  1. 嵌套模型：对每个目标区 t、每个来源区 o，训练以 o 为目标、来源不含 t 的模型（设置、种子同原模型），各种子平均概率存 inner/<t>__<o>.npz；
  2. 每个目标区、每个种子，训练两组修正头：限定范围 / 不限定，作用在原模型（base_output 的各种子平均概率）上，各种子修正后概率再平均；
  3. 评价：两组各自对原模型、限定对不限定；带名字证据的小区上改对改坏；自检。
输出到 output_dir：inner/、probs/<组>/<区>.npz、eval.json、config.json。
"""
import sys, json, time, argparse, copy
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np
import torch
from zoning import data, rules, evidence, train, evaluate, correction
from zoning.data import ALL

ARMS = {'限定范围': True, '不限定': False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8')); bc = json.loads((REPO / cfg['base_config']).read_text(encoding='utf-8'))
    out = REPO / cfg['output_dir']; (out / 'inner').mkdir(parents=True, exist_ok=True); t0 = time.time()
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    D = {d: data.load(d) for d in ALL}; M = {d: correction.name_mask(D[d]) for d in ALL}
    base = {d: np.load(REPO / cfg['base_output'] / 'probs' / f'{d}.npz')['probs'].mean(0) for d in ALL}
    checks = {}

    # 1. 嵌套模型
    need = [(t, o) for t in ALL for o in ALL if o != t and not (out / 'inner' / f'{t}__{o}.npz').exists()]
    if need:
        R = {d: rules.run(D[d], bc['rules']['preset']) for d in ALL}; X = {d: evidence.build(D[d], R[d], bc['evidence']) for d in ALL}
        print('证据', f'{time.time() - t0:.0f}s', flush=True)
        for t, o in need:
            c = copy.deepcopy(bc); c['train']['items']['exclude'] = [t]
            assert len(train.training_batches(o, X, D, c['train']['items'])) == 1 + 6          # 自检：来源只剩六个区，不含目标区 t
            res = train.fit(o, X, D, c, bc['train']['seeds'])
            np.savez_compressed(out / 'inner' / f'{t}__{o}.npz', probs=np.stack([res[s][1] for s in bc['train']['seeds']]).mean(0))
            print('嵌套', t, o, f'{time.time() - t0:.0f}s', flush=True)
    inner = {(t, o): np.load(out / 'inner' / f'{t}__{o}.npz')['probs'] for t in ALL for o in ALL if o != t}

    # 自检：修正头训练前输出等于原模型
    m0 = correction.Correction(len(correction.FEATURES), 0, cfg['correction']['width'])
    checks['训练前与原模型最大差'] = {arm: max(float(np.abs(correction.apply(m0, D[d], base[d], M[d], sc) - base[d]).max()) for d in ALL) for arm, sc in ARMS.items()}

    # 2. 修正头
    P = {arm: {} for arm in ARMS}
    for t in ALL:
        sets = [(D[o], inner[(t, o)], M[o], D[o].train_answers) for o in ALL if o != t]
        for arm, sc in ARMS.items():
            rows = M[t].any(1) if sc else np.ones(D[t].n, bool); outs = []
            for seed in cfg['correction']['seeds']:
                outs.append(correction.apply(correction.fit(sets, cfg['correction'], seed, sc), D[t], base[t], M[t], sc)[rows])
            Pf = base[t].copy(); Pf[rows] = np.mean(outs, 0); P[arm][t] = Pf
            (out / 'probs' / arm).mkdir(parents=True, exist_ok=True); np.savez_compressed(out / 'probs' / arm / f'{t}.npz', probs=Pf)
        print('修正', t, f'{time.time() - t0:.0f}s', flush=True)

    # 3. 评价
    rows = {'原模型': []} | {arm: [] for arm in ARMS}; hit = []
    for d in ALL:
        rows['原模型'] += evaluate.correct(base[d], D[d])
        for arm in ARMS: rows[arm] += evaluate.correct(P[arm][d], D[d])
        hit += [bool(M[d][e['q']].any()) for e in D[d].eval_rows]
    hit = np.array(hit)
    nohit_same = all(np.array_equal(P['限定范围'][d][~M[d].any(1)], base[d][~M[d].any(1)]) for d in ALL)
    checks['限定范围：没有名字证据的小区概率与原模型逐位相同'] = bool(nohit_same)
    checks['限定范围：没有名字证据的评价小区对错与原模型相同'] = bool(all(rows['限定范围'][i]['ok'] == rows['原模型'][i]['ok'] for i in np.flatnonzero(~hit)))
    ev = cfg['evaluate']
    res = dict(checks=checks, summary={k: evaluate.summary(v) for k, v in rows.items()},
               compare={'限定范围−原模型': evaluate.compare(rows['限定范围'], rows['原模型'], ev), '不限定−原模型': evaluate.compare(rows['不限定'], rows['原模型'], ev),
                        '限定范围−不限定': evaluate.compare(rows['限定范围'], rows['不限定'], ev)})
    sub = {}
    for name, sel in [('带名字证据', hit), ('不带名字证据', ~hit)]:
        ix = np.flatnonzero(sel); b = np.array([rows['原模型'][i]['ok'] for i in ix])
        sub[name] = dict(n=int(len(ix)), 原模型判对=int(b.sum()))
        for arm in ARMS:
            x = np.array([rows[arm][i]['ok'] for i in ix]); sub[name][arm] = dict(判对=int(x.sum()), 改对=int((x & ~b).sum()), 改坏=int((~x & b).sum()))
    res['按有无名字证据'] = sub
    (out / 'eval.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
