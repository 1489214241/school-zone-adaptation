"""核对本代码库与项目旧程序（两城最终版）是否算得一样。只有这个脚本和 export_data.py 引用旧代码。
  python scripts/verify.py rules      第一、二步：规则判出、四至状态、排除逐格比对
  python scripts/verify.py evidence   第三步：按旧列布局重建输入，与旧程序的模型输入逐位比对
  python scripts/verify.py train      第四、五步：按旧列布局训练一个区一个种子，与旧程序的已存预测比对
  python scripts/verify.py import     把旧程序的两城最终版预测（10 个种子）存成本代码库的输出格式 outputs/legacy_final/，供 run.py --compare
"""
import os, sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; ROOT = REPO.parent; sys.path.insert(0, str(REPO))
import numpy as np
from zoning import data, rules, evidence, train
TWO = ROOT / '科研推进/两城训练检验_20261103'; A_PRED = ROOT / '科研推进/训练题消融_20261106/predictions'


def legacy():
    os.environ['BRWQ'] = '1'; cwd = os.getcwd(); os.chdir(TWO); sys.path.insert(0, str(TWO))
    import city, train as T  # noqa
    os.chdir(cwd); return city, T


def check_rules():
    city, _ = legacy(); res = {}
    for d in data.ALL:
        g, _, _, _ = city.load(d); r = rules.run(data.load(d), 'final_20261103')
        k_old = {int(q): sorted(int(s) for s in v) for q, v in g['known_sets'].items()}
        neg_old = np.asarray(g['negative']).astype(bool); neg_new = r.negative if d in data.CITIES['长沙'] else np.zeros_like(neg_old)
        res[d] = dict(known_same=r.known == k_old, status_cells_diff=int((r.statuses != g['statuses']).sum()), negative_cells_diff=int((neg_new != neg_old).sum()))
        print('规则', d, res[d], flush=True)
    return res


def legacy_inputs(patch_negative=True):
    """本代码库按旧列布局建的输入；天心的排除按旧输入打 1 格补丁（旧输入的白沙公寓一格是改名前算的，统一执行器判得对）。"""
    city, _ = legacy(); cfg = dict(layout='legacy', exclusion_to_model='changsha_only'); X = {}; D = {}
    for d in data.ALL:
        D[d] = data.load(d); r = rules.run(D[d], 'final_20261103')
        if patch_negative and d in data.CITIES['长沙']:
            g, _, _, _ = city.load(d); r.negative = np.asarray(g['negative']).astype(bool)
        X[d] = evidence.build(D[d], r, cfg)
    return X, D


def check_evidence():
    city, T = legacy(); X, _ = legacy_inputs(); res = {}
    for d in data.ALL:
        dt = T.data(d); pm, am = dt.masks(T.FULL); pm, am = pm.numpy().astype(bool), am.numpy().astype(bool); x = X[d]
        same_refs = np.array_equal(dt.ri, x.ref_sites) and np.array_equal(dt.rs, x.ref_schools)
        res[d] = dict(refs_same=bool(same_refs), group_bias_same=bool(np.array_equal(dt.group_bias, x.group_bias)),
                      supervised_same=bool(np.array_equal(np.sort(dt.supervised), x.supervised)), y_same=bool(np.array_equal(dt.y, x.y)),
                      pair_maxdiff=float(np.abs(dt.pair * pm - x.pair).max()), anchor_maxdiff=float(np.abs(dt.anchor * am - x.anchor).max()))
        print('证据', d, res[d], flush=True)
    return res


def check_train(d='芙蓉区', seed=20261131):
    X, D = legacy_inputs(); cfg = json.loads((REPO / 'configs/final_20261103.json').read_text(encoding='utf-8'))
    P = train.fit(d, X, D, cfg, [seed])[seed][1]; old = np.load(A_PRED / d / 'A' / str(seed) / 'predictions.npz')['scores']
    q = [e['q'] for e in D[d].eval_rows]
    r = dict(district=d, seed=seed, max_prob_diff=float(np.abs(P - old).max()), answers_differ=int((P[q].argmax(1) != old[q].argmax(1)).sum()))
    print('训练', r, flush=True); return r


def import_legacy():
    out = REPO / 'outputs' / 'legacy_final' / 'probs'; out.mkdir(parents=True, exist_ok=True); seeds = list(range(20261131, 20261141))
    for d in data.ALL:
        np.savez_compressed(out / f'{d}.npz', probs=np.stack([np.load(A_PRED / d / 'A' / str(s) / 'predictions.npz')['scores'] for s in seeds]), seeds=seeds)
    return dict(saved=str(out))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'rules'
    r = {'rules': check_rules, 'evidence': check_evidence, 'train': check_train, 'import': import_legacy}[what]()
    (REPO / 'outputs').mkdir(exist_ok=True); (REPO / 'outputs' / f'verify_{what}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
