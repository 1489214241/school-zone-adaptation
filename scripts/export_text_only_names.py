"""一次性：给杭州四区导出"只用范围文字"的名单点名，写 data/<区>/observations/names_policy_text.json。

杭州现在的名单点名有两个来源（科研推进/杭州四区主模型_20261024/v1/policy_rules/）：
  范围文字里点名的楼盘（scope_text_named_rows），以及入学平台另给的逐校"户籍生点名名单"（official_named_scopes.json）。
这里只取前者，用与原构建脚本 build_named_rules.py 相同的合并方法（同一坐标组：正例取并集；同一学校既有正例又有排除时回到未知、记冲突）。
先用同一方法把两个来源合起来的结果还原一遍，必须与现在用的名单逐格相同。
用法：D:/ana/python.exe -X utf8 scripts/export_text_only_names.py
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]; ROOT = REPO.parent
V1 = ROOT / '科研推进/杭州四区主模型_20261024/v1/policy_rules'; INP = ROOT / '杭州市/四区数据集_20261024_v3/实验输入'
rd = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))


def aggregate(named_rows, rows, gids):
    gi = {g: i for i, g in enumerate(gids)}; by = defaultdict(list)
    for i, r in enumerate(rows): by[r['coordinate_group_id']].append(i)
    named = np.full((len(gids), named_rows.shape[1]), -1, np.int8); conflict = np.zeros(named.shape, bool)
    for g, ii in by.items():
        k = gi[g]; pos = (named_rows[ii] == 1).any(0); neg = (named_rows[ii] == 0).any(0)
        named[k, pos & ~neg] = 1; named[k, neg & ~pos] = 0; conflict[k] = pos & neg
    return named, conflict


def main():
    rows = rd(INP / '小区候选点.json'); schools = rd(INP / '学校表.json'); grules = rd(V1 / 'coordinate_group_rules.json')
    gids = [r['coordinate_group_id'] for r in grules]
    with np.load(V1 / 'arrays.npz') as z: full, full_rows, text_rows = z['named'], z['named_rows'], z['scope_text_named_rows']
    assert full_rows.shape == text_rows.shape == (len(rows), len(schools)) and full.shape == (len(gids), len(schools))
    named_chk, conf_chk = aggregate(full_rows, rows, gids)
    assert np.array_equal(named_chk, full), '合并方法还原不出现在的名单'
    sids = [s['school_id'] for s in schools]
    assert all(set(np.array(sids)[conf_chk[k]]) == set(r['conflicting_positive_exclusion_school_ids']) for k, r in enumerate(grules)), '冲突还原不一致'
    named, conflict = aggregate(text_rows, rows, gids)
    for d in ['拱墅区', '萧山区', '富阳区', '临安区']:
        cols = [j for j, s in enumerate(schools) if s['district_name'] == d]
        obs = rd(REPO / 'data' / d / 'observations/observations.json')
        old = {int(q): v for q, v in obs['name_known'].items()}
        assert old == {q: np.flatnonzero(full[q, cols] == 1).tolist() for q in range(len(gids)) if (full[q, cols] == 1).any()}, d
        out = dict(source='只用范围文字的名单点名（scope_text_named_rows），合并方法同 build_named_rules.py',
                   name_known={str(q): np.flatnonzero(named[q, cols] == 1).tolist() for q in range(len(gids)) if (named[q, cols] == 1).any()},
                   name_conflict=[q for q in range(len(gids)) if conflict[q, cols].any()],
                   name_negative=[[int(q), int(s)] for q, s in zip(*np.nonzero(named[:, cols] == 0))])
        (REPO / 'data' / d / 'observations/names_policy_text.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
        print(d, '名单点名：现在', len(old), '只用范围文字', len(out['name_known']), '冲突', len(out['name_conflict']), '排除', len(out['name_negative']))


if __name__ == '__main__':
    main()
