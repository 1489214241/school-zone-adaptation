"""一次性：萧山范围描述点名的核实别名（科研推进/萧山点名别名修复_20261107/核实记录.json，百度地图检索 ≤300 米才采用）。
写 data/萧山区/observations/names_policy_text_aliases.json：{坐标组编号: [学校]}，规则在 hangzhou_names = policy_text_aliases 时并入只用范围描述的点名。
用法：D:/ana/python.exe -X utf8 scripts/export_xs_aliases.py
"""
import json
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; ROOT = REPO.parent
V1 = ROOT / '科研推进/杭州四区主模型_20261024/v1/policy_rules'


def main():
    rec = json.loads((ROOT / '科研推进/萧山点名别名修复_20261107/核实记录.json').read_text(encoding='utf-8'))
    gi = {r['coordinate_group_id']: i for i, r in enumerate(json.loads((V1 / 'coordinate_group_rules.json').read_text(encoding='utf-8')))}
    obs = json.loads((REPO / 'data/萧山区/observations/observations.json').read_text(encoding='utf-8')); si = {n: i for i, n in enumerate(obs['school_names'])}
    add = defaultdict(set)
    for r in rec:
        for g in r['accepted_groups']: add[gi[g]].add(si[r['school']])
    out = dict(source='萧山范围描述点名的核实别名（百度地图检索，≤300 米）', name_known={str(q): sorted(v) for q, v in sorted(add.items())})
    (REPO / 'data/萧山区/observations/names_policy_text_aliases.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print('别名点名：小区点', len(add), '（小区, 学校）对', sum(len(v) for v in add.values()))


if __name__ == '__main__':
    main()
