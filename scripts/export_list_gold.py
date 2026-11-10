"""一次性：杭州"地图＋名单"口径的官方答案。写 data/<区>/answers_list_gold.json。

现在杭州的官方答案只来自入学平台的学区地图（小区代表点落在哪所学校的地图里）。约一半学校没有可计算的地图。
这里对没有可用地图的学校，用同一条平台记录里的小区名单补答案：
  - 只用名单里严格按全名对上我们小区点的行（科研推进/杭州四区主模型_20261024/v1/policy_rules/named_match_evidence.json 里
    evidence_type = official_household_scope_rows 且 state = positive；村、社区、门牌、带楼栋限制、跨区同名等都是 review，不用）；
  - 地图优先：这个小区已经有地图答案的，不动；
  - 只补"名单点名的学校全都没有可用地图"的小区；名单同时点了有地图学校的（点却不在那所学校地图里），跳过；
  - 名单点了多所没地图的学校，全部保留（多校）；
  - 空间块：1 公里网格（与现有答案相同，已核对八区逐个一致）。
用法：D:/ana/python.exe -X utf8 scripts/export_list_gold.py
"""
import sys, json
from collections import defaultdict, Counter
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; ROOT = REPO.parent; sys.path.insert(0, str(REPO))
from zoning import data

V1 = ROOT / '科研推进/杭州四区主模型_20261024/v1/policy_rules'
ANS = json.loads((ROOT / '科研推进/长沙杭州统一基准_20261030/package/evaluation/answers.json').read_text(encoding='utf-8'))


def main():
    ev = json.loads((V1 / 'named_match_evidence.json').read_text(encoding='utf-8'))
    gids = [r['coordinate_group_id'] for r in json.loads((V1 / 'coordinate_group_rules.json').read_text(encoding='utf-8'))]; gi = {g: i for i, g in enumerate(gids)}
    pos = defaultdict(set)
    for x in ev:
        if x['evidence_type'] == 'official_household_scope_rows' and x['state'] == 'positive': pos[gi[x['coordinate_group_id']]].add(x['school_id'])
    for d in ['拱墅区', '萧山区', '富阳区', '临安区']:
        D = data.load(d); sids = ANS['school_ids'][d]; sidx = {s: i for i, s in enumerate(sids)}
        assert len(sids) == D.school_n
        mapped = set(D.scope); here = {int(o): i for i, o in enumerate(D.orig)}
        rows, c = [], Counter()
        for g, ss in sorted(pos.items()):
            ss = {sidx[s] for s in ss if s in sidx}                                        # 本区学校
            if not ss or g not in here: continue
            q = here[g]
            if q in D.answers: c['已有地图答案（地图优先，不动）'] += 1; continue
            if ss & mapped: c['名单点了有地图的学校（跳过）'] += 1; continue
            x, y = D.coords[q]
            rows.append(dict(q=q, truth=sorted(ss), block=str((int(x // 1000), int(y // 1000)))))
            c['新增名单答案'] += 1; c['其中多校'] += len(ss) > 1
        out = dict(rule=__doc__.split('\n\n')[1].strip(), counts=dict(c), rows=rows)
        (REPO / 'data' / d / 'answers_list_gold.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
        print(d, dict(c))


if __name__ == '__main__':
    main()
