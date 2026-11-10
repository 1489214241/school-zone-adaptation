"""一次性数据导出：从项目里分散的旧程序与旧数据，导出本代码库使用的统一数据格式（data/）。
这是整个代码库里唯一引用旧代码的地方；导出之后，zoning/ 下的代码只读 data/。

每个区导出到 data/<区>/：
  district.json        区名、城市、坐标系、小区名、学校名、（杭州）小区在全市原始编号
  arrays.npz           小区坐标 coords、楼盘组 groups、校址 school_xy
  observations/        政策文字的原子观察（统一证据执行器第一层的输出）：observations.json、arrays.npz
  exceptions.json      已登记的例外：不当参照、不当训练题的小区（天心 10/11 个只有名单关系、没有位置依据的小区）
  answers.json         官方答案：全部有答案的小区；评价用的规则未知小区及其空间块；别区当训练题用的答案；杭州有地图的学校
  extra_evidence.npz   两类由旧程序算好的证据：社区（community_name、community_strict、community_school[n,S,4]）、名字（name_match[n,S,4]）
另导出 data/lines/<城市>.pkl（OSM 道路、河流、铁路线）。
用法：D:/ana/python.exe -X utf8 scripts/export_data.py
"""
import os, sys, json, shutil, pickle
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; ROOT = REPO.parent; DATA = REPO / 'data'
TWO = ROOT / '科研推进/两城训练检验_20261103'; EXE = ROOT / '科研推进/统一证据执行器_20261105'; BR = ROOT / '科研推进/边界与社区证据参照模型_20261103'
os.chdir(TWO); sys.path.insert(0, str(TWO)); sys.path.insert(0, str(BR))
import city, feats  # noqa: E402


def export(d):
    g, gold, ev, scope = city.load(d); out = DATA / d; out.mkdir(parents=True, exist_ok=True)
    hz = d in city.HZ
    names = [s.get('name', '') for s in g['sites']]
    meta = dict(name=d, city=g['city'], crs=int(g['crs']), n=int(g['n']), school_n=int(g['school_n']), site_names=names,
                school_names=[str(s) for s in g['school_names']], orig=[int(x) for x in g['orig']] if hz else None)
    (out / 'district.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(out / 'arrays.npz', coords=np.asarray(g['coords'], np.float64), groups=np.asarray(g['groups']).astype(str), school_xy=np.asarray(g['school_xy'], np.float64))
    if (out / 'observations').exists(): shutil.rmtree(out / 'observations')
    shutil.copytree(EXE / 'observations' / d, out / 'observations')
    # 例外：规则已知但不当参照 / 不当训练题（由旧输入里的参照表与训练表反推，逐个列出）
    known = {int(q): sorted(int(s) for s in v) for q, v in g['known_sets'].items()}
    ref_sites = {int(r['global_i']) for r in g['refs']}; sup = {int(q) for q in g['supervised_ids']}
    single = {q for q, v in known.items() if len(v) == 1}
    exc = dict(not_reference=sorted(q for q in known if q not in ref_sites), not_training=sorted(single - sup))
    exc['names'] = {str(q): names[q] for q in exc['not_reference'] + exc['not_training']}
    (out / 'exceptions.json').write_text(json.dumps(exc, ensure_ascii=False, indent=1), encoding='utf-8')
    # 官方答案
    if hz:
        sid = {s: i for i, s in enumerate(city.ANS['school_ids'][d])}; pos = {int(o): i for i, o in enumerate(g['orig'])}
        allans = {pos[r['site']]: sorted(sid[t] for t in r['truth']) for r in city.ANS['rows'] if r['district'] == d and r['site'] in pos}
    else:
        nm = [city.EV.legacy.canonical(s, d, city.ALIASES) for s in g['school_names']]; idx = {n: i for i, n in enumerate(nm)}
        allans = {int(r['index']): sorted(idx.get(t, -1) for t in r['truth']) for r in city.CSROWS if r['district'] == d and r['has_gold']}
    ans = dict(all={str(q): v for q, v in sorted(allans.items())},
               eval=[dict(q=int(e['q']), truth=sorted(int(t) for t in e['truth']), block=str(e['block'])) for e in ev],
               train={str(q): int(s) for q, s in sorted(gold.items())}, scope=sorted(int(s) for s in scope) if scope is not None else None)
    (out / 'answers.json').write_text(json.dumps(ans, ensure_ascii=False), encoding='utf-8')
    # 社区与名字证据
    z = np.load(TWO / 'features' / f'{d}.npz'); ae = z['anchor_extra']
    cname = np.array([''] * g['n'], dtype=object); strict = np.zeros(g['n'], bool)
    if z['pair_extra'][..., 6:8].any():
        import comm
        inf = comm.infer(g['coords'], comm.points(d))
        cname = np.array([x[0] or '' for x in inf], dtype=object); strict = np.array([x[0] is not None and x[2] == 0 and x[1] <= 400 for x in inf])
    np.savez_compressed(out / 'extra_evidence.npz', community_name=cname.astype(str), community_strict=strict,
                        community_school=ae[..., 6:10].astype(np.float32), name_match=ae[..., 10:14].astype(np.float32))
    print('导出', d, 'n', g['n'], '学校', g['school_n'], '答案', len(allans), '评价', len(ev), '例外', len(exc['not_reference']), len(exc['not_training']), flush=True)


if __name__ == '__main__':
    (DATA / 'lines').mkdir(parents=True, exist_ok=True)
    shutil.copy(TWO / 'lines_cs.pkl', DATA / 'lines' / '长沙.pkl'); shutil.copy(TWO / 'lines_hz.pkl', DATA / 'lines' / '杭州.pkl')
    for d in city.ALL: export(d)
    (DATA / 'README.txt').write_text('由 scripts/export_data.py 从项目旧程序导出；格式见该脚本开头说明。\n', encoding='utf-8')
