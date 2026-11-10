"""读 data/<区>/（由 scripts/export_data.py 导出）。"""
import json
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
CITIES = {'长沙': ['天心区', '芙蓉区', '雨花区', '开福区'], '杭州': ['拱墅区', '萧山区', '富阳区', '临安区']}
ALL = CITIES['长沙'] + CITIES['杭州']
EXTRA = {'杭州': ['上城区', '西湖区', '滨江区', '钱塘区']}   # 候补的杭州新区（科研推进/杭州换区尝试_20261107）；默认不用


def city_of(d):
    return '长沙' if d in CITIES['长沙'] else '杭州'


@dataclass
class District:
    name: str
    city: str
    crs: int
    site_names: list          # 小区名
    school_names: list        # 学校名
    orig: object              # 杭州：小区在全市原始编号（规则观察用这个编号）；长沙为 None
    coords: np.ndarray        # [n, 2] 米
    groups: np.ndarray        # [n] 楼盘组（同一楼盘的几个点）
    school_xy: np.ndarray     # [S, 2] 校址
    answers: dict             # {小区: [学校]} 全部有官方答案的小区
    eval_rows: list           # 评价用：规则未知、有官方答案的小区 [{q, truth, block}]
    train_answers: dict       # 本区作为"别区"时当训练题的官方答案 {小区: 学校}（规则未知、单一答案）
    scope: object             # 杭州：有官方地图的学校集合（评价只看这些）；长沙为 None
    not_reference: set        # 已登记例外：规则已知但不当参照
    not_training: set         # 已登记例外：规则已知但不当训练题
    community_name: np.ndarray    # [n] 小区所在社区（只雨花有；不过任何路的最近居委会点）
    community_strict: np.ndarray  # [n] 社区认定是否可靠（不隔路且 400 米内）
    community_school: np.ndarray  # [n, S, 4] 所在社区是否被该校整社区点名（只雨花）
    name_match: np.ndarray        # [n, S, 4] 小区名与该校政策名单部分对上
    answer_source: dict = None    # {小区: 'map' | 'list'}；只在"地图＋名单"口径下有（with_list_gold）
    lines_key: str = None         # 道路线文件名（data/lines/<名>.pkl）；缺省用城市名。杭州新区用 杭州_新区（科研推进/杭州换区尝试_20261107）

    @property
    def n(self): return len(self.coords)

    @property
    def school_n(self): return len(self.school_xy)

    @property
    def obs_dir(self): return DATA / self.name / 'observations'


_cache = {}


def load(d) -> District:
    if d in _cache: return _cache[d]
    f = DATA / d; meta = json.loads((f / 'district.json').read_text(encoding='utf-8'))
    a = np.load(f / 'arrays.npz'); e = np.load(f / 'extra_evidence.npz')
    ans = json.loads((f / 'answers.json').read_text(encoding='utf-8')); exc = json.loads((f / 'exceptions.json').read_text(encoding='utf-8'))
    x = District(name=d, city=meta['city'], crs=meta['crs'], site_names=meta['site_names'], school_names=meta['school_names'],
                 orig=np.array(meta['orig']) if meta['orig'] is not None else None,
                 coords=a['coords'], groups=a['groups'].astype(str), school_xy=a['school_xy'],
                 answers={int(q): v for q, v in ans['all'].items()}, eval_rows=ans['eval'],
                 train_answers={int(q): int(s) for q, s in ans['train'].items()},
                 scope=set(ans['scope']) if ans['scope'] is not None else None,
                 not_reference=set(exc['not_reference']), not_training=set(exc['not_training']),
                 community_name=e['community_name'].astype(str), community_strict=e['community_strict'],
                 community_school=e['community_school'], name_match=e['name_match'], lines_key=meta.get('lines'))
    _cache[d] = x
    return x


def with_list_gold(x: District) -> District:
    """杭州"地图＋名单"口径：没有可用地图的学校，用入学平台同一条记录里的小区名单补官方答案（scripts/export_list_gold.py 导出）。
    地图答案不动；名单答案标 source='list'。长沙原样返回。"""
    f = DATA / x.name / 'answers_list_gold.json'
    if x.city != '杭州' or not f.exists(): return x
    L = json.loads(f.read_text(encoding='utf-8'))['rows']
    answers = dict(x.answers); src = {q: 'map' for q in answers}
    for r in L:
        assert r['q'] not in answers; answers[r['q']] = r['truth']; src[r['q']] = 'list'
    return replace(x, answers=answers, answer_source=src)


def with_mca_community(x: District) -> District:
    """杭州社区证据：范围描述里点名的社区、村 ＋ 民政部地名库的社区/村点，小区归最近的社区/村点（科研推进/杭州社区村条款_20261107）。长沙原样返回。"""
    f = DATA / x.name / 'community_mca.npz'
    if x.city != '杭州' or not f.exists(): return x
    z = np.load(f, allow_pickle=False)
    return replace(x, community_name=z['community_name'].astype(str), community_strict=z['community_strict'], community_school=z['community_school'])
