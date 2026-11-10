"""2026 长沙逐小区政策关系：只读官方原文、政策观察和预先审核的实体证据。

不导入 data.load，不读 Gold、概率或评估文件。候选来自已知错误审计，因此不是盲测。
学校集合由下列政策词在 observations 的全部条款匹配得到，不手填 Gold 学校编号。
"""
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
OBS = REPO / 'data/雨花区/observations'
SOURCE = WORK / '长沙市/雨花区/原始资料/教育局2026_csv/关于2026年雨花区公办小学学区范围及覆盖具体楼盘信息的公告.csv'
SOURCE_SHA = '7c14c17ee85d94ae6c36e2397b7f28d0523e461d60bf2753dd0c233052a8ca07'
SOURCE_URL = 'http://www.yuhua.gov.cn/sjkf/shkfjy/xxxqfwlp/202604/P020260401531463878098.csv'
AUDIT = WORK / '科研推进/长沙多校关系补全_20261109'


def evidence(url, finding, level='public_source'):
    return dict(url=url, checked='2026-11-09', finding=finding, level=level)


RECORDS = [
    dict(q=250, name='江河双水湾', term='双水湾', status='approved',
         identity_basis='人工实体判定：项目介绍在江河双水湾页面内使用双水湾简称，地址沙湾路1088号、开发商江河水利置业相符；仅连接该住宅小区。',
         applicability='2026年双水湾住宅；只应用当前小区记录，不扩展至其他江河项目或商业物业。',
         evidence=[
             evidence('https://www.huxing.com/xiaoqu-3161dbe5367a11ee90eb00f1f30222b6-jiangheshuangshuiwan.html', '项目全称、简称、地址和开发商在同页；不采用该页学区字段。', 'secondary_identity'),
             evidence('https://mfang.58.com/cs/xinfang/xq100385662/xiangqing/', '独立房产目录地址为沙湾路1088号，开发商为江河水利置业。', 'secondary_identity'),
             evidence('https://www.sohu.com/a/1009035551_121124006', '署名学校党政办的更名公告转载：东澜湾更名为澜湾，学区范围不变。', 'school_notice_mirror'),
             evidence('https://www.sohu.com/a/1052975897_121677885', '教育资讯明确列出悦东方更名悦雅；属二手报道，并非更名公文。', 'secondary_rename'),
             evidence('https://mbaokao.offcn.com/ggcx/gg/181204.html', '来源标注学校公众号的2026招聘公告：悦东方小学电话0731-88307686，与官方2026CSV悦雅小学电话一致。', 'school_notice_mirror'),
         ]),
    dict(q=349, name='三0一佳园', term='三零一核工业部', status='approved',
         identity_basis='人工实体判定：三〇一佳园是核工业三〇一大队参与三供一业改造的既有小区；三0一、三零一为同址数字写法。',
         applicability='仅三〇一佳园既有宿舍小区；保留原有宿舍限定，不扩展到新增楼盘或核工业其他院区。',
         evidence=[
             evidence('https://cs.rednet.cn/content/2018/12/21/4349414.html', '长沙晚报报道三〇一大队投入三〇一佳园小区提质和三供一业改造，支持既有单位住宅身份。'),
             evidence('https://www.kdniao.com/kuaidi/network/430100000000_430111000000/9.html', '配送地址表将中意一路977号列作301佳园、三零一佳园，同时列三0一佳园；仅作地址与数字写法佐证。', 'secondary_address'),
         ]),
    dict(q=388, name='香樟路电信宿舍', term='香樟路电信局宿舍', status='approved',
         identity_basis='人工实体判定：保留香樟路和宿舍两个限定，电信/电信局为单位称谓差异；房产目录与市供水报道均将其作为当地既有住宅小区。长郡雨外为长郡雨花外国语简称，第三附属序号不变。',
         applicability='仅香樟路电信住宅宿舍；不适用于桔园路等其他电信宿舍、办公营业用房。',
         evidence=[
             evidence('https://www.fang.com/xiaoqu/cs-2710873508/', '小区在韶山南路与香樟路交叉口，1995年住宅、单位自管，支持宿舍属性。', 'secondary_identity'),
             evidence('https://www.icswb.com/h/151/20230407/808810.html', '长沙晚报转述市供水公司名单，将香樟路电信宿舍与瑞都华庭、湘樟园等分别列为用户。'),
             evidence('https://m.voc.com.cn/xhn/news/202304/16377146.html', '教育报道使用长郡雨外第三附属小学简称；与原文第三附属学校序号一致。'),
         ]),
    dict(q=435, name='中航长沙设计研究院宿舍', term='航三院职工宿舍', status='approved',
         identity_basis='人工实体判定：行业协会确认中航长沙设计研究院原名中国航空工业第三设计研究院、地址香樟路254号；同址住宅目录及社区活动报道支持航三院住宅身份。',
         applicability='仅香樟路254号院的职工住宅宿舍，不包括办公楼、轻工设计院或中航其他新项目。',
         evidence=[
             evidence('https://cskcsj.com/img1/showimg.php?id=420', '行业协会会员介绍明确原单位名称和香樟路254号地址。', 'industry_association'),
             evidence('https://mfang.58.com/cs/xinfang/xq100388158/xiangqing/', '中航工业长沙院为1998年住宅，地址香樟路254号（旧004号）。', 'secondary_identity'),
             evidence('https://www.hunantoday.cn/news/xhn/202405/19936297.html', '社区老年活动在香樟路社区航三院举行，中航长沙设计研究院共同实施。'),
         ]),
    dict(q=82, name='世茂璀璨天城', term='世贸云锦轩', status='pending',
         reason='云锦轩推广名已查到，但项目同时含住宅和商业公寓，当前点未区分用途；另一条款是2026年暂时过渡，不能直接视为无条件双校。',
         evidence=[evidence('https://m.lejucaijing.com/news-6977929197272743681.html', '转述住建局和区委办回复：云锦轩推广名为世茂璀璨天城，同一项目有10栋高层住宅及6栋商业。')]),
    dict(q=271, name='中建嘉和城凤凰台', term='中建凤凰台', status='pending',
         reason='中建凤凰台的单校条款成立；无法证明中建嘉和城整项目条款覆盖凤凰台。公共服务名单分别列两项目，不能按父项目名称包含关系补第二校。',
         evidence=[evidence('https://finance.sina.com.cn/jjxw/2026-08-02/doc-inikxfmf2917385.shtml', '市供水停水名单分别列中建嘉和城、中建嘉和城凤凰台。')]),
    dict(q=392, name='江河中央洋房', term='中央洋房', status='pending',
         reason='查到沙湾路389号、江河开发商和住宅属性，学校更名亦有佐证；本轮未取得政策简称中央洋房与当前住宅范围的直接对应证据，保留待核。',
         evidence=[evidence('https://www.focus.cn/xf/cs-181860/', '项目目录显示沙湾路389号、开发商江河水利置业；尚不足以单独闭合政策简称和住宅范围。', 'secondary_identity')]),
    dict(q=540, name='中国轻总会长沙设计院', term=None, status='pending',
         reason='未找到支持当前双校关系的政策点名；不能因坐标接近中航院而合并实体。红星小学另有轻工设计院宿舍条款，且当前记录未写宿舍，保留单位/住宅及坐标疑点。',
         evidence=[evidence('https://qkzzs.jcu.edu.cn/zgtcgy/info/1025/3372.htm', '期刊作者单位使用中国轻工总会长沙设计院，不能据此认作航空三院。')]),
]


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('官方原始政策发生变化，须重新审核')
    obs_raw = (OBS/'observations.json').read_bytes(); obs = json.loads(obs_raw)
    originals = []; school = ''
    for i, cells in enumerate(csv.reader(raw.decode('gb18030').splitlines()), 1):
        if len(cells) > 1 and cells[1].strip():
            school = cells[1].strip()
        originals.append(dict(csv_record=i, original_school=school, cells=cells))
    records = []
    for spec in RECORDS:
        r = dict(spec)
        if obs['names'][r['q']] != r['name']:
            raise ValueError('小区索引变化，须重新审核')
        term = r.pop('term')
        cs = [dict(school=c['school'], clause=c['clause'], text=c['text'])
              for c in obs['clauses'] if term and term in c['text']]
        quotes = [x for x in originals if term and any(term in cell for cell in x['cells'])]
        r.update(policy_term=term, clauses=cs, original_records=quotes)
        if r['status'] == 'approved':
            ss = sorted({c['school'] for c in cs})
            if len(ss) != 2 or len(quotes) != 2:
                raise ValueError('来源点名关系变化，须重新审核')
            r.update(schools=ss, school_names=[obs['school_names'][s] for s in ss],
                     relation_type='policy_named_overlap', choice_status='unconfirmed',
                     output_action='replace_with_policy_set')
        records.append(r)
    payload = dict(schema='changsha_verified_relations_v1', district='雨花区', policy_year=2026,
                   checked='2026-11-09', observations_sha256=hashlib.sha256(obs_raw).hexdigest(),
                   source=dict(path=str(SOURCE.relative_to(WORK)), url=SOURCE_URL, sha256=SOURCE_SHA),
                   selection_note='候选由已有多校错误审计定位；关系取自政策和外部实体证据，结果仅属同批数据开发验证。',
                   records=records)
    dest = OBS/'verified_multischool.json'
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT/'source_review.json').write_bytes(dest.read_bytes())
    print('approved', sum(r['status'] == 'approved' for r in records), 'pending', sum(r['status'] == 'pending' for r in records))


if __name__ == '__main__':
    main()
