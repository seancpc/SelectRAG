"""語言漂移(簡體字元佔比)——確定性字表,零依賴(2026-09-17 核准,方案 A)。

字表原則:只收「簡體專用、非合法繁體字」的常用簡化字;簡繁共用字(后、云、里、干、面、几、
丑、谷、斗、周、系、缺、没、床、强、户、温、黄、决、余、骨、呼、叠、翻、禁、咨、表…)一律不收,
寧漏勿誤——量到的是簡體字的下界。字表覆蓋率是動工前列出的高風險假設,由真實輸出人工抽驗。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TABLE = (
    "们这为个时间说会对应药医疗诊剂参资体关于与数据见观认识论议记讲语词读调试询请谢该设计让许证评译谁谈诉课谓谨谱"
    "经给结组级线练统继绝维绍络缩编缓纯纳纸纪约红纤织绩细纲终绿缘绳紧"
    "车轻转输载较辆轮软辑辅轨轩轴辈辖辗"
    "门问闭闻阅闲阔闪闯闰闷闸阀阁阐"
    "马驱验驾骂驰驶驻骗骤"
    "鱼鲜鲁鸟鸡鸭鸣"
    "贝财贡员责货质购贵费贴贸贷账赛赞赠赔负贯赵"
    "视规觉览"
    "页顶项顺须预领颈频题颗额颜颁顿"
    "风飞饭饮饱饰馆饲饥饶"
    "长张东"
    "钟钙钠钾钢铁铜银锌锅错键锁镁镜针钉钱铅链镇"
    "电庆广库兰"
    "仅仪优伤价众传伦伪侧侨侦俭债倾偿储儿处备复学劳势动励区华协卖厂压厌历厉发"
    "号叶吗听吨呜响哑园团围国图圆场块坏坚坛垫声夹夺奋奖妇妈娱孙宁宝实审宪宽寻导尔尽层属岁岛"
    "币师帅带帮异弃弹归当录态总恶恳悬惊惯愿战执扩扫扬抢报拟择挤损换搁摆敌断无旧昼显晒"
    "术权杂条来极构枪标栏树样检桥档欢毁气汇汉沟泪洁测济浓涂润涨渐湿满滤滨灭灯灿炼烂热烧"
    "爱牵犹状独狮献环现玛产画疟疮痪监盖盘矿码础礼祸祷种积称稳穷窃窍竞笔筛签简类粮"
    "罚罢罗羡职联肠肤肿胀脉脏脑脱脸舰节芦苏范荐荣莱获萨蓝虏虑虽蚕蚀补装"
    "订讨训讯讼访诗详误诸诺谊"
    "边达迁过运还进远违连迟适选逊递遗邓邮邻释"
    "队阳阴际陆陈险随隐难雾韩"
    "办义专业严丰临举么乡书买乱亚亲亿从仓兴农况减净凤刘则创划剧"
    "齐齿龄龙麦"
)
SIMPLIFIED_ONLY: frozenset[str] = frozenset(_TABLE)


def _is_cjk(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF


def simplified_ratio(text: str) -> float:
    """CJK 字元中落在簡體專用字表的比例;無 CJK 字元回 0.0。"""
    cjk = [c for c in text if _is_cjk(c)]
    if not cjk:
        return 0.0
    return sum(1 for c in cjk if c in SIMPLIFIED_ONLY) / len(cjk)


def drift_summary(texts: Mapping[str, str]) -> dict[str, Any]:
    """每題佔比 → 題數、含簡體題數、平均、最大、含簡體 id(排序)。"""
    ratios = {k: simplified_ratio(v) for k, v in texts.items()}
    ids = sorted(k for k, r in ratios.items() if r > 0)
    ratio_max = max(ratios.values()) if ratios else 0.0
    # 浮點加總可讓 mean 高出 max 一個 ulp(三筆 0.4 → 0.4000000000000001);
    # mean ≤ max 是定義上的不變量,截回
    ratio_mean = min(sum(ratios.values()) / len(ratios), ratio_max) if ratios else 0.0
    return {
        "n": len(ratios),
        "texts_with_simplified": len(ids),
        "ratio_mean": ratio_mean,
        "ratio_max": ratio_max,
        "ids": ids,
    }
