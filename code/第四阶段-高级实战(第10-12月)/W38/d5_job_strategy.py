"""
W38 Day 5 - 求职策略
====================
主题: 目标公司分析模板, 渠道选择, 时间规划
"""

import os
import json
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 目标公司分析
# =============================================
def analyze_target_companies():
    """分析目标公司"""
    print_section("目标公司分析模板")

    company_template = {
        "AI岗位分类": {
            "大厂AI团队": {
                "代表公司": ["字节跳动", "阿里巴巴", "腾讯", "百度", "华为"],
                "特点": "AI投入大, 团队强, 要求高",
                "适合岗位": "AI算法工程师, AI应用工程师",
                "优势": "技术氛围好, 成长快, 薪资高",
                "挑战": "竞争激烈, 可能需要算法研究能力"
            },
            "AI创业公司": {
                "代表公司": ["月之暗面", "智谱AI", "MiniMax", "百川智能"],
                "特点": "聚焦AI, 速度快, 灵活",
                "适合岗位": "AI应用工程师, LLM开发工程师",
                "优势": "成长空间大, 接触前沿, 股权激励",
                "挑战": "风险较高, 加班多, 稳定性差"
            },
            "传统企业AI转型": {
                "代表公司": ["银行/保险AI部门", "制造业AI部门", "零售/电商AI"],
                "特点": "AI应用场景明确, 节奏相对稳定",
                "适合岗位": "AI应用工程师, 数据科学家",
                "优势": "业务场景丰富, 工作生活平衡",
                "挑战": "技术栈可能不够前沿, 晋升慢"
            },
            "外企AI团队": {
                "代表公司": ["Google", "Microsoft", "Amazon", "Apple"],
                "特点": "技术领先, 文化好, 要求高",
                "适合岗位": "ML Engineer, AI Engineer",
                "优势": "技术好, culture好, WLB好",
                "挑战": "英语要求, 可能需要出国, 竞争激烈"
            }
        },
        "公司分析维度": [
            "1. 公司AI战略和投入程度",
            "2. 团队规模和技术氛围",
            "3. 具体的AI应用场景和产品",
            "4. 技术栈和工具",
            "5. 薪资范围和福利",
            "6. 成长空间和晋升路径",
            "7. 公司文化和工作节奏",
            "8. 地点要求和远程政策"
        ],
        "公司调研清单": {
            "基本信息": ["公司规模", "融资轮次", "AI团队规模", "核心产品"],
            "技术信息": ["使用的技术栈", "技术博客", "开源项目", "论文发表"],
            "文化信息": ["工作节奏", "加班情况", "远程政策", "团队氛围"],
            "薪资信息": ["薪资范围", "年终奖", "股票/期权", "其他福利"]
        }
    }

    for category, details in company_template.items():
        print(f"\n【{category}】")
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, list):
                            print(f"    {k}: {', '.join(v)}")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    {item}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(details, list):
            for item in details:
                print(f"  {item}")


# =============================================
# 2. 求职渠道
# =============================================
def select_job_channels():
    """选择求职渠道"""
    print_section("求职渠道选择")

    channels = {
        "内推(推荐指数: ★★★★★)": {
            "来源": ["前同事/同学", "技术社区", "LinkedIn", "猎头"],
            "优势": "简历直达, 成功率高, 流程快",
            "建议": "优先使用内推, 成功率是海投的3-5倍"
        },
        "招聘平台(推荐指数: ★★★★☆)": {
            "平台": [
                "Boss直聘 - 直接沟通, 响应快",
                "拉勾网 - 互联网岗位为主",
                "猎聘 - 中高端岗位",
                "LinkedIn - 外企和国际化公司",
                "脉脉 - 职场社交+内推"
            ]
        },
        "公司官网(推荐指数: ★★★★☆)": {
            "说明": "直接在目标公司官网投递",
            "优势": "信息准确, 可以了解最新职位"
        },
        "猎头(推荐指数: ★★★☆☆)": {
            "说明": "通过猎头寻找机会",
            "优势": "匹配度高, 可以了解薪资范围",
            "注意": "选择专业AI领域的猎头"
        },
        "技术社区(推荐指数: ★★★★☆)": {
            "渠道": [
                "GitHub Jobs",
                "Hugging Face Jobs",
                "技术Meetup招聘",
                "Discord社区招聘频道"
            ]
        },
        "社交网络(推荐指数: ★★★☆☆)": {
            "渠道": ["Twitter/X技术圈", "知乎", "微信技术群"],
            "说明": "展示技术能力, 吸引招聘方"
        }
    }

    for channel, details in channels.items():
        print(f"\n【{channel}】")
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")

    # 投递策略
    print("\n--- 投递策略 ---")
    strategies = [
        "1. 梯度投递: 先投一般公司练手, 再投心仪公司",
        "2. 优先内推: 所有目标公司都先找内推",
        "3. 定制简历: 根据JD调整简历关键词和项目顺序",
        "4. 跟进: 投递后3天没回复可以主动跟进",
        "5. 多线并行: 同时推进多家公司, 不要只等一家",
        "6. 记录: 用表格记录每家公司的状态和进度"
    ]
    for s in strategies:
        print(f"  {s}")


# =============================================
# 3. 时间规划
# =============================================
def create_job_timeline():
    """创建求职时间规划"""
    print_section("求职时间规划")

    timeline = {
        "准备阶段(第1-2周)": [
            "完善简历(STAR法则, 量化成果)",
            "搭建个人网站/GitHub展示",
            "准备自我介绍和项目介绍",
            "整理目标公司清单(10-15家)",
            "寻找内推渠道",
            "刷面试题(每天10题)"
        ],
        "投递阶段(第3-4周)": [
            "投递第一批公司(B级公司, 练手)",
            "参加第一批面试(积累经验)",
            "根据面试反馈调整简历和准备",
            "投递第二批公司(A级公司, 目标)",
            "继续刷题和模拟面试"
        ],
        "面试阶段(第5-8周)": [
            "集中面试(每周3-5场)",
            "记录每场面试的问题和表现",
            "复盘改进, 针对性补强",
            "推进到终面的公司重点准备",
            "保持学习和项目更新"
        ],
        "Offer阶段(第9-10周)": [
            "收Offer, 对比分析",
            "薪资谈判(见Day 6)",
            "做出选择",
            "背景调查准备",
            "提交离职申请(如需要)"
        ]
    }

    for phase, tasks in timeline.items():
        print(f"\n【{phase}】")
        for task in tasks:
            print(f"  {task}")

    # 面试进度跟踪模板
    tracking_template = {
        "公司名称": "",
        "职位": "",
        "投递日期": "",
        "投递渠道": "内推/官网/招聘平台",
        "当前状态": "已投递/笔试/一面/二面/HR面/Offer",
        "下次面试时间": "",
        "面试记录": [
            {"轮次": "一面", "日期": "", "面试官": "", "问题": "", "表现": ""}
        ],
        "薪资范围": "",
        "优先级": "高/中/低",
        "备注": ""
    }

    print("\n--- 面试进度跟踪模板 ---")
    print(json.dumps(tracking_template, ensure_ascii=False, indent=2))

    output_file = os.path.join(os.path.dirname(__file__), "job_tracking_template.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tracking_template, f, ensure_ascii=False, indent=2)
    print(f"\n  跟踪模板已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W38 Day 5 - 求职策略")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    analyze_target_companies()
    select_job_channels()
    create_job_timeline()

    print("\n" + "=" * 60)
    print("  求职是一场战役, 好的策略事半功倍!")
    print("  建议: 今天整理目标公司清单, 找到内推渠道")
    print("=" * 60)
