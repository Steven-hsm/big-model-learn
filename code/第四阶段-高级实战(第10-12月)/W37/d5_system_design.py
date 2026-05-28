"""
W37 Day 5 - 系统设计
====================
主题: 推荐系统设计模板, RAG系统设计, ML系统设计
"""

import os
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 推荐系统设计
# =============================================
def design_recommendation_system():
    """设计推荐系统"""
    print_section("推荐系统设计模板")

    design = {
        "1. 需求澄清(5分钟)": [
            "用户规模: 日活多少? 总用户数?",
            "物品规模: 商品/文章/视频数量?",
            "推荐类型: 首页推荐/相关推荐/个性化推送?",
            "实时性: 实时推荐还是离线推荐?",
            "指标: CTR? 转化率? 用户停留时间?"
        ],
        "2. 系统架构": {
            "整体架构": "数据层 -> 特征层 -> 召回层 -> 排序层 -> 重排层 -> 展示层",
            "数据层": [
                "用户行为数据: 点击、购买、浏览、收藏",
                "物品特征: 类别、标签、属性",
                "用户画像: 人口统计、兴趣标签",
                "社交数据: 关注、好友关系"
            ],
            "召回层(Recall)": [
                "协同过滤: User-CF / Item-CF",
                "内容匹配: 基于物品特征的匹配",
                "向量召回: 双塔模型(DSSM)",
                "热门/新物品召回",
                "多路召回, 每路100-500个候选"
            ],
            "排序层(Ranking)": [
                "粗排: 轻量模型快速筛选(如双塔)",
                "精排: 复杂模型精确排序(如DeepFM/DIN)",
                "特征: 用户特征 + 物品特征 + 交叉特征 + 上下文",
                "目标: CTR预估 / 多目标优化(CTR*CVR*价格)"
            ],
            "重排层(Re-ranking)": [
                "多样性: 避免同类物品过多",
                "新鲜度: 插入新物品",
                "业务规则: 去重、过滤已购、广告位",
                "探索与利用(E&E): 探索新兴趣"
            ]
        },
        "3. 技术选型": {
            "数据存储": "Redis(实时特征) + HBase(行为日志) + Hive(离线特征)",
            "特征工程": "Spark + Flink(实时特征)",
            "模型训练": "PyTorch + PySpark MLlib",
            "向量检索": "Faiss / Milvus",
            "在线服务": "TensorFlow Serving / 自研推理框架",
            "A/B测试": "自研分流平台"
        },
        "4. 关键挑战与解决方案": {
            "冷启动": "新用户: 热门推荐 -> 快速收集行为; 新物品: 内容特征匹配",
            "数据稀疏": "矩阵分解 + 侧信息 + 迁移学习",
            "实时性": "Flink实时特征 + 在线学习",
            "可解释性": "提供推荐理由, 增强用户信任",
            "反馈循环": "探索与利用平衡, 避免信息茧房"
        },
        "5. 指标监控": [
            "离线: AUC, NDCG, Recall@K",
            "在线: CTR, 转化率, GMV, 用户停留时长",
            "系统: 推荐延迟(<100ms), 系统可用性(99.9%)"
        ]
    }

    for section, content in design.items():
        print(f"\n【{section}】")
        if isinstance(content, list):
            for item in content:
                print(f"  {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")


# =============================================
# 2. RAG系统设计
# =============================================
def design_rag_system():
    """设计RAG系统"""
    print_section("RAG系统设计模板")

    design = {
        "1. 需求分析": [
            "文档规模: 多少文档? 多少种格式?",
            "查询类型: 事实问答/总结/比较/多步推理?",
            "性能要求: 延迟? 并发? 准确率?",
            "用户规模: 日活? 峰值QPS?"
        ],
        "2. 数据处理Pipeline": {
            "文档解析": [
                "PDF: PyMuPDF / pdfplumber",
                "Word: python-docx",
                "Markdown: mistune",
                "HTML: BeautifulSoup",
                "图片/表格: OCR + 表格识别"
            ],
            "分块策略": [
                "固定大小分块: chunk_size=512, overlap=50",
                "语义分块: 按段落/章节边界分割",
                "递归分块: RecursiveCharacterTextSplitter",
                "关键: 保持语义完整性, 避免切断上下文"
            ],
            "向量化": [
                "Embedding模型: text2vec-base-chinese / BGE / OpenAI",
                "维度: 768(BERT) / 1024 / 1536(OpenAI)",
                "批量处理提高效率",
                "元数据: 保存来源、页码、标题等"
            ]
        },
        "3. 检索策略": {
            "基础检索": "向量相似度搜索(余弦/L2)",
            "混合检索": [
                "向量检索: 语义相似",
                "BM25: 关键词精确匹配",
                "融合: Reciprocal Rank Fusion(RRF)"
            ],
            "高级优化": [
                "Query改写: 扩展/简化用户查询",
                "HyDE: 先生成假设文档再检索",
                "重排序: Cross-Encoder精排",
                "多查询: 分解复杂查询为子查询"
            ]
        },
        "4. 生成策略": {
            "Prompt设计": [
                "系统提示: 角色定义和行为规范",
                "上下文注入: 检索到的文档片段",
                "引用要求: 标注信息来源",
                "不确定性: 允许回答'我不知道'"
            ],
            "模型选择": [
                "闭源: GPT-4(效果好), Claude(长文本)",
                "开源: LLaMA/Qwen(可控, 私有部署)",
                "考虑: 成本、延迟、效果、数据安全"
            ]
        },
        "5. 系统架构": {
            "API层": "FastAPI + 流式响应(SSE)",
            "业务层": "查询解析 -> 检索 -> 重排 -> 生成",
            "数据层": "向量数据库 + 文档存储 + 缓存(Redis)",
            "推理层": "vLLM/TGI模型服务",
            "监控": "延迟、Token消耗、用户反馈、准确率"
        },
        "6. 评估框架": {
            "检索评估": "Precision@K, Recall@K, MRR, nDCG",
            "生成评估": "RAGAS(忠实度、相关性、完整性)",
            "端到端": "准确率、用户满意度、引用准确率",
            "A/B测试": "线上效果对比"
        }
    }

    for section, content in design.items():
        print(f"\n【{section}】")
        if isinstance(content, list):
            for item in content:
                print(f"  {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")


# =============================================
# 3. ML系统设计
# =============================================
def design_ml_system():
    """设计ML系统"""
    print_section("ML系统设计模板")

    design = {
        "1. ML系统设计四要素": {
            "数据(Data)": "数据收集、清洗、特征工程、版本管理",
            "模型(Model)": "模型选型、训练、评估、调优",
            "部署(Deploy)": "模型服务、A/B测试、灰度发布",
            "监控(Monitor)": "性能监控、数据漂移、模型衰减"
        },
        "2. ML Pipeline架构": {
            "数据Pipeline": [
                "数据收集 -> 数据验证 -> 特征工程 -> 特征存储",
                "工具: Apache Beam, Spark, Flink",
                "特征存储: Feast, Tecton"
            ],
            "训练Pipeline": [
                "实验管理 -> 超参搜索 -> 模型训练 -> 模型评估",
                "工具: MLflow, Weights & Biases, Kubeflow",
                "自动化: 定期重训, 触发式重训"
            ],
            "部署Pipeline": [
                "模型注册 -> 模型验证 -> 模型服务 -> 灰度发布",
                "工具: TensorFlow Serving, Triton, Seldon",
                "策略: 蓝绿部署, 金丝雀发布"
            ],
            "监控Pipeline": [
                "性能监控 -> 数据质量 -> 漂移检测 -> 告警",
                "工具: Prometheus, Grafana, EvidentlyAI",
                "指标: 准确率、延迟、数据分布、特征分布"
            ]
        },
        "3. 关键设计决策": {
            "在线 vs 离线": [
                "在线: 实时预测, 延迟要求高",
                "离线: 批量预测, 吞吐要求高",
                "近线: 准实时, 平衡延迟和吞吐"
            ],
            "模型更新策略": [
                "手动更新: 人工评估后手动部署",
                "定时更新: 每天每周自动重训",
                "触发更新: 性能下降时自动重训",
                "在线学习: 实时更新模型"
            ],
            "A/B测试框架": [
                "用户分流: Hash-based分流",
                "多层实验: 不同实验互不影响",
                "指标统计: 确保显著性",
                "自动化: 自动评估 -> 自动决策"
            ]
        },
        "4. 生产化最佳实践": {
            "代码质量": [
                "版本控制: 代码 + 数据 + 配置 + 模型",
                "测试: 单元测试 + 集成测试 + 模型测试",
                "文档: 模型卡片、数据卡片、Pipeline文档"
            ],
            "可复现性": [
                "记录所有超参数和随机种子",
                "使用Docker容器化环境",
                "数据和模型版本化(DVC)"
            ],
            "成本优化": [
                "模型压缩: 量化、剪枝、蒸馏",
                "推理优化: 批处理、缓存、弹性伸缩",
                "GPU利用: 多模型共享GPU"
            ]
        }
    }

    for section, content in design.items():
        print(f"\n【{section}】")
        if isinstance(content, list):
            for item in content:
                print(f"  {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W37 Day 5 - 系统设计")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    design_recommendation_system()
    design_rag_system()
    design_ml_system()

    print("\n" + "=" * 60)
    print("  系统设计面试考察的是全局思维和工程经验!")
    print("  建议: 画架构图, 从需求出发, 逐步深入细节")
    print("=" * 60)
