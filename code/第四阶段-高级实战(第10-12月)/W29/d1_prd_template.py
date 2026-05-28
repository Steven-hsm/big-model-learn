"""
W29-D1: PRD模板 - 产品需求文档生成器
=====================================
功能:
  - 生成PRD文档模板
  - 项目范围定义
  - 用户角色定义
  - 功能优先级矩阵
"""

import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum


# ============================================================
# 1. 枚举与数据类定义
# ============================================================

class Priority(Enum):
    """功能优先级枚举"""
    P0 = "P0-必须有(MVP)"
    P1 = "P1-重要(第一版)"
    P2 = "P2-期望(第二版)"
    P3 = "P3-锦上添花"


class UserRole(Enum):
    """用户角色枚举"""
    END_USER = "终端用户"
    ADMIN = "管理员"
    DEVELOPER = "开发者"
    GUEST = "访客"


class FeatureStatus(Enum):
    """功能状态枚举"""
    PLANNED = "计划中"
    IN_PROGRESS = "开发中"
    COMPLETED = "已完成"
    DEFERRED = "延期"


@dataclass
class UserStory:
    """用户故事"""
    role: str          # 作为...
    action: str        # 我想要...
    benefit: str       # 以便...
    acceptance: List[str] = field(default_factory=list)  # 验收标准

    def to_text(self) -> str:
        return f"作为{self.role}，我想要{self.action}，以便{self.benefit}"


@dataclass
class Feature:
    """功能特性"""
    name: str
    description: str
    priority: Priority
    status: FeatureStatus = FeatureStatus.PLANNED
    effort: str = "待评估"     # 工作量估算
    dependencies: List[str] = field(default_factory=list)
    user_stories: List[UserStory] = field(default_factory=list)


@dataclass
class UserPersona:
    """用户画像"""
    name: str
    role: UserRole
    description: str
    goals: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    technical_level: str = "中等"


@dataclass
class ProjectScope:
    """项目范围"""
    in_scope: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


# ============================================================
# 2. PRD文档生成器
# ============================================================

class PRDGenerator:
    """产品需求文档生成器"""

    def __init__(self, project_name: str, version: str = "1.0"):
        self.project_name = project_name
        self.version = version
        self.created_date = datetime.now().strftime("%Y-%m-%d")
        self.author = "AI工程项目组"
        self.features: List[Feature] = []
        self.personas: List[UserPersona] = []
        self.scope = ProjectScope()

    def add_feature(self, feature: Feature):
        """添加功能特性"""
        self.features.append(feature)

    def add_persona(self, persona: UserPersona):
        """添加用户画像"""
        self.personas.append(persona)

    def set_scope(self, scope: ProjectScope):
        """设置项目范围"""
        self.scope = scope

    def generate_priority_matrix(self) -> str:
        """生成功能优先级矩阵"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("功能优先级矩阵")
        lines.append("=" * 70)

        # 按优先级分组
        for priority in Priority:
            group = [f for f in self.features if f.priority == priority]
            lines.append(f"\n【{priority.value}】({len(group)}项)")
            lines.append("-" * 50)
            for i, feat in enumerate(group, 1):
                lines.append(f"  {i}. {feat.name}")
                lines.append(f"     描述: {feat.description}")
                lines.append(f"     状态: {feat.status.value} | 工作量: {feat.effort}")
                if feat.dependencies:
                    lines.append(f"     依赖: {', '.join(feat.dependencies)}")
                lines.append("")

        return "\n".join(lines)

    def generate_user_stories_section(self) -> str:
        """生成用户故事部分"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("用户故事")
        lines.append("=" * 70)

        story_count = 0
        for feat in self.features:
            for story in feat.user_stories:
                story_count += 1
                lines.append(f"\nUS-{story_count:03d}: {story.to_text()}")
                lines.append(f"  关联功能: {feat.name}")
                if story.acceptance:
                    lines.append("  验收标准:")
                    for acc in story.acceptance:
                        lines.append(f"    - [ ] {acc}")

        lines.append(f"\n共 {story_count} 个用户故事")
        return "\n".join(lines)

    def generate_scope_section(self) -> str:
        """生成项目范围部分"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("项目范围定义")
        lines.append("=" * 70)

        lines.append("\n【范围内】")
        for item in self.scope.in_scope:
            lines.append(f"  + {item}")

        lines.append("\n【范围外】")
        for item in self.scope.out_of_scope:
            lines.append(f"  - {item}")

        lines.append("\n【假设条件】")
        for item in self.scope.assumptions:
            lines.append(f"  * {item}")

        lines.append("\n【约束条件】")
        for item in self.scope.constraints:
            lines.append(f"  ! {item}")

        return "\n".join(lines)

    def generate_persona_section(self) -> str:
        """生成用户画像部分"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("用户角色定义")
        lines.append("=" * 70)

        for persona in self.personas:
            lines.append(f"\n角色: {persona.name} ({persona.role.value})")
            lines.append(f"描述: {persona.description}")
            lines.append(f"技术水平: {persona.technical_level}")
            lines.append("目标:")
            for goal in persona.goals:
                lines.append(f"  - {goal}")
            lines.append("痛点:")
            for pain in persona.pain_points:
                lines.append(f"  - {pain}")

        return "\n".join(lines)

    def generate_full_prd(self) -> str:
        """生成完整PRD文档"""
        sections = [
            self._generate_header(),
            self.generate_scope_section(),
            self.generate_persona_section(),
            self.generate_priority_matrix(),
            self.generate_user_stories_section(),
            self._generate_timeline(),
            self._generate_footer(),
        ]
        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """生成文档头部"""
        return (
            "=" * 70 + "\n"
            f"产品需求文档 (PRD)\n"
            f"项目名称: {self.project_name}\n"
            f"版本: {self.version}\n"
            f"日期: {self.created_date}\n"
            f"作者: {self.author}\n"
            + "=" * 70
        )

    def _generate_timeline(self) -> str:
        """生成时间线"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("里程碑时间线")
        lines.append("=" * 70)

        milestones = [
            ("第1周", "需求分析, 技术选型, MVP定义"),
            ("第2-3周", "核心功能开发 (检索/生成/对话)"),
            ("第4周", "前端集成, 部署, 测试"),
            ("第5-6周", "评估优化, 功能扩展, 发布准备"),
        ]

        for phase, desc in milestones:
            lines.append(f"\n  {phase}: {desc}")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """生成文档尾部"""
        return (
            "\n" + "=" * 70 + "\n"
            "文档结束\n"
            "变更记录需在版本控制中维护\n"
            + "=" * 70
        )

    def export_json(self) -> str:
        """导出为JSON格式"""
        data = {
            "project": self.project_name,
            "version": self.version,
            "date": self.created_date,
            "scope": asdict(self.scope),
            "personas": [
                {
                    "name": p.name,
                    "role": p.role.value,
                    "description": p.description,
                    "goals": p.goals,
                    "pain_points": p.pain_points,
                }
                for p in self.personas
            ],
            "features": [
                {
                    "name": f.name,
                    "priority": f.priority.value,
                    "status": f.status.value,
                    "effort": f.effort,
                }
                for f in self.features
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# 3. 演示: 创建RAG项目的PRD
# ============================================================

def demo_rag_prd():
    """演示: 为RAG项目创建PRD"""

    print("=" * 60)
    print("W29-D1: PRD模板 - 产品需求文档生成器")
    print("=" * 60)

    # 创建PRD生成器
    prd = PRDGenerator(
        project_name="智能文档问答系统 (RAG)",
        version="1.0"
    )

    # ---- 设置项目范围 ----
    prd.set_scope(ProjectScope(
        in_scope=[
            "多格式文档上传与解析 (PDF/Word/TXT/Markdown)",
            "文档自动切分与向量化存储",
            "基于语义的智能检索与问答",
            "多轮对话上下文管理",
            "流式回答输出",
            "管理后台 (文档管理/用户管理)",
        ],
        out_of_scope=[
            "实时协作编辑",
            "多语言翻译功能",
            "移动端原生应用",
            "离线部署模式",
        ],
        assumptions=[
            "用户有基本的电脑使用能力",
            "文档以中文和英文为主",
            "网络环境稳定",
            "LLM API可用且响应时间<5秒",
        ],
        constraints=[
            "项目周期: 6周",
            "团队规模: 1-2人",
            "预算: 使用开源工具+API调用",
            "响应延迟: <3秒 (首token)",
        ],
    ))

    # ---- 添加用户画像 ----
    prd.add_persona(UserPersona(
        name="知识工作者",
        role=UserRole.END_USER,
        description="需要从大量文档中快速获取信息的业务人员",
        goals=["快速找到文档中的关键信息", "获得准确的答案而非模糊的搜索结果"],
        pain_points=["传统搜索返回太多无关结果", "需要翻阅多个文档才能找到答案"],
        technical_level="初级",
    ))

    prd.add_persona(UserPersona(
        name="系统管理员",
        role=UserRole.ADMIN,
        description="负责维护知识库和管理系统的人员",
        goals=["方便地上传和管理文档", "监控系统运行状态"],
        pain_points=["缺少统一的管理工具", "难以追踪系统问题"],
        technical_level="中级",
    ))

    # ---- 添加功能特性 ----
    # P0 - MVP核心功能
    prd.add_feature(Feature(
        name="文档上传与解析",
        description="支持PDF/Word/TXT/Markdown格式文档的上传和文本提取",
        priority=Priority.P0,
        effort="3天",
        dependencies=[],
        user_stories=[
            UserStory("知识工作者", "上传PDF文档到系统", "系统可以理解文档内容并回答相关问题",
                      ["支持拖拽上传", "上传后显示解析状态", "解析失败时给出明确提示"]),
        ],
    ))

    prd.add_feature(Feature(
        name="智能问答",
        description="基于RAG的文档问答功能, 支持自然语言提问",
        priority=Priority.P0,
        effort="5天",
        dependencies=["文档上传与解析", "向量存储"],
        user_stories=[
            UserStory("知识工作者", "用自然语言提问并获得准确回答", "快速获取文档中的信息",
                      ["回答基于文档内容", "回答中标注来源", "无法回答时给出提示"]),
        ],
    ))

    prd.add_feature(Feature(
        name="向量存储与检索",
        description="将文档向量化存储并支持语义检索",
        priority=Priority.P0,
        effort="4天",
        dependencies=[],
        user_stories=[
            UserStory("开发者", "将文档切分后向量化存储", "支持语义级别的检索",
                      ["支持批量嵌入", "检索延迟<500ms", "支持相似度阈值过滤"]),
        ],
    ))

    # P1 - 重要功能
    prd.add_feature(Feature(
        name="多轮对话",
        description="支持上下文关联的多轮对话",
        priority=Priority.P1,
        effort="3天",
        dependencies=["智能问答"],
        user_stories=[
            UserStory("知识工作者", "进行追问以获得更详细的信息", "深入理解文档内容",
                      ["支持上下文关联", "对话历史可查看", "可清除对话历史"]),
        ],
    ))

    prd.add_feature(Feature(
        name="流式输出",
        description="回答以流式方式逐步展示, 提升用户体验",
        priority=Priority.P1,
        effort="2天",
        dependencies=["智能问答"],
    ))

    # P2 - 期望功能
    prd.add_feature(Feature(
        name="混合检索",
        description="结合关键词检索和语义检索, 提升检索质量",
        priority=Priority.P2,
        effort="3天",
        dependencies=["向量存储与检索"],
    ))

    prd.add_feature(Feature(
        name="权限管理",
        description="用户角色和文档访问权限控制",
        priority=Priority.P2,
        effort="4天",
        dependencies=[],
    ))

    # P3 - 锦上添花
    prd.add_feature(Feature(
        name="对话导出",
        description="将对话记录导出为文档",
        priority=Priority.P3,
        effort="1天",
        dependencies=["多轮对话"],
    ))

    # ---- 生成完整PRD ----
    print(prd.generate_full_prd())

    # ---- 导出JSON ----
    print("\n" + "=" * 60)
    print("PRD JSON格式导出 (前500字符):")
    print("=" * 60)
    json_output = prd.export_json()
    print(json_output[:500] + "...")


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    demo_rag_prd()
