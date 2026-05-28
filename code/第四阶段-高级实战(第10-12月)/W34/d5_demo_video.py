"""
W34-D5 Demo脚本
================
生成演示视频的脚本和话术, 包括:
- 演示流程设计
- 话术模板
- 时间安排

好的演示能让观众快速理解项目价值。
"""

import time
from typing import List, Dict
from dataclasses import dataclass, field


# ============================================================
# 1. 演示流程设计
# ============================================================

@dataclass
class DemoStep:
    """演示步骤"""
    order: int
    title: str
    duration_seconds: int
    action: str
    narration: str
    key_points: List[str] = field(default_factory=list)
    screen_state: str = ""  # 屏幕上应该显示什么


class DemoScriptBuilder:
    """演示脚本构建器"""

    def __init__(self, total_minutes: int = 10):
        self.total_minutes = total_minutes
        self.steps: List[DemoStep] = []

    def add_step(self, title: str, duration_seconds: int, action: str,
                  narration: str, key_points: List[str] = None,
                  screen_state: str = ""):
        order = len(self.steps) + 1
        self.steps.append(DemoStep(
            order=order,
            title=title,
            duration_seconds=duration_seconds,
            action=action,
            narration=narration,
            key_points=key_points or [],
            screen_state=screen_state,
        ))

    def build_rag_demo(self):
        """构建标准RAG演示流程"""
        self.steps = []

        # 开场
        self.add_step(
            "开场介绍", 60,
            "展示项目标题页",
            "大家好, 今天我要演示的是我开发的RAG知识库问答系统。"
            "这个系统基于检索增强生成技术, 能够基于企业知识库准确回答用户问题。",
            ["项目背景", "核心价值"],
            "项目Logo和标题"
        )

        # 功能概览
        self.add_step(
            "功能概览", 60,
            "展示系统架构图",
            "系统包含四大核心模块: 智能检索、文档管理、AI问答和系统管理。"
            "让我逐一演示这些功能。",
            ["四大模块", "架构设计"],
            "系统架构图"
        )

        # 文档上传
        self.add_step(
            "文档上传", 90,
            "上传3篇示例文档",
            "首先, 让我上传几篇知识文档。系统支持txt、pdf和md格式。"
            "上传后, 系统会自动分块、索引并生成向量嵌入。",
            ["支持多格式", "自动分块索引"],
            "Gradio界面的文档上传区域"
        )

        # 基础查询
        self.add_step(
            "智能问答", 120,
            "输入: 什么是RAG技术?",
            "现在让我提一个问题: 什么是RAG技术?"
            "可以看到系统很快就给出了回答, 同时标注了信息来源。"
            "回答质量基于检索到的相关文档, 确保了准确性。",
            ["快速响应", "来源可追溯"],
            "聊天界面显示问答过程"
        )

        # 复杂查询
        self.add_step(
            "复杂问题", 90,
            "输入: 如何优化检索质量?",
            "接下来测试一个更复杂的问题。"
            "系统能够综合多篇文档的信息, 给出全面的回答。",
            ["多文档综合", "回答全面"],
            "聊天界面显示复杂查询结果"
        )

        # 参数调节
        self.add_step(
            "参数调节", 60,
            "调整Top-K从3到10",
            "系统支持调节检索参数。可以看到, 增大Top-K后, "
            "检索到的文档更多, 回答也更全面。",
            ["灵活配置", "参数透明"],
            "参数调节面板"
        )

        # 性能展示
        self.add_step(
            "性能数据", 60,
            "展示监控仪表盘",
            "在性能方面, 系统查询平均延迟250ms, P95延迟520ms。"
            "通过缓存优化, 缓存命中率达到45%。",
            ["低延迟", "高可用"],
            "性能监控仪表盘"
        )

        # 总结
        self.add_step(
            "总结", 60,
            "展示总结页",
            "总结一下, 这个RAG系统实现了: "
            "混合检索引擎, Recall@5达到82%; "
            "流式输出, 用户体验流畅; "
            "企业级安全和多租户支持; "
            "完整的评估和监控体系。"
            "谢谢大家!",
            ["核心成果", "技术亮点"],
            "总结页面"
        )

    def print_script(self):
        """打印演示脚本"""
        total = sum(s.duration_seconds for s in self.steps)
        print(f"\n{'='*60}")
        print(f"演示脚本 (预计{total//60}分{total%60}秒)")
        print(f"{'='*60}")

        for step in self.steps:
            print(f"\n--- 步骤{step.order}: {step.title} ({step.duration_seconds}秒) ---")
            print(f"[画面] {step.screen_state}")
            print(f"[操作] {step.action}")
            print(f"[话术] {step.narration}")
            if step.key_points:
                print(f"[要点] {', '.join(step.key_points)}")

    def generate_checklist(self) -> str:
        """生成演示前检查清单"""
        return """
演示前检查清单:
========================================

技术准备:
  [ ] 系统正常运行, 无报错
  [ ] 示例数据已预加载
  [ ] 网络连接稳定
  [ ] 屏幕分辨率设置好
  [ ] 备用方案准备(离线Demo)

内容准备:
  [ ] 演示脚本熟悉
  [ ] 关键数据记住
  [ ] 可能的问题准备答案
  [ ] 时间控制练习

环境准备:
  [ ] 关闭通知
  [ ] 清理桌面
  [ ] 准备录屏(如需要)
  [ ] 测试麦克风(如需要)
"""


# ============================================================
# 2. 话术模板
# ============================================================

class NarrationTemplates:
    """话术模板"""

    @staticmethod
    def opening() -> str:
        return (
            "大家好, 今天我要演示的是我开发的{project_name}。"
            "\n\n这个项目解决的核心问题是: {problem}。"
            "\n\n我将从以下几个方面进行演示: {outline}。"
        )

    @staticmethod
    def feature_demo() -> str:
        return (
            "接下来演示{feature_name}功能。"
            "\n\n[操作] {action_description}"
            "\n\n大家可以看到{observation}, 这说明{implication}。"
        )

    @staticmethod
    def data_driven() -> str:
        return (
            "这里有一些关键数据:"
            "\n\n- {metric_1}: 从{before_1}提升到{after_1}, 提升了{improvement_1}"
            "\n- {metric_2}: {value_2}"
            "\n- {metric_3}: {value_3}"
            "\n\n这些数据是通过{method}得出的。"
        )

    @staticmethod
    def closing() -> str:
        return (
            "最后做一个总结。"
            "\n\n这个项目的核心亮点是{highlights}。"
            "\n\n未来计划{future_plans}。"
            "\n\n谢谢大家! 欢迎提问。"
        )


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W34-D5 Demo脚本")
    print("=" * 60)

    # --- 构建演示脚本 ---
    builder = DemoScriptBuilder(total_minutes=10)
    builder.build_rag_demo()
    builder.print_script()

    # --- 检查清单 ---
    print(f"\n{'='*60}")
    print("--- 演示前检查清单 ---")
    print(builder.generate_checklist())

    # --- 话术模板 ---
    print(f"{'='*60}")
    print("--- 话术模板示例 ---")
    print(f"{'='*60}")

    templates = NarrationTemplates()

    print("\n[开场白]")
    print(templates.opening().format(
        project_name="RAG知识库问答系统",
        problem="企业知识库信息量大但查找困难, 传统搜索无法理解语义",
        outline="文档上传、智能问答、性能展示"
    ))

    print("\n[功能演示]")
    print(templates.feature_demo().format(
        feature_name="智能问答",
        action_description="输入'什么是RAG技术?'",
        observation="系统快速给出了基于知识库的准确回答",
        implication="检索增强生成技术有效提升了回答质量"
    ))

    print("\n[数据驱动]")
    print(templates.data_driven().format(
        metric_1="检索Recall@5", before_1="45%", after_1="82%", improvement_1="82%",
        metric_2="查询延迟(P95)", value_2="520ms",
        metric_3="用户满意度", value_3="78%",
        method="消融实验和A/B测试"
    ))

    print("\n[结尾]")
    print(templates.closing().format(
        highlights="混合检索引擎、流式输出、企业级安全、完整评估体系",
        future_plans="支持多模态、引入Agent能力、构建知识图谱"
    ))

    print("\n完成!")
