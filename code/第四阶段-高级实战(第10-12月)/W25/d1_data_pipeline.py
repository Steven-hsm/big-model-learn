"""
W25-D1 数据Pipeline设计
=======================
ETL流程, 数据版本管理(DVC概念), 数据质量检查, 自动化Pipeline
"""

import os
import json
import hashlib
import sqlite3
import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D1 数据Pipeline设计")
print("=" * 60)

# ============================================================
# 1. ETL流程概述
# ============================================================
print("\n--- 1. ETL流程概述 ---")
print("""
ETL (Extract-Transform-Load) 是数据Pipeline的核心模式:

  Extract  - 从多个数据源抽取数据
  Transform - 清洗、转换、聚合、特征计算
  Load     - 加载到目标存储(数据库/数据仓库/特征库)

常见工具:
  - Apache Airflow: 工作流编排
  - Apache Spark:   大规模数据处理
  - DVC:           数据版本管理
  - Great Expectations: 数据质量验证
""")

# ============================================================
# 2. 简化版ETL Pipeline实现
# ============================================================
print("\n--- 2. 简化版ETL Pipeline实现 ---")


class ETLPipeline:
    """简化版ETL Pipeline框架"""

    def __init__(self, name):
        self.name = name
        self.steps = []
        self.log = []
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
        }

    def add_step(self, name, func):
        """添加处理步骤"""
        self.steps.append({'name': name, 'func': func})
        return self

    def _log_step(self, step_name, status, detail=''):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': status,
            'detail': detail
        }
        self.log.append(entry)
        print(f"  [{status}] {step_name}: {detail}")

    def run(self, data):
        """执行Pipeline"""
        self.stats['start_time'] = datetime.now()
        self.stats['input_count'] = len(data)
        self._log_step('PIPELINE_START', 'INFO', f'Pipeline [{self.name}] 启动, 输入 {len(data)} 条记录')

        current_data = data.copy()

        for i, step in enumerate(self.steps):
            step_name = step['name']
            try:
                t0 = time.time()
                current_data = step['func'](current_data)
                elapsed = time.time() - t0
                self._log_step(
                    step_name, 'OK',
                    f'输出 {len(current_data)} 条记录, 耗时 {elapsed:.3f}s'
                )
            except Exception as e:
                self.stats['errors'] += 1
                self._log_step(step_name, 'ERROR', str(e))
                raise

        self.stats['output_count'] = len(current_data)
        self.stats['end_time'] = datetime.now()
        self._log_step('PIPELINE_END', 'INFO', f'完成, 输出 {len(current_data)} 条记录')
        return current_data

    def get_stats(self):
        return self.stats


# --- 生成模拟数据 ---
np.random.seed(42)
n = 1000
raw_data = pd.DataFrame({
    'user_id': [f'U{i:04d}' for i in range(n)],
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(50000, 15000, n),
    'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n),
    'signup_date': [datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(n)],
    'purchase_amount': np.random.exponential(500, n),
    'is_vip': np.random.choice([0, 1], n, p=[0.85, 0.15]),
})

# 注入一些脏数据
raw_data.loc[10, 'age'] = 200       # 异常年龄
raw_data.loc[20, 'income'] = -1000  # 负收入
raw_data.loc[30, 'age'] = np.nan    # 缺失值
raw_data.loc[40, 'city'] = None     # 缺失城市

print(f"\n原始数据: {raw_data.shape}")
print(raw_data.head(3))

# --- 定义ETL步骤 ---


def extract_validate(df):
    """Step: 数据抽取与基本验证"""
    assert 'user_id' in df.columns, "缺少 user_id 列"
    assert len(df) > 0, "数据为空"
    return df


def transform_clean(df):
    """Step: 数据清洗"""
    # 修复异常年龄
    df = df.copy()
    df.loc[df['age'] > 120, 'age'] = np.nan
    df.loc[df['income'] < 0, 'income'] = np.nan

    # 填充缺失值
    df['age'] = df['age'].fillna(df['age'].median())
    df['income'] = df['income'].fillna(df['income'].median())
    df['city'] = df['city'].fillna('未知')

    # 去重
    df = df.drop_duplicates(subset=['user_id'])
    return df


def transform_features(df):
    """Step: 特征计算"""
    df = df.copy()
    df['income_per_age'] = df['income'] / df['age']
    df['is_high_value'] = (df['purchase_amount'] > 1000).astype(int)
    df['city_tier'] = df['city'].map({
        '北京': 1, '上海': 1, '广州': 2, '深圳': 2, '杭州': 2, '未知': 3
    })
    df['days_since_signup'] = (pd.Timestamp.now() - pd.to_datetime(df['signup_date'])).dt.days
    return df


def transform_normalize(df):
    """Step: 数据标准化"""
    df = df.copy()
    for col in ['age', 'income', 'purchase_amount']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_scaled'] = (df[col] - mean) / std
    return df


def load_validate(df):
    """Step: 输出验证"""
    assert df['user_id'].nunique() == len(df), "存在重复 user_id"
    assert df['age'].between(18, 120).all(), "年龄范围异常"
    return df


# --- 构建并运行Pipeline ---
pipeline = ETLPipeline('user_feature_pipeline')
pipeline.add_step('数据抽取验证', extract_validate)
pipeline.add_step('数据清洗', transform_clean)
pipeline.add_step('特征计算', transform_features)
pipeline.add_step('标准化', transform_normalize)
pipeline.add_step('输出验证', load_validate)

result = pipeline.run(raw_data)
print(f"\n处理结果: {result.shape}")
print(result[['user_id', 'age', 'income', 'city_tier', 'is_high_value']].head())

# ============================================================
# 3. 数据版本管理 (DVC概念)
# ============================================================
print("\n--- 3. 数据版本管理 (DVC概念) ---")


class DataVersionManager:
    """简化版数据版本管理器 (模拟DVC概念)"""

    def __init__(self, repo_path='./data_repo'):
        self.repo_path = repo_path
        self.versions = {}  # version -> metadata
        self._load_registry()

    def _compute_hash(self, data):
        """计算数据指纹(MD5)"""
        data_str = data.to_csv(index=False)
        return hashlib.md5(data_str.encode()).hexdigest()[:12]

    def _load_registry(self):
        """加载版本注册表"""
        registry_path = os.path.join(self.repo_path, 'version_registry.json')
        if os.path.exists(registry_path):
            with open(registry_path, 'r', encoding='utf-8') as f:
                self.versions = json.load(f)

    def _save_registry(self):
        """保存版本注册表"""
        os.makedirs(self.repo_path, exist_ok=True)
        registry_path = os.path.join(self.repo_path, 'version_registry.json')
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)

    def commit(self, data, message='', tags=None):
        """提交数据版本"""
        data_hash = self._compute_hash(data)
        version = f"v{len(self.versions) + 1}"
        metadata = {
            'version': version,
            'hash': data_hash,
            'rows': len(data),
            'columns': list(data.columns),
            'message': message,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat(),
        }

        # 保存数据快照
        snapshot_path = os.path.join(
            self.repo_path, f'snapshot_{data_hash}.csv'
        )
        data.to_csv(snapshot_path, index=False)

        self.versions[version] = metadata
        self._save_registry()
        print(f"  数据版本 {version} 已提交 (hash={data_hash}, {len(data)} 行)")
        return version

    def log(self):
        """查看版本历史"""
        print("\n  数据版本历史:")
        print(f"  {'版本':<8} {'哈希':<15} {'行数':<8} {'信息'}")
        print("  " + "-" * 60)
        for ver, meta in self.versions.items():
            print(f"  {ver:<8} {meta['hash']:<15} {meta['rows']:<8} {meta['message']}")

    def diff(self, v1, v2):
        """比较两个版本的差异"""
        if v1 not in self.versions or v2 not in self.versions:
            print("  版本不存在")
            return
        m1, m2 = self.versions[v1], self.versions[v2]
        print(f"\n  差异: {v1} -> {v2}")
        print(f"  行数变化: {m1['rows']} -> {m2['rows']} ({m2['rows'] - m1['rows']:+d})")
        cols_added = set(m2['columns']) - set(m1['columns'])
        cols_removed = set(m1['columns']) - set(m2['columns'])
        if cols_added:
            print(f"  新增列: {cols_added}")
        if cols_removed:
            print(f"  删除列: {cols_removed}")


# 使用版本管理器
dvm = DataVersionManager('./data_repo')
dvm.commit(raw_data, '原始数据', ['raw', 'initial'])
dvm.commit(result, '清洗和特征工程后', ['processed', 'v1'])
dvm.log()
dvm.diff('v1', 'v2')

# ============================================================
# 4. 数据质量检查
# ============================================================
print("\n--- 4. 数据质量检查 ---")


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self):
        self.rules = []
        self.results = []

    def add_rule(self, name, check_func, severity='warning'):
        self.rules.append({
            'name': name,
            'func': check_func,
            'severity': severity,
        })

    def run_checks(self, df):
        self.results = []
        for rule in self.rules:
            try:
                passed, detail = rule['func'](df)
            except Exception as e:
                passed, detail = False, str(e)

            self.results.append({
                'rule': rule['name'],
                'severity': rule['severity'],
                'passed': passed,
                'detail': detail,
            })
        return self.results

    def report(self):
        print(f"\n  {'规则':<25} {'级别':<10} {'状态':<8} {'详情'}")
        print("  " + "-" * 70)
        for r in self.results:
            status = 'PASS' if r['passed'] else 'FAIL'
            print(f"  {r['rule']:<25} {r['severity']:<10} {status:<8} {r['detail']}")


checker = DataQualityChecker()

# 添加质量规则
checker.add_rule('无空值(user_id)', lambda df: (df['user_id'].notna().all(), f'空值数: {df["user_id"].isna().sum()}'))
checker.add_rule('年龄范围合理', lambda df: (df['age'].between(18, 120).all(), f'范围: {df["age"].min():.0f}-{df["age"].max():.0f}'))
checker.add_rule('收入非负', lambda df: (df['income'].ge(0).all(), f'最小值: {df["income"].min():.0f}'))
checker.add_rule('无重复ID', lambda df: (df['user_id'].nunique() == len(df), f'重复数: {len(df) - df["user_id"].nunique()}'))
checker.add_rule('VIP比例合理', lambda df: (0 < df['is_vip'].mean() < 0.5, f'VIP比例: {df["is_vip"].mean():.2%}'), 'critical')

print("\n对清洗后数据运行质量检查:")
checker.run_checks(result)
checker.report()

# ============================================================
# 5. 自动化Pipeline调度概念
# ============================================================
print("\n--- 5. 自动化Pipeline调度概念 ---")
print("""
Pipeline调度策略:

  1. 定时调度 (Schedule)
     - Cron表达式: "0 2 * * *" (每天凌晨2点)
     - 适合定期批量处理

  2. 事件驱动 (Event-driven)
     - 数据到达时触发
     - 上游Pipeline完成时触发

  3. 条件触发 (Condition-based)
     - 数据量达到阈值
     - 数据漂移检测到时

  调度器选型:
  ┌─────────────────┬──────────────────────────────┐
  │ 工具            │ 特点                         │
  ├─────────────────┼──────────────────────────────┤
  │ Apache Airflow  │ Python生态, DAG定义, 社区大  │
  │ Prefect         │ 轻量级, Pythonic API         │
  │ Dagster         │ 数据感知, 类型安全           │
  │ Kubeflow Pipes  │ K8s原生, ML专用              │
  │ Luigi           │ Spotify出品, 批处理专用      │
  └─────────────────┴──────────────────────────────┘
""")

# --- Pipeline执行可视化 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: Pipeline各步骤耗时
steps = [s['name'] for s in pipeline.log if s['status'] == 'OK']
# 模拟耗时数据
times = [0.02, 0.15, 0.08, 0.05, 0.01][:len(steps)]

axes[0].barh(range(len(steps)), times, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'][:len(steps)])
axes[0].set_yticks(range(len(steps)))
axes[0].set_yticklabels(steps)
axes[0].set_xlabel('耗时 (秒)')
axes[0].set_title('Pipeline 各步骤耗时')
for i, t in enumerate(times):
    axes[0].text(t + 0.002, i, f'{t:.3f}s', va='center')

# 右图: 数据质量检查结果
pass_count = sum(1 for r in checker.results if r['passed'])
fail_count = len(checker.results) - pass_count
axes[1].pie(
    [pass_count, fail_count],
    labels=[f'通过 ({pass_count})', f'失败 ({fail_count})'],
    colors=['#4CAF50', '#F44336'],
    autopct='%1.0f%%',
    startangle=90,
)
axes[1].set_title('数据质量检查结果')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d1_pipeline_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d1_pipeline_overview.png")

# 清理临时文件
import shutil
if os.path.exists('./data_repo'):
    shutil.rmtree('./data_repo')

print("\n完成! 数据Pipeline设计要点:")
print("  1. ETL流程: Extract -> Transform -> Load, 每步独立可测试")
print("  2. 数据版本: 像Git一样管理数据版本, 可追溯可回滚")
print("  3. 质量检查: 自动化验证规则, 拦截脏数据")
print("  4. 自动调度: 定时/事件驱动, 保证Pipeline持续运行")
