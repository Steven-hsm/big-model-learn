"""
W25-D3 数据验证
===============
数据验证(Great Expectations概念), Schema验证, 分布漂移检测, 异常值自动检测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D3 数据验证")
print("=" * 60)

# ============================================================
# 1. 数据验证概念
# ============================================================
print("\n--- 1. 数据验证概念 ---")
print("""
数据验证是ML Pipeline的关键环节:

  Great Expectations 核心概念:
    - Expectation:  数据的"期望"(断言), 如"此列无空值"
    - Suite:        一组Expectation的集合
    - Checkpoint:   验证执行点, 运行Suite并生成报告
    - Data Docs:    自动生成的数据质量文档

  验证维度:
    - Schema验证:   列名、类型是否符合预期
    - 完整性:       空值率、重复率
    - 范围验证:     数值是否在合理区间
    - 分布验证:     数据分布是否发生漂移
    - 唯一性:       主键是否唯一
    - 引用完整性:   外键是否有效
""")

# ============================================================
# 2. Schema验证
# ============================================================
print("\n--- 2. Schema验证 ---")


class SchemaValidator:
    """Schema验证器"""

    def __init__(self):
        self.schema = {}  # column -> {dtype, nullable, min, max, ...}
        self.results = []

    def define_column(self, name, dtype, nullable=True, min_val=None,
                      max_val=None, allowed_values=None, regex=None):
        self.schema[name] = {
            'dtype': dtype, 'nullable': nullable,
            'min_val': min_val, 'max_val': max_val,
            'allowed_values': allowed_values, 'regex': regex,
        }

    def validate(self, df):
        self.results = []
        # 检查必需列
        missing_cols = set(self.schema.keys()) - set(df.columns)
        if missing_cols:
            self.results.append({
                'check': 'required_columns',
                'passed': False,
                'detail': f'缺少列: {missing_cols}',
            })
        else:
            self.results.append({
                'check': 'required_columns',
                'passed': True,
                'detail': f'所有 {len(self.schema)} 列均存在',
            })

        for col, rules in self.schema.items():
            if col not in df.columns:
                continue

            # 类型检查
            actual_dtype = str(df[col].dtype)
            expected_dtype = rules['dtype']
            type_ok = expected_dtype in actual_dtype or actual_dtype.startswith(expected_dtype)
            self.results.append({
                'check': f'{col}.dtype',
                'passed': type_ok,
                'detail': f'期望={expected_dtype}, 实际={actual_dtype}',
            })

            # 空值检查
            if not rules['nullable']:
                null_count = df[col].isna().sum()
                self.results.append({
                    'check': f'{col}.not_null',
                    'passed': null_count == 0,
                    'detail': f'空值数={null_count}',
                })

            # 范围检查
            if rules['min_val'] is not None:
                violations = (df[col] < rules['min_val']).sum()
                self.results.append({
                    'check': f'{col}.min({rules["min_val"]})',
                    'passed': violations == 0,
                    'detail': f'违反数={violations}',
                })

            if rules['max_val'] is not None:
                violations = (df[col] > rules['max_val']).sum()
                self.results.append({
                    'check': f'{col}.max({rules["max_val"]})',
                    'passed': violations == 0,
                    'detail': f'违反数={violations}',
                })

            # 允许值检查
            if rules['allowed_values'] is not None:
                invalid = ~df[col].isin(rules['allowed_values'])
                self.results.append({
                    'check': f'{col}.allowed_values',
                    'passed': invalid.sum() == 0,
                    'detail': f'非法值数={invalid.sum()}',
                })

        return self.results

    def report(self):
        print(f"\n  {'检查项':<30} {'状态':<6} {'详情'}")
        print("  " + "-" * 70)
        for r in self.results:
            status = 'PASS' if r['passed'] else 'FAIL'
            print(f"  {r['check']:<30} {status:<6} {r['detail']}")
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)
        print(f"\n  总计: {passed}/{total} 通过 ({passed/total:.0%})")


# 创建测试数据
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'user_id': [f'U{i:04d}' for i in range(n)],
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(50000, 15000, n),
    'gender': np.random.choice(['M', 'F'], n),
    'score': np.random.uniform(0, 100, n),
})

# 注入一些违规数据
df.loc[0, 'age'] = 150       # 超范围
df.loc[1, 'gender'] = 'X'    # 非法值
df.loc[2, 'income'] = np.nan  # 空值

# 定义Schema并验证
validator = SchemaValidator()
validator.define_column('user_id', 'object', nullable=False)
validator.define_column('age', 'int', min_val=0, max_val=120)
validator.define_column('income', 'float', min_val=0)
validator.define_column('gender', 'object', allowed_values=['M', 'F'])
validator.define_column('score', 'float', min_val=0, max_val=100)

validator.validate(df)
validator.report()

# ============================================================
# 3. 分布漂移检测
# ============================================================
print("\n\n--- 3. 分布漂移检测 ---")


class DistributionDriftDetector:
    """分布漂移检测器"""

    @staticmethod
    def ks_test(reference, current, alpha=0.05):
        """KS检验 - 检测分布是否相同"""
        stat, p_value = scipy_stats.ks_2samp(reference, current)
        drift_detected = p_value < alpha
        return {
            'method': 'KS检验',
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'interpretation': '分布不同' if drift_detected else '分布一致',
        }

    @staticmethod
    def psi(reference, current, bins=10, threshold=0.2):
        """Population Stability Index (PSI)"""
        # 统一bin边界
        all_data = np.concatenate([reference, current])
        bin_edges = np.percentile(all_data, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_hist = np.histogram(reference, bins=bin_edges)[0] / len(reference)
        cur_hist = np.histogram(current, bins=bin_edges)[0] / len(current)

        # 避免除零
        ref_hist = np.clip(ref_hist, 1e-6, None)
        cur_hist = np.clip(cur_hist, 1e-6, None)

        psi_value = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

        if psi_value < 0.1:
            level = '无显著变化'
        elif psi_value < 0.2:
            level = '轻微漂移'
        else:
            level = '显著漂移'

        return {
            'method': 'PSI',
            'psi_value': psi_value,
            'threshold': threshold,
            'drift_detected': psi_value > threshold,
            'interpretation': level,
        }

    @staticmethod
    def chi_square_test(reference, current, categories):
        """卡方检验 - 用于分类变量"""
        ref_counts = pd.Series(reference).value_counts()
        cur_counts = pd.Series(current).value_counts()

        ref_freq = np.array([ref_counts.get(c, 0) for c in categories])
        cur_freq = np.array([cur_counts.get(c, 0) for c in categories])

        # 标准化
        ref_freq = ref_freq / ref_freq.sum()
        cur_freq = cur_freq / cur_freq.sum()

        # 卡方检验
        chi2, p_value = scipy_stats.chisquare(cur_freq * len(current), f_exp=ref_freq * len(current))

        return {
            'method': '卡方检验',
            'statistic': chi2,
            'p_value': p_value,
            'drift_detected': p_value < 0.05,
            'interpretation': '分布变化' if p_value < 0.05 else '分布稳定',
        }


# 生成参考数据和当前数据
np.random.seed(42)
reference_income = np.random.normal(50000, 15000, 1000)
current_income = np.random.normal(55000, 18000, 1000)  # 均值和方差都漂移了

reference_age = np.random.randint(18, 65, 1000)
current_age = np.random.randint(25, 75, 1000)  # 年龄分布变化

detector = DistributionDriftDetector()

# KS检验
ks_result = detector.ks_test(reference_income, current_income)
print(f"\n  收入KS检验: statistic={ks_result['statistic']:.4f}, p={ks_result['p_value']:.4f}")
print(f"  结论: {ks_result['interpretation']}")

# PSI检验
psi_result = detector.psi(reference_income, current_income)
print(f"\n  收入PSI: value={psi_result['psi_value']:.4f}")
print(f"  结论: {psi_result['interpretation']}")

# KS检验 - 年龄
ks_age = detector.ks_test(reference_age.astype(float), current_age.astype(float))
print(f"\n  年龄KS检验: statistic={ks_age['statistic']:.4f}, p={ks_age['p_value']:.4f}")
print(f"  结论: {ks_age['interpretation']}")

# ============================================================
# 4. 异常值自动检测
# ============================================================
print("\n--- 4. 异常值自动检测 ---")


class AnomalyDetector:
    """异常值检测器"""

    @staticmethod
    def iqr_method(series, factor=1.5):
        """IQR方法检测异常值"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (series < lower) | (series > upper)
        return {
            'method': 'IQR',
            'anomaly_count': mask.sum(),
            'anomaly_ratio': mask.mean(),
            'lower_bound': lower,
            'upper_bound': upper,
            'anomalies': series[mask],
        }

    @staticmethod
    def zscore_method(series, threshold=3):
        """Z-score方法"""
        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std
        mask = z_scores.abs() > threshold
        return {
            'method': 'Z-Score',
            'anomaly_count': mask.sum(),
            'anomaly_ratio': mask.mean(),
            'threshold': threshold,
            'anomalies': series[mask],
        }

    @staticmethod
    def isolation_forest_concept(series, contamination=0.05):
        """Isolation Forest概念实现(简化版)"""
        n = len(series)
        n_anomalies = int(n * contamination)
        # 用距离均值的标准差倍数模拟
        mean = series.mean()
        std = series.std()
        distances = ((series - mean) / std).abs()
        threshold = distances.nlargest(n_anomalies).min()
        mask = distances > threshold
        return {
            'method': 'Isolation(简化)',
            'anomaly_count': mask.sum(),
            'anomaly_ratio': mask.mean(),
            'threshold_sigma': threshold,
            'anomalies': series[mask],
        }


# 生成含异常值的数据
np.random.seed(42)
data_with_outliers = pd.Series(np.concatenate([
    np.random.normal(100, 10, 490),
    np.random.normal(200, 5, 10),  # 异常值
]))

detector = AnomalyDetector()

# IQR方法
iqr_result = detector.iqr_method(data_with_outliers)
print(f"\n  IQR方法: 检测到 {iqr_result['anomaly_count']} 个异常值 "
      f"({iqr_result['anomaly_ratio']:.2%})")
print(f"  正常范围: [{iqr_result['lower_bound']:.1f}, {iqr_result['upper_bound']:.1f}]")

# Z-score方法
zscore_result = detector.zscore_method(data_with_outliers)
print(f"\n  Z-Score方法: 检测到 {zscore_result['anomaly_count']} 个异常值 "
      f"({zscore_result['anomaly_ratio']:.2%})")

# Isolation Forest概念
iso_result = detector.isolation_forest_concept(data_with_outliers)
print(f"\n  Isolation(简化): 检测到 {iso_result['anomaly_count']} 个异常值 "
      f"({iso_result['anomaly_ratio']:.2%})")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: Schema验证结果
ax = axes[0, 0]
passed = sum(1 for r in validator.results if r['passed'])
failed = len(validator.results) - passed
colors = ['#4CAF50', '#F44336']
ax.bar(['通过', '失败'], [passed, failed], color=colors)
ax.set_title(f'Schema验证结果 ({passed}/{len(validator.results)} 通过)')
ax.set_ylabel('检查项数')
for i, v in enumerate([passed, failed]):
    ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# 右上: 分布漂移对比
ax = axes[0, 1]
ax.hist(reference_income, bins=30, alpha=0.5, density=True, label='参考分布', color='#2196F3')
ax.hist(current_income, bins=30, alpha=0.5, density=True, label='当前分布', color='#F44336')
ax.axvline(reference_income.mean(), color='#1565C0', linestyle='--', label=f'参考均值={reference_income.mean():.0f}')
ax.axvline(current_income.mean(), color='#C62828', linestyle='--', label=f'当前均值={current_income.mean():.0f}')
ax.set_title(f'收入分布漂移 (PSI={psi_result["psi_value"]:.3f})')
ax.set_xlabel('收入')
ax.set_ylabel('密度')
ax.legend(fontsize=8)

# 左下: 异常值检测
ax = axes[1, 0]
normal_mask = ~data_with_outliers.index.isin(iqr_result['anomalies'].index)
ax.scatter(data_with_outliers[normal_mask].index, data_with_outliers[normal_mask],
           c='#2196F3', s=10, alpha=0.5, label='正常')
ax.scatter(iqr_result['anomalies'].index, iqr_result['anomalies'],
           c='#F44336', s=30, label='异常', zorder=5)
ax.axhline(iqr_result['upper_bound'], color='orange', linestyle='--', label='IQR上界')
ax.axhline(iqr_result['lower_bound'], color='orange', linestyle='--', label='IQR下界')
ax.set_title(f'异常值检测 (IQR方法, {iqr_result["anomaly_count"]}个异常)')
ax.set_xlabel('样本索引')
ax.set_ylabel('值')
ax.legend(fontsize=8)

# 右下: PSI评分卡
ax = axes[1, 1]
features = ['income', 'age', 'score']
psi_values = []
for feat, (ref, cur) in [('income', (reference_income, current_income)),
                           ('age', (reference_age.astype(float), current_age.astype(float))),
                           ('score', (np.random.normal(50, 15, 1000), np.random.normal(50, 15, 1000)))]:
    psi_values.append(detector_psi := DistributionDriftDetector.psi(
        DistributionDriftDetector(), ref, cur
    )['psi_value'])

colors_psi = ['#4CAF50' if v < 0.1 else '#FF9800' if v < 0.2 else '#F44336' for v in psi_values]
bars = ax.bar(features, psi_values, color=colors_psi)
ax.axhline(0.1, color='orange', linestyle='--', alpha=0.7, label='轻微漂移阈值')
ax.axhline(0.2, color='red', linestyle='--', alpha=0.7, label='显著漂移阈值')
ax.set_title('特征漂移PSI评分')
ax.set_ylabel('PSI值')
ax.legend(fontsize=8)
for bar, val in zip(bars, psi_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d3_data_validation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d3_data_validation.png")

print("\n完成! 数据验证要点:")
print("  1. Schema验证: 确保数据结构符合预期(列名、类型、约束)")
print("  2. 分布漂移: KS检验(连续)、PSI(通用)、卡方(分类)")
print("  3. 异常检测: IQR(简单)、Z-Score(正态)、Isolation Forest(复杂)")
print("  4. 自动化: 将验证嵌入Pipeline, 确保数据质量门控")
