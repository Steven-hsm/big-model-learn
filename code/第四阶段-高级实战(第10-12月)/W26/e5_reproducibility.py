"""
W26-E5 实验可复现性
====================
实验可复现性, 随机种子管理, 环境锁定, 配置管理(YAML/JSON), 代码版本与实验关联
"""

import os
import json
import sys
import platform
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-E5 实验可复现性")
print("=" * 60)

# ============================================================
# 1. 可复现性概念
# ============================================================
print("\n--- 1. 可复现性概念 ---")
print("""
ML实验可复现性 = 相同代码 + 相同数据 + 相同环境 => 相同结果

  可复现性层级:
    Level 1 - 代码版本: Git管理, 固定commit
    Level 2 - 数据版本: DVC/哈希, 固定数据快照
    Level 3 - 环境锁定: requirements.txt, Docker
    Level 4 - 随机种子: numpy, random, torch.seed
    Level 5 - 配置管理: YAML/JSON参数文件

  不可复现的常见原因:
    - 随机种子未固定
    - 包版本不一致
    - 数据版本不同
    - 浮点数精度差异(GPU vs CPU)
    - 并行计算的不确定性
""")

# ============================================================
# 2. 随机种子管理
# ============================================================
print("\n--- 2. 随机种子管理 ---")


class SeedManager:
    """全局随机种子管理器"""

    def __init__(self, seed=42):
        self.seed = seed
        self.apply()

    def apply(self):
        """应用种子到所有随机源"""
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 尝试设置PyTorch种子
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            print(f"  PyTorch 种子已设置: {self.seed}")
        except ImportError:
            pass

        print(f"  全局种子已设置: Python random={self.seed}, NumPy={self.seed}")

    @staticmethod
    def generate_worker_seed(base_seed, worker_id):
        """为多进程生成独立种子"""
        return base_seed + worker_id * 1000


# 演示种子对结果的影响
print("\n  --- 种子对结果的影响 ---")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results_no_seed = []
results_with_seed = []

for i in range(10):
    # 无固定种子
    rf_no = RandomForestClassifier(n_estimators=50)
    rf_no.fit(X_train, y_train)
    acc_no = accuracy_score(y_test, rf_no.predict(X_test))
    results_no_seed.append(acc_no)

    # 固定种子
    rf_yes = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_yes.fit(X_train, y_train)
    acc_yes = accuracy_score(y_test, rf_yes.predict(X_test))
    results_with_seed.append(acc_yes)

print(f"\n  无固定种子: mean={np.mean(results_no_seed):.4f}, "
      f"std={np.std(results_no_seed):.4f}, "
      f"range=[{np.min(results_no_seed):.4f}, {np.max(results_no_seed):.4f}]")
print(f"  固定种子42: mean={np.mean(results_with_seed):.4f}, "
      f"std={np.std(results_with_seed):.4f} (全部相同={len(set(results_with_seed)) == 1})")

# ============================================================
# 3. 环境锁定
# ============================================================
print("\n--- 3. 环境锁定 ---")


class EnvironmentSnapshot:
    """环境快照"""

    @staticmethod
    def capture():
        """捕获当前环境信息"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'python': {
                'version': sys.version,
                'executable': sys.executable,
            },
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
            },
            'packages': {},
        }

        # 捕获关键包版本
        key_packages = [
            'numpy', 'pandas', 'scikit-learn', 'matplotlib',
            'torch', 'tensorflow', 'xgboost', 'lightgbm',
        ]

        for pkg in key_packages:
            try:
                mod = __import__(pkg.replace('-', '_'))
                snapshot['packages'][pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                snapshot['packages'][pkg] = 'not_installed'

        return snapshot

    @staticmethod
    def generate_requirements(snapshot=None):
        """生成requirements.txt内容"""
        if snapshot is None:
            snapshot = EnvironmentSnapshot.capture()

        lines = [f"# Generated at {snapshot['timestamp']}",
                 f"# Python {snapshot['python']['version'].split()[0]}", ""]

        for pkg, version in sorted(snapshot['packages'].items()):
            if version not in ('not_installed', 'unknown'):
                lines.append(f"{pkg}>={version}")
            elif version == 'not_installed':
                lines.append(f"# {pkg} not installed")

        return '\n'.join(lines)

    @staticmethod
    def compare(snapshot1, snapshot2):
        """比较两个环境快照"""
        diffs = []
        all_pkgs = set(snapshot1['packages'].keys()) | set(snapshot2['packages'].keys())
        for pkg in sorted(all_pkgs):
            v1 = snapshot1['packages'].get(pkg, 'missing')
            v2 = snapshot2['packages'].get(pkg, 'missing')
            if v1 != v2:
                diffs.append({'package': pkg, 'env1': v1, 'env2': v2})
        return diffs


# 捕获环境
env = EnvironmentSnapshot.capture()
print(f"\n  Python: {env['python']['version'].split()[0]}")
print(f"  平台: {env['platform']['system']} {env['platform']['release']}")
print(f"  关键包版本:")
for pkg, ver in env['packages'].items():
    if ver != 'not_installed':
        print(f"    {pkg}: {ver}")

# 生成requirements
req_content = EnvironmentSnapshot.generate_requirements(env)
print(f"\n  requirements.txt:")
for line in req_content.split('\n')[:10]:
    print(f"    {line}")

# Dockerfile模板
print("""
  Docker环境锁定:
    FROM python:3.10-slim
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . /app
    WORKDIR /app
    CMD ["python", "train.py"]
""")

# ============================================================
# 4. 配置管理
# ============================================================
print("\n--- 4. 配置管理 ---")


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self.configs = {}
        self.config_history = []

    def load_from_dict(self, config_dict, name='default'):
        """从字典加载配置"""
        self.configs[name] = config_dict
        self.config_history.append({
            'name': name,
            'action': 'load',
            'timestamp': datetime.now().isoformat(),
            'config_hash': self._hash_config(config_dict),
        })
        return config_dict

    def save_to_json(self, config, filepath):
        """保存为JSON"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  配置已保存: {filepath}")

    def load_from_json(self, filepath):
        """从JSON加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"  配置已加载: {filepath}")
        return config

    def _hash_config(self, config):
        """计算配置哈希"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def diff_configs(self, config1, config2):
        """比较两个配置的差异"""
        diffs = []
        all_keys = set(str(k) for k in self._flatten(config1).keys()) | \
                   set(str(k) for k in self._flatten(config2).keys())
        flat1 = self._flatten(config1)
        flat2 = self._flatten(config2)

        for key in sorted(all_keys):
            v1 = flat1.get(key, '<missing>')
            v2 = flat2.get(key, '<missing>')
            if v1 != v2:
                diffs.append({'key': key, 'old': v1, 'new': v2})

        return diffs

    def _flatten(self, d, parent_key='', sep='.'):
        """扁平化嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def generate_yaml_template():
        """生成YAML配置模板"""
        return """
# 实验配置文件模板
experiment:
  name: "breast_cancer_v2"
  description: "乳腺癌分类实验"

data:
  dataset: "sklearn_breast_cancer"
  test_size: 0.2
  random_state: 42
  preprocessing:
    scaler: "StandardScaler"
    feature_selection: false

model:
  type: "RandomForest"
  params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    random_state: 42

training:
  cross_validation: 5
  scoring: "accuracy"
  early_stopping: false

tracking:
  experiment_name: "breast_cancer_classification"
  log_artifacts: true
  log_model: true
"""


# 创建实验配置
config_mgr = ConfigManager()

config_v1 = {
    'experiment': {'name': 'breast_cancer_v1'},
    'data': {'test_size': 0.2, 'random_state': 42},
    'model': {'type': 'RandomForest', 'params': {'n_estimators': 50, 'max_depth': 5}},
}

config_v2 = {
    'experiment': {'name': 'breast_cancer_v2'},
    'data': {'test_size': 0.2, 'random_state': 42},
    'model': {'type': 'RandomForest', 'params': {'n_estimators': 200, 'max_depth': 10}},
}

config_mgr.load_from_dict(config_v1, 'v1')
config_mgr.load_from_dict(config_v2, 'v2')

# 配置差异
diffs = config_mgr.diff_configs(config_v1, config_v2)
print("\n  配置差异:")
for d in diffs:
    print(f"    {d['key']}: {d['old']} -> {d['new']}")

# YAML模板
print(f"\n  YAML配置模板:")
print(config_mgr.generate_yaml_template())

# ============================================================
# 5. 代码版本与实验关联
# ============================================================
print("\n--- 5. 代码版本与实验关联 ---")


class ExperimentContext:
    """实验上下文: 关联代码版本、数据版本、配置"""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.context = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'git': self._get_git_info(),
            'environment': EnvironmentSnapshot.capture(),
            'data_hash': None,
            'config_hash': None,
        }

    def _get_git_info(self):
        """获取Git信息"""
        git_info = {
            'commit': 'unknown',
            'branch': 'unknown',
            'dirty': False,
        }
        try:
            import subprocess
            git_info['commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
            ).decode().strip()[:12]
            git_info['branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            pass
        return git_info

    def set_data_hash(self, data):
        """设置数据哈希"""
        if isinstance(data, pd.DataFrame):
            data_str = data.to_csv(index=False)
        else:
            data_str = str(data)
        self.context['data_hash'] = hashlib.md5(data_str.encode()).hexdigest()[:12]

    def set_config(self, config):
        """设置配置"""
        self.context['config'] = config
        self.context['config_hash'] = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]

    def fingerprint(self):
        """生成实验指纹"""
        fp_str = (
            f"{self.context['git']['commit']}-"
            f"{self.context['data_hash']}-"
            f"{self.context['config_hash']}"
        )
        return fp_str

    def summary(self):
        """打印实验上下文摘要"""
        print(f"\n  实验上下文: {self.experiment_name}")
        print(f"    时间: {self.context['timestamp']}")
        print(f"    Git commit: {self.context['git']['commit']}")
        print(f"    Git branch: {self.context['git']['branch']}")
        print(f"    数据哈希: {self.context['data_hash']}")
        print(f"    配置哈希: {self.context['config_hash']}")
        print(f"    实验指纹: {self.fingerprint()}")

    def to_json(self, filepath):
        """保存上下文"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.context, f, indent=2, ensure_ascii=False, default=str)


# 创建实验上下文
ctx = ExperimentContext('breast_cancer_v2')
ctx.set_data_hash(pd.DataFrame(X_train))
ctx.set_config(config_v2)
ctx.summary()

# ============================================================
# 6. 可复现训练流程示例
# ============================================================
print("\n--- 6. 可复现训练流程 ---")


def reproducible_train(config, seed=42):
    """可复现的训练函数"""
    # 1. 设置种子
    sm = SeedManager(seed)

    # 2. 加载和分割数据 (使用配置中的random_state)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # 3. 训练模型
    params = config['model']['params']
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 4. 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  可复现训练结果: accuracy={acc:.4f}")
    return model, acc


# 使用相同配置运行两次, 结果应该相同
print("\n  第一次运行:")
_, acc1 = reproducible_train(config_v2, seed=42)

print("\n  第二次运行 (相同配置):")
_, acc2 = reproducible_train(config_v2, seed=42)

print(f"\n  结果一致性: acc1={acc1:.4f}, acc2={acc2:.4f}, 相同={acc1 == acc2}")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 左: 种子对结果的影响
ax = axes[0]
ax.boxplot([results_no_seed, results_with_seed], labels=['无固定种子', '固定种子42'])
ax.set_ylabel('准确率')
ax.set_title('随机种子对结果的影响')
ax.grid(True, alpha=0.3)

# 中: 环境依赖图
ax = axes[1]
packages = {k: v for k, v in env['packages'].items() if v != 'not_installed'}
if packages:
    ax.barh(list(packages.keys()), [1]*len(packages), color='#2196F3')
    for i, (pkg, ver) in enumerate(packages.items()):
        ax.text(0.5, i, ver, ha='center', va='center', fontweight='bold', color='white')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title('环境依赖版本')

# 右: 可复现性检查清单
ax = axes[2]
ax.axis('off')
checklist = [
    ('随机种子固定', True),
    ('包版本锁定', True),
    ('数据版本哈希', True),
    ('Git commit记录', ctx.context['git']['commit'] != 'unknown'),
    ('配置文件管理', True),
    ('Docker镜像', False),
    ('CI/CD集成', False),
]
for i, (item, done) in enumerate(checklist):
    symbol = 'V' if done else 'x'
    color = '#4CAF50' if done else '#F44336'
    ax.text(0.1, 0.9 - i * 0.12, f'[{symbol}] {item}', fontsize=11,
            transform=ax.transAxes, color=color, fontfamily='monospace')
ax.set_title('可复现性检查清单')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/e5_reproducibility.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: e5_reproducibility.png")

print("\n完成! 实验可复现性要点:")
print("  1. 随机种子: 固定numpy/random/torch种子, 确保确定性")
print("  2. 环境锁定: requirements.txt + Docker")
print("  3. 配置管理: YAML/JSON参数文件, 版本化")
print("  4. 代码关联: Git commit + 数据哈希 + 配置哈希")
print("  5. 实验指纹: 唯一标识一次可复现实验")
