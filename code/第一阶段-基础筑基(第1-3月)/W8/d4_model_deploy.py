"""
W08 Day 4: 模型保存 + FastAPI推理服��
训练模型、保存模型、构建推理服务代码示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import json
import os
from datetime import datetime

# ============================================================
# Matplotlib 中文显示设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 训练模型
# ============================================================
print("=" * 60)
print("模型保存 + FastAPI 推理服务")
print("=" * 60)

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练 Pipeline
model_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    ))
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n模型训练完成, Test RMSE: {test_rmse:.4f}")

# ============================================================
# 2. 保存模型 + 版本管理
# ============================================================
print("\n--- 2. 模型保存 ---")

# 创建模型目录
MODEL_DIR = "model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

# 版本号 (基于日期)
version = datetime.now().strftime("v%Y%m%d_%H%M")
model_path = os.path.join(MODEL_DIR, f"california_housing_{version}.joblib")
scaler_path = os.path.join(MODEL_DIR, f"scaler_{version}.joblib")

# 保存完整 Pipeline
joblib.dump(model_pipeline, model_path)
print(f"  Pipeline 已保存: {model_path}")

# 也可以单独保存 scaler 和 model
joblib.dump(model_pipeline.named_steps['scaler'], scaler_path)
joblib.dump(model_pipeline.named_steps['model'],
            os.path.join(MODEL_DIR, f"model_{version}.joblib"))
print(f"  Scaler 已保存: {scaler_path}")

# 保存模型元信息
metadata = {
    "model_name": "California Housing Price Predictor",
    "version": version,
    "created_at": datetime.now().isoformat(),
    "features": list(X.columns),
    "test_rmse": float(test_rmse),
    "model_type": "GradientBoostingRegressor",
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1
    },
    "training_samples": len(X_train),
    "test_samples": len(X_test)
}

metadata_path = os.path.join(MODEL_DIR, f"metadata_{version}.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  元信息已保存: {metadata_path}")

# 同时保存一个 "latest" 版本的软链接（复制文件）
latest_path = os.path.join(MODEL_DIR, "latest_model.joblib")
joblib.dump(model_pipeline, latest_path)
print(f"  Latest 版本已保存: {latest_path}")

# ============================================================
# 3. 加载并验证模型
# ============================================================
print("\n--- 3. 加载并验证模型 ---")

loaded_pipeline = joblib.load(model_path)
loaded_pred = loaded_pipeline.predict(X_test[:5])
original_pred = model_pipeline.predict(X_test[:5])

print("  加载验证 (前5个样本):")
print(f"    原始预测: {original_pred.round(4)}")
print(f"    加载预测: {loaded_pred.round(4)}")
print(f"    一致性: {np.allclose(original_pred, loaded_pred)}")

# 加载元信息
with open(metadata_path, 'r', encoding='utf-8') as f:
    loaded_meta = json.load(f)
print(f"\n  模型版本: {loaded_meta['version']}")
print(f"  Test RMSE: {loaded_meta['test_rmse']:.4f}")

# ============================================================
# 4. 单条预测函数
# ============================================================
print("\n--- 4. 单条预测函数 ---")


def predict_single(features_dict: dict) -> float:
    """对单条数据进行预测"""
    feature_order = loaded_meta['features']
    values = [features_dict[feat] for feat in feature_order]
    X_single = np.array([values])
    prediction = loaded_pipeline.predict(X_single)[0]
    return float(prediction)


# 示例预测
sample_input = {
    'MedInc': 5.0,
    'HouseAge': 25.0,
    'AveRooms': 5.5,
    'AveBedrms': 1.1,
    'Population': 1500.0,
    'AveOccup': 3.0,
    'Latitude': 34.0,
    'Longitude': -118.5
}

predicted_price = predict_single(sample_input)
print(f"  输入特征: {sample_input}")
print(f"  预测房价: {predicted_price:.4f} 万美元")

# ============================================================
# 5. FastAPI 推理服务代码
# ============================================================
print("\n--- 5. FastAPI 推理服务代码 ---")

fastapi_code = '''
# ========================================
# FastAPI 推理服务 - app.py
# 安装: pip install fastapi uvicorn pydantic joblib scikit-learn
# 运行: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# ========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import json
from datetime import datetime

# ---- 初始化 ----
app = FastAPI(
    title="California Housing Price Predictor",
    description="基于 GradientBoosting 的加州房价预测 API",
    version="1.0.0"
)

# 加载模型
MODEL_PATH = "model_artifacts/latest_model.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
    with open("model_artifacts/metadata_latest.json", "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"模型文件未找到: {MODEL_PATH}")


# ---- 请求模型 ----
class HousingFeatures(BaseModel):
    """输入特征模型"""
    MedInc: float = Field(..., gt=0, description="收入中位数 (万美元)")
    HouseAge: float = Field(..., gt=0, description="房龄中位数 (年)")
    AveRooms: float = Field(..., gt=0, description="平均房间数")
    AveBedrms: float = Field(..., gt=0, description="平均卧室数")
    Population: float = Field(..., gt=0, description="人口")
    AveOccup: float = Field(..., gt=0, description="平均入住率")
    Latitude: float = Field(..., ge=32, le=42, description="纬度")
    Longitude: float = Field(..., ge=-125, le=-114, description="经度")

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 5.0,
                "HouseAge": 25.0,
                "AveRooms": 5.5,
                "AveBedrms": 1.1,
                "Population": 1500.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.5
            }
        }


class PredictionResponse(BaseModel):
    """预测响应模型"""
    predicted_price: float
    model_version: str
    prediction_time: str


# ---- 健康检查 ----
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


# ---- 模型信息 ----
@app.get("/model/info")
async def model_info():
    return {
        "model_name": metadata["model_name"],
        "version": metadata["version"],
        "model_type": metadata["model_type"],
        "test_rmse": metadata["test_rmse"],
        "features": metadata["features"]
    }


# ---- 预测接口 ----
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    """
    根据输入特征预测加州房价

    - **MedInc**: 收入中位数
    - **HouseAge**: 房龄中位数
    - **AveRooms**: 平均房间数
    - 其他特征...
    """
    try:
        feature_order = metadata["features"]
        values = [getattr(features, feat) for feat in feature_order]
        X = np.array([values])
        prediction = model_pipeline.predict(X)[0]

        return PredictionResponse(
            predicted_price=round(float(prediction), 4),
            model_version=metadata["version"],
            prediction_time=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 批量预测 ----
@app.post("/predict/batch")
async def predict_batch(features_list: list[HousingFeatures]):
    """批量预测"""
    if len(features_list) > 100:
        raise HTTPException(
            status_code=400,
            detail="批量预测最多支持100条数据"
        )

    results = []
    for features in features_list:
        feature_order = metadata["features"]
        values = [getattr(features, feat) for feat in feature_order]
        results.append(values)

    X = np.array(results)
    predictions = model_pipeline.predict(X)

    return {
        "predictions": [round(float(p), 4) for p in predictions],
        "count": len(predictions)
    }
'''

print("  FastAPI 代码已生成 (见上方)")
print("\n  启动命令:")
print("    uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
print("\n  测试 curl 命令:")
print('    curl -X POST "http://localhost:8000/predict" \\')
print('      -H "Content-Type: application/json" \\')
print('      -d \'{"MedInc":5.0,"HouseAge":25.0,"AveRooms":5.5,')
print('          "AveBedrms":1.1,"Population":1500.0,"AveOccup":3.0,')
print('          "Latitude":34.0,"Longitude":-118.5}\'')
print("\n  健康检查:")
print('    curl http://localhost:8000/health')
print("\n  模型信息:")
print('    curl http://localhost:8000/model/info')

# ============================================================
# 6. Flask 替代方案
# ============================================================
print("\n--- 6. Flask 替代方案 ---")

flask_code = '''
# ========================================
# Flask 推理服务 - flask_app.py
# 安装: pip install flask joblib scikit-learn
# 运行: python flask_app.py
# ========================================

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
model_pipeline = joblib.load("model_artifacts/latest_model.joblib")

FEATURE_ORDER = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # 输入验证
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"缺少特征: {missing}"}), 400

    values = [data[f] for f in FEATURE_ORDER]
    X = np.array([values])
    prediction = model_pipeline.predict(X)[0]

    return jsonify({
        "predicted_price": round(float(prediction), 4),
        "input_features": data
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''

print("  Flask 代码已生成 (见上方)")
print("  启动命令: python flask_app.py")
print("  测试: curl -X POST http://localhost:5000/predict -H ...")

# ============================================================
# 7. 模型版本管理概念
# ============================================================
print("\n--- 7. 模型版本管理最佳实践 ---")
print("""
  模型版本管理策略:

  1. 文件命名规范
     - model_v20260527_0800.joblib
     - metadata_v20260527_0800.json
     - latest_model.joblib (指向最新版本)

  2. 元信息记录
     - 训练时间、数据版本、超参数
     - 性能指标 (RMSE, MAE, R2)
     - 特征列表及顺序

  3. 版本控制工具
     - MLflow: 实验跟踪 + 模型注册
     - DVC: 数据版本控制
     - Git LFS: 大文件存储

  4. 部署策略
     - 蓝绿部署: 新旧版本同时运行
     - A/B 测试: 部分流量到新模型
     - 影子模式: 新模型只记录不响应
""")

# ============================================================
# 8. 模型性能可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 预测 vs 实际
y_pred_all = model_pipeline.predict(X_test)
axes[0].scatter(y_test, y_pred_all, alpha=0.3, s=5, c='steelblue')
axes[0].plot([0, 5.5], [0, 5.5], 'r--', linewidth=1.5, label='完美预测')
axes[0].set_xlabel('实际房价', fontsize=12)
axes[0].set_ylabel('预测房价', fontsize=12)
axes[0].set_title(f'模型预测性能 (RMSE={test_rmse:.4f})', fontsize=13)
axes[0].legend()

# 特征重要性
model = model_pipeline.named_steps['model']
importances = model.feature_importances_
feat_imp = pd.DataFrame({'特征': X.columns, '重要性': importances})
feat_imp = feat_imp.sort_values('重要性', ascending=True)
axes[1].barh(feat_imp['特征'], feat_imp['重要性'], color='teal', edgecolor='white')
axes[1].set_xlabel('重要性', fontsize=12)
axes[1].set_title('特征重要性', fontsize=13)

plt.tight_layout()
plt.savefig('d4_model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[图表已保存] d4_model_performance.png")

# 保存 FastAPI 和 Flask 代码到文件
with open(os.path.join(MODEL_DIR, 'app_fastapi.py'), 'w', encoding='utf-8') as f:
    f.write(fastapi_code)
print(f"[代码已保存] {os.path.join(MODEL_DIR, 'app_fastapi.py')}")

with open(os.path.join(MODEL_DIR, 'app_flask.py'), 'w', encoding='utf-8') as f:
    f.write(flask_code)
print(f"[代码已保存] {os.path.join(MODEL_DIR, 'app_flask.py')}")

print("\n" + "=" * 60)
print("模型保存 + 推理服务总结")
print("=" * 60)
print(f"""
1. 模型已保存至: {MODEL_DIR}/
2. 版本管理: {version}
3. Test RMSE: {test_rmse:.4f}
4. FastAPI 服务: {os.path.join(MODEL_DIR, 'app_fastapi.py')}
5. Flask 服务: {os.path.join(MODEL_DIR, 'app_flask.py')}
6. 单条预测示例: {predicted_price:.4f} 万美元
""")
