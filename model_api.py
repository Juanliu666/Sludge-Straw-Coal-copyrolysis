import os
import numpy as np
import joblib

# 需要 tensorflow 环境
from tensorflow.keras.models import load_model


# =========================================================
# 路径配置
# 与训练脚本保持一致：模型保存在 model_api.py 所在目录下的
# saved_models_attention_lstm/
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "saved_models_attention_lstm")

model_path = os.path.join(model_dir, "attention_lstm_model.keras")
scaler_x_path = os.path.join(model_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
feature_names_path = os.path.join(model_dir, "feature_names.pkl")
output_names_path = os.path.join(model_dir, "output_names.pkl")

# 兜底特征名（与 Attention-LSTM 训练脚本一致）
default_input_features = [
    '低阶煤比例（%）',
    '污泥比例（%）',
    '秸秆水解残渣比例（%）',
    '粘结剂比例（%）',
    '水分含量（%）',
    '挥发分含量（%）',
    'C 含量（%）',
    'N 含量（%）',
    'S 含量（%）',
    '热值（MJ/kg）',
    '热解温度（℃）',
    '停留时间（min）',
    '升温速率（℃/min）',
]

default_output_features = [
    '型炭产率（%）',
    '热解气产率（%）',
    '产物总能量（MJ/kg）',
]


def load_models():
    """
    加载训练好的 Attention-LSTM 模型和标准化器
    """
    try:
        # 加载 Keras 模型
        model = load_model(model_path)

        # 加载标准化器
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        # 加载特征名与输出名
        try:
            features = joblib.load(feature_names_path)
            outputs = joblib.load(output_names_path)
        except Exception:
            features = default_input_features
            outputs = default_output_features

        print("Attention-LSTM 模型和标准化器加载完成！")
        print(f"输入特征: {features}")
        print(f"输出特征: {outputs}")
        return model, scaler_X, scaler_y, features, outputs

    except FileNotFoundError as e:
        print(f"错误: 找不到模型文件 - {e}")
        print("请确保模型文件存在于正确的路径中。")
        return None, None, None, None, None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None, None, None, None


# 加载模型和标准化器
print("正在加载 Attention-LSTM 模型和标准化器...")
model, scaler_X, scaler_y, loaded_features, loaded_outputs = load_models()
print("Attention-LSTM 模型和标准化器加载完成！")


# =========================================================
# 内部工具
# =========================================================
def _to_lstm_input(x_scaled_2d: np.ndarray) -> np.ndarray:
    """
    将 (n_samples, n_features) 的标准化数据 reshape 为 LSTM 需要的
    (n_samples, time_steps=1, n_features) 三维格式。
    """
    return x_scaled_2d.reshape(x_scaled_2d.shape[0], 1, x_scaled_2d.shape[1])


# =========================================================
# 单样本预测
# =========================================================
def predict(input_values: list):
    """
    预测函数，接收输入值列表（按 loaded_features 的顺序），返回预测结果
    """
    try:
        print("-------")
        print(f"输入值: {input_values}")

        if model is None:
            return {'success': False, 'error': '模型未成功加载'}

        if len(input_values) != len(loaded_features):
            return {
                'success': False,
                'error': f'需要 {len(loaded_features)} 个特征，但只提供了 {len(input_values)} 个'
            }

        # 转为 numpy 数组 (1, n_features)
        new_data = np.array([input_values], dtype=float)

        # 标准化
        new_data_scaled = scaler_X.transform(new_data)

        # reshape 为 LSTM 输入 (1, 1, n_features)
        new_data_lstm = _to_lstm_input(new_data_scaled)

        # 预测
        prediction_scaled = model.predict(new_data_lstm, verbose=0)

        # 反标准化
        prediction = scaler_y.inverse_transform(prediction_scaled)

        # 组织结果
        results = {}
        for i, feature in enumerate(loaded_outputs):
            results[feature] = float(prediction[0][i])

        return {
            'success': True,
            'input_values': dict(zip(loaded_features, input_values)),
            'predictions': results
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# =========================================================
# 批量预测
# =========================================================
def predict_batch(input_data: list):
    """
    批量预测函数，接收多个输入值列表，返回预测结果
    """
    try:
        print("-------")
        print(f"批量预测，共 {len(input_data)} 组数据")

        if model is None:
            return {'success': False, 'error': '模型未成功加载'}

        if not isinstance(input_data, list):
            return {'success': False, 'error': '输入数据必须是列表格式'}

        for i, values in enumerate(input_data):
            if len(values) != len(loaded_features):
                return {
                    'success': False,
                    'error': f'第 {i+1} 组数据需要 {len(loaded_features)} 个特征，但只提供了 {len(values)} 个'
                }

        new_data = np.array(input_data, dtype=float)
        new_data_scaled = scaler_X.transform(new_data)
        new_data_lstm = _to_lstm_input(new_data_scaled)

        prediction_scaled = model.predict(new_data_lstm, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        results = []
        for j in range(len(prediction)):
            result = {}
            for i, feature in enumerate(loaded_outputs):
                result[feature] = float(prediction[j][i])
            results.append(result)

        return {
            'success': True,
            'input_values': [dict(zip(loaded_features, values)) for values in input_data],
            'predictions': results
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# =========================================================
# 字典输入预测
# =========================================================
def predict_with_dict(input_dict: dict):
    """
    使用字典形式进行预测
    """
    try:
        print("-------")
        print(f"输入字典: {input_dict}")

        input_values = []
        for feature in loaded_features:
            if feature in input_dict:
                input_values.append(input_dict[feature])
            else:
                return {
                    'success': False,
                    'error': f'缺少特征: {feature}'
                }

        return predict(input_values)

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }