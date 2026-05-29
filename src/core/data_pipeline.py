"""
数据处理流水线 — 多源传感器数据读取、标准化、序列化
支持：分布式光纤、渗压计、位移计、加速度计等水利工程常见传感器
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


# 水利工程常见传感器类型
SENSOR_TYPES = {
    'strain': '应变/光纤传感器',
    'seepage': '渗压计',
    'displacement': '位移计',
    'settlement': '沉降仪',
    'acceleration': '加速度计（地震）',
    'temperature': '温度计',
    'water_level': '水位计',
}

# 灾害类型
HAZARD_TYPES = ['seepage', 'deformation', 'settlement', 'seismic']

# 损伤等级阈值（基于标准规范）
DAMAGE_THRESHOLDS = {
    'normal': {'span': 5, 'label': '正常'},
    'mild': {'span': 30, 'label': '轻度损伤'},
    'moderate': {'span': 100, 'label': '中度损伤'},
    'severe': {'span': 200, 'label': '重度损伤'},
}


def build_sliding_windows(data_matrix, positions, sequence_length=30, pred_horizon=10):
    """Sliding window over spatial dimension for sequence-to-sequence learning."""
    X_list, y_list, meta_list = [], [], []

    total_len = len(positions)
    for start in range(0, total_len - sequence_length - pred_horizon, pred_horizon):
        end_input = start + sequence_length
        end_output = end_input + pred_horizon

        X = data_matrix[start:end_input]
        y = data_matrix[end_input:end_output]

        # Determine damage condition
        strain_slice = data_matrix[start:end_output]
        strain_mean = strain_slice.mean(axis=1) if strain_slice.ndim > 1 else strain_slice

        # Compute relaxation zone span
        rz_span = compute_damage_span(strain_mean, positions[start:end_output])

        if rz_span < DAMAGE_THRESHOLDS['normal']['span']:
            condition = 'normal'
        elif rz_span < DAMAGE_THRESHOLDS['mild']['span']:
            condition = 'mild_damage'
        elif rz_span < DAMAGE_THRESHOLDS['moderate']['span']:
            condition = 'moderate_damage'
        else:
            condition = 'severe_damage'

        X_list.append(X)
        y_list.append(y)
        meta_list.append({
            'start_pos': positions[start],
            'end_pos': positions[end_output],
            'condition': condition,
            'rz_span': rz_span,
        })

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, meta_list


def compute_damage_span(values, positions):
    """Identify damage zone span based on value drop from edge."""
    valid = ~np.isnan(values)
    if np.sum(valid) < 10:
        return 0.0

    vals = values[valid]
    pos = positions[valid]
    edge_mean = (np.mean(vals[:min(5, len(vals))]) + np.mean(vals[-min(5, len(vals)):])) / 2
    threshold = edge_mean * 0.7
    relaxed_mask = vals < threshold

    if np.sum(relaxed_mask) == 0:
        return 0.0

    relaxed_indices = np.where(relaxed_mask)[0]
    return float(pos[relaxed_indices[-1]] - pos[relaxed_indices[0]])


def load_csv_data(filepath, header_row=0, pos_col=1):
    """Load generic CSV sensor data with position column."""
    df = pd.read_csv(filepath, header=header_row) if str(filepath).endswith('.csv') else pd.read_excel(filepath)

    # Attempt to auto-detect position and sensor columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError(f"Need at least 2 numeric columns in {filepath}, got {len(numeric_cols)}")

    positions = df.iloc[:, pos_col].values.astype(np.float64)
    sensor_cols = [c for c in numeric_cols if c != df.columns[pos_col]]
    sensor_data = df[sensor_cols].values.astype(np.float64)

    return positions, sensor_data, sensor_cols


def normalize_data(X, y=None, scaler_type='standard'):
    """Normalize data with fit on X only."""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_flat).reshape(original_shape)

    if y is not None:
        # Use same scaler for y (since same features)
        y_flat = y.reshape(-1, y.shape[-1])
        y_scaled = scaler.transform(y_flat).reshape(y.shape)
        return X_scaled, y_scaled, scaler

    return X_scaled, None, scaler


def save_scaler(scaler, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def merge_multi_hazard_data(strain_data=None, seepage_data=None, displacement_data=None,
                            settlement_data=None, seismic_data=None):
    """Merge multi-hazard sensor data into unified feature matrix.

    Each input: dict with 'positions' and 'values' (2D array: [n_positions, n_features]).

    Returns:
        merged_positions: aligned position grid
        merged_features: [n_positions, total_features]
        feature_labels: list of feature names
    """
    datasets = []
    labels = []

    if strain_data is not None:
        datasets.append(strain_data['values'])
        for i in range(strain_data['values'].shape[1]):
            labels.append(f'strain_{i}')
    if seepage_data is not None:
        datasets.append(seepage_data['values'])
        for i in range(seepage_data['values'].shape[1]):
            labels.append(f'seepage_{i}')
    if displacement_data is not None:
        datasets.append(displacement_data['values'])
        for i in range(displacement_data['values'].shape[1]):
            labels.append(f'displacement_{i}')
    if settlement_data is not None:
        datasets.append(settlement_data['values'])
        for i in range(settlement_data['values'].shape[1]):
            labels.append(f'settlement_{i}')
    if seismic_data is not None:
        datasets.append(seismic_data['values'])
        for i in range(seismic_data['values'].shape[1]):
            labels.append(f'seismic_{i}')

    if not datasets:
        raise ValueError("At least one sensor dataset required")

    # Use the first dataset's positions as reference
    merged_features = np.column_stack(datasets)

    # Ensure all have same number of positions
    min_positions = min(d.shape[0] for d in datasets)
    merged_features = merged_features[:min_positions]

    return merged_features, labels


def generate_sample_data(n_positions=360, n_sensors=9):
    """Generate synthetic sample data for testing pipeline."""
    rng = np.random.RandomState(42)
    positions = np.linspace(0, 360, n_positions)

    # Base strain pattern
    strain_base = 100 + 10 * np.sin(positions * np.pi / 180)
    # Add damage zone at center
    damage_zone = np.ones(n_positions)
    center = 180
    sigma = 40
    damage_zone = 1 - 0.4 * np.exp(-((positions - center) ** 2) / (2 * sigma ** 2))

    sensor_matrix = np.zeros((n_positions, n_sensors))
    for i in range(n_sensors):
        noise = rng.randn(n_positions) * 2
        sensor_matrix[:, i] = strain_base + noise + (i * 5) * damage_zone

    return positions, sensor_matrix


if __name__ == '__main__':
    positions, data = generate_sample_data()
    X, y, meta = build_sliding_windows(data, positions)
    X_norm, y_norm, scaler = normalize_data(X, y)
    print(f"Sample data: {len(positions)} positions, {data.shape[1]} sensors")
    print(f"Windows: X={X.shape}, y={y.shape}")
    print(f"Normalized: X={X_norm.shape}, y={y_norm.shape}")
    conditions = [m['condition'] for m in meta]
    for cond in ['normal', 'mild_damage', 'moderate_damage', 'severe_damage']:
        print(f"  {cond}: {conditions.count(cond)}")
