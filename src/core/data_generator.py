"""
Physics-based multi-hazard dam monitoring data generator.

数据源标定：
  - 小湾拱坝 (Xiaowan): 292m, 935m弧长, 12m/73m 顶/底厚, E=21GPa
    来源: NCDC 国家冰川冻土沙漠科学数据中心 (https://www.ncdc.ac.cn)
    数据集: "小湾拱坝坝踵竖向应力2017年后计算值与监测值对比"
    数据集: "小湾拱坝应力场重构计算结果"
  - 构皮滩拱坝 (Goupitan): 232.5m, 抛物线型拱圈, E=36.9-44.5GPa
    来源: NCDC 数据集 "构皮滩拱坝放空安全控制指标体系原始数据集"
    文献: 构皮滩水电站拱坝设计 (中国大坝协会2012学术年会)
  - ICOLD Benchmark: Pine Flat 混凝土重力坝 (122m, 561m坝顶长)
    来源: 15th ICOLD International Benchmark Workshop (Milan, 2019)
    Proceedings: Numerical Analysis of Dams, Springer, 2021

Generates realistic 9-channel sensor data for arch dam health monitoring
with 4 hazard types (seepage, deformation, settlement, seismic) and 4 severity
levels each, grounded in structural mechanics first principles.

Sensor channels (9-D feature vector):
  0-2: Distributed fiber optic strain      [ue]   — 3 spatial zones
  3-4: Piezometer pore pressure            [kPa]  — upstream/downstream
  5-6: Displacement meters                 [mm]   — radial/tangential
  7:   Settlement gauge                    [mm]   — vertical
  8:   Accelerometer                       [gal]  — seismic (1 gal = 1 cm/s2)

Hazard physics:
  Seepage:     pore pressure UP, strain DOWN (effective stress loss), displacement UP
  Deformation: strain DOWN localized (relaxation zone), displacement UP
  Settlement:  settlement UP (differential), strain dipole pattern
  Seismic:     acceleration UP transient, all channels oscillatory
"""
import numpy as np
from pathlib import Path


# ── Arch dam profile presets (calibrated from real-world data) ──
# Each preset captures geometry, material, and operational parameters
# of a real-world arch dam, used to scale the physics simulation.

DAM_PROFILES = {
    'xiaowan': {   # Xiaowan: world's 2nd highest arch dam, Lancang River, Yunnan
        'name': 'Xiaowan Arch Dam (292m)',
        'height_m': 292.0,
        'crest_length_m': 935.0,
        'crest_thickness_m': 12.0,
        'base_thickness_m': 73.0,
        'elastic_modulus_gpa': 21.0,
        'density_kgm3': 2400,
        'poisson_ratio': 0.189,
        'normal_pool_m': 1240.0,    # 正常蓄水位高程
        'low_pool_m': 1181.0,       # 低水位高程
        'design_pga_g': 0.308,      # 设计地震加速度 (IX度)
        'valley_shape': 'V',        # V valley, 40-42 deg bank slope
        'source': 'NCDC · 国家冰川冻土沙漠科学数据中心',
        'dataset_ids': [
            '8236cd76-671e-4a4a-9837-9551dc108dd6',  # 小湾坝踵应力
            '4f568cca-94ee-45d1-be4c-ced00c6e6d06',  # 小湾应力场重构
        ],
    },
    'goupitan': {  # Goupitan Arch Dam, Wujiang River, Guizhou
        'name': 'Goupitan Arch Dam (232.5m)',
        'height_m': 232.5,
        'crest_length_m': 560.0,     # estimated from arch description
        'crest_thickness_m': 10.25,
        'base_thickness_m': 50.28,
        'elastic_modulus_gpa': 38.0,  # 28d 中值 (36.9-40.0)
        'density_kgm3': 2400,
        'poisson_ratio': 0.167,
        'normal_pool_m': 630.0,
        'low_pool_m': 585.0,
        'design_pga_g': 0.10,        # VI度设防
        'valley_shape': 'V',
        'max_central_angle_deg': 88.07,
        'source': 'NCDC · 国家冰川冻土沙漠科学数据中心',
        'dataset_ids': [
            '89b27f1a-2cf1-4f3b-b4c4-b6412285ed78',  # 构皮滩放空安全
        ],
    },
    'pineflat_gravity': {  # Pine Flat | ICOLD Benchmark | Concrete Gravity Dam
        'name': 'Pine Flat Gravity Dam (ICOLD Benchmark)',
        'height_m': 122.0,            # 400 ft
        'crest_length_m': 561.0,      # 1,840 ft
        'crest_thickness_m': 9.8,     # 32 ft
        'base_thickness_m': 97.5,     # ~320 ft
        'elastic_modulus_gpa': 22.4,
        'density_kgm3': 2483,
        'poisson_ratio': 0.20,
        'normal_pool_m': 290.0,
        'low_pool_m': 260.0,
        'design_pga_g': 0.20,
        'valley_shape': 'U',          # wide U valley
        'dam_type': 'gravity',        # gravity dam (not arch)
        'source': '15th ICOLD International Benchmark Workshop (Milan, 2019)',
        'reference': 'Bolzon, G. et al. (2021) Numerical Analysis of Dams. Springer.',
    },
}

# Default active profile
ACTIVE_PROFILE = 'xiaowan'

# ── Physical constants derived from active profile ──
def _get_profile():
    return DAM_PROFILES[ACTIVE_PROFILE]


def _dam_height():
    return _get_profile()['height_m']


def _crest_length():
    return _get_profile()['crest_length_m']


def _elastic_modulus_pa():
    """Elastic modulus in Pa."""
    return _get_profile()['elastic_modulus_gpa'] * 1e9


# Spatial resolution
N_POSITIONS = 360          # 1 degree resolution along crest (scales with profile)
N_FEATURES = 9


def _baseline_params():
    """Compute baseline sensor values from dam profile physics.

    Strain = sigma/E where sigma ~ rho*g*h at dam/3 for arch action
    Pore pressure ~ gamma_water*h at monitoring depth
    Displacement ~ (F*L^3)/(3*E*I) for simplified arch beam
    """
    profile = _get_profile()
    H = profile['height_m']
    E = profile['elastic_modulus_gpa'] * 1e9  # Pa
    rho_c = profile['density_kgm3']           # kg/m3
    rho_w = 1000.0                             # water density kg/m3
    h_water = H * 0.85                         # approximate water head at monitoring points

    # Hydrostatic stress ~ rho*g*h at 1/3 height (approximate arch ring)
    sigma_hydrostatic = rho_w * 9.81 * h_water  # Pa
    # Strain: sigma/E * 1e6 [microstrain]
    strain_baseline = (sigma_hydrostatic / E) * 1e6

    # Pore pressure: uplift at monitoring points [kN/m2 = kPa]
    pore_pressure_baseline = rho_w * 9.81 * h_water * 0.3 / 1000  # kPa (30% uplift)

    # Displacement: simplified arch deflection
    # delta ~ (w*L^4)/(384*E*I) scaled
    L = profile['crest_length_m']
    I_equiv = (profile['base_thickness_m'] ** 4) / 12  # approximate per meter
    w = rho_w * 9.81 * h_water * 1.0  # load per meter
    displacement_baseline = (w * L**4) / (384 * E * I_equiv) * 1000  # mm

    # Clamp to physically reasonable ranges
    strain_baseline = max(strain_baseline, 50.0)
    pore_pressure_baseline = max(pore_pressure_baseline, 30.0)
    displacement_baseline = max(displacement_baseline, 0.5)

    return {
        'strain':       strain_baseline,
        'pore_pressure': pore_pressure_baseline,
        'displacement':  displacement_baseline,
        'settlement':    max(0.2, displacement_baseline * 0.05),  # small in normal state
        'acceleration':  0.5,  # ambient microtremor [gal]
    }


# Legacy BASELINE for backward compat; updated on first generate_base_state call
BASELINE = {
    'strain':       120.0,
    'pore_pressure': 80.0,
    'displacement':   3.0,
    'settlement':     0.5,
    'acceleration':   0.5,
}


# Hazard-specific amplitude ranges by severity [normal, mild, moderate, severe]
SEVERITY_AMPLITUDE = {
    'seepage': {
        'pore_pressure_factor': [1.0, 1.5, 2.5, 4.0],
        'zone_width_deg':       [0,   15,  40,  80],
    },
    'deformation': {
        'strain_drop_pct':      [0,   20,  45,  70],
        'zone_width_deg':       [0,   12,  35,  90],
    },
    'settlement': {
        'settlement_mm':        [0.5, 8,   25,  60],
        'zone_width_deg':       [0,   20,  50,  100],
    },
    'seismic': {
        'pga_gal':              [0.5, 25,  80,  200],
        'zone_width_deg':       [0,   60,  120, 200],
    },
}

HAZARD_NAMES = ['seepage', 'deformation', 'settlement', 'seismic']
SEVERITY_NAMES = ['normal', 'mild', 'moderate', 'severe']


def _gkern(width_deg, center_deg=None, n=N_POSITIONS):
    """Gaussian kernel for spatial smoothing of damage zones."""
    if center_deg is None:
        center_deg = N_POSITIONS / 2
    x = np.arange(n)
    sigma = width_deg / 2.355  # FWHM -> sigma
    if sigma < 0.5:
        sigma = 0.5
    kernel = np.exp(-0.5 * ((x - center_deg) / sigma) ** 2)
    return kernel


def _smooth_noise(rng, n, scale=1.0, smoothness=3):
    """Spatially correlated noise via moving average of white noise."""
    raw = rng.randn(n + 2 * smoothness) * scale
    kernel = np.ones(smoothness) / smoothness
    smoothed = np.convolve(raw, kernel, mode='valid')
    return smoothed[:n]


def generate_base_state(rng=None):
    """Generate normal-state sensor readings for a healthy arch dam.

    Physics: sinusoidal variation along crest — maximum at crown (center),
    minimum at abutments (edges), reflecting arch action.

    Baseline values are derived from the active DAM_PROFILE (Xiaowan by default)
    using hydrostatic stress, arch deflection, and uplift pressure calculations.
    """
    profile = _get_profile()
    H = profile['height_m']
    L = profile['crest_length_m']

    # Calibrate baselines from profile physics
    bl = _baseline_params()
    # Update global BASELINE for backward compat
    for k, v in bl.items():
        BASELINE[k] = v

    rng = rng or np.random.RandomState()
    positions = np.linspace(0, L, N_POSITIONS)
    theta = positions / L * np.pi  # normalized arch angle [0, π]

    # Arch action: higher loading at crown (center), lower at abutments
    arch_profile = 0.7 + 0.3 * np.sin(theta)  # [0.7, 1.0] — crowned at midspan

    data = np.zeros((N_POSITIONS, N_FEATURES))

    # Channels 0-2: Strain (3 spatial zones — left abutment, crown, right abutment)
    for i, center in enumerate([N_POSITIONS * 0.25, N_POSITIONS * 0.5, N_POSITIONS * 0.75]):
        zone_profile = 1.0 + 0.15 * np.sin(theta - np.pi * (center / N_POSITIONS - 0.5))
        data[:, i] = bl['strain'] * arch_profile * zone_profile
        data[:, i] += _smooth_noise(rng, N_POSITIONS, scale=bl['strain'] * 0.02, smoothness=5)

    # Channels 3-4: Pore pressure (upstream/downstream)
    data[:, 3] = bl['pore_pressure'] * arch_profile  # upstream — higher
    data[:, 3] += _smooth_noise(rng, N_POSITIONS, scale=bl['pore_pressure'] * 0.04, smoothness=8)
    data[:, 4] = bl['pore_pressure'] * 0.6 * arch_profile  # downstream — lower
    data[:, 4] += _smooth_noise(rng, N_POSITIONS, scale=bl['pore_pressure'] * 0.03, smoothness=8)

    # Channels 5-6: Displacement (radial, tangential)
    data[:, 5] = bl['displacement'] * arch_profile  # radial — follows arch load
    data[:, 5] += _smooth_noise(rng, N_POSITIONS, scale=bl['displacement'] * 0.1, smoothness=6)
    data[:, 6] = bl['displacement'] * 0.3  # tangential — small in symmetric loading
    data[:, 6] += _smooth_noise(rng, N_POSITIONS, scale=bl['displacement'] * 0.07, smoothness=6)

    # Channel 7: Settlement (near zero in normal state)
    data[:, 7] = bl['settlement'] + _smooth_noise(rng, N_POSITIONS, scale=bl['settlement'] * 0.2, smoothness=10)

    # Channel 8: Acceleration (ambient microtremor)
    data[:, 8] = bl['acceleration'] + rng.randn(N_POSITIONS) * 0.2

    return positions, data


def apply_seepage_damage(data, severity, rng, center_deg=None):
    """Apply seepage damage: elevated pore pressure, reduced effective stress.

    Physics: Increased pore pressure reduces effective stress in concrete,
    causing slight strain decrease. Localized near cracks/joints.
    """
    params = SEVERITY_AMPLITUDE['seepage']
    center = center_deg if center_deg is not None else rng.uniform(60, N_POSITIONS - 60)
    zone = _gkern(params['zone_width_deg'][severity], center)

    factor = params['pore_pressure_factor'][severity]

    # Pore pressure increases in the zone
    data[:, 3] += BASELINE['pore_pressure'] * (factor - 1.0) * zone
    data[:, 4] += BASELINE['pore_pressure'] * 0.6 * (factor - 1.0) * zone

    # Strain slightly decreases (effective stress reduction)
    strain_drop = BASELINE['strain'] * 0.08 * severity * zone
    for ch in [0, 1, 2]:
        data[:, ch] -= strain_drop

    # Minor displacement increase
    data[:, 5] += BASELINE['displacement'] * 0.3 * severity * zone

    return data


def apply_deformation_damage(data, severity, rng, center_deg=None):
    """Apply deformation damage: localized strain relaxation (V-shaped zone).

    Physics: Concrete cracking / steel yielding causes stress redistribution.
    Strain drops in the damaged zone (relaxation), displacement increases.
    """
    params = SEVERITY_AMPLITUDE['deformation']
    center = center_deg if center_deg is not None else rng.uniform(60, N_POSITIONS - 60)
    zone = _gkern(params['zone_width_deg'][severity], center)

    drop_pct = params['strain_drop_pct'][severity] / 100.0

    # Strain drops in damage zone
    for ch in [0, 1, 2]:
        data[:, ch] -= BASELINE['strain'] * drop_pct * zone

    # Displacement increases (radial more than tangential)
    data[:, 5] += BASELINE['displacement'] * 3.0 * severity * zone  # radial — large change
    data[:, 6] += BASELINE['displacement'] * 1.5 * severity * zone  # tangential

    # Add strain "rebound" at zone edges (elastic recovery) for severity ≥ 2
    if severity >= 2:
        edge_zone = _gkern(params['zone_width_deg'][severity] * 1.3, center) - zone
        edge_zone = np.clip(edge_zone, 0, None)
        for ch in [0, 1, 2]:
            data[:, ch] += BASELINE['strain'] * 0.05 * edge_zone

    return data


def apply_settlement_damage(data, severity, rng, center_deg=None):
    """Apply settlement damage: differential vertical displacement.

    Physics: Foundation erosion or consolidation causes uneven settlement,
    creating a strain dipole pattern (tension on one side, compression on other).
    """
    params = SEVERITY_AMPLITUDE['settlement']
    center = center_deg if center_deg is not None else rng.uniform(60, N_POSITIONS - 60)
    zone = _gkern(params['zone_width_deg'][severity], center)

    sett_mm = params['settlement_mm'][severity]

    # Settlement increases in the zone
    data[:, 7] += sett_mm * zone

    # Strain dipole: compression on upstream side, tension on downstream
    dipole = np.gradient(zone)
    dipole = dipole / (np.abs(dipole).max() + 1e-10)  # normalize to [-1, 1]
    for ch in [0, 1, 2]:
        data[:, ch] += BASELINE['strain'] * 0.15 * severity * dipole

    # Radial displacement follows settlement pattern
    data[:, 5] += sett_mm * 0.4 * zone

    # Pore pressure may increase (consolidation effect)
    if severity >= 1:
        data[:, 3] += BASELINE['pore_pressure'] * 0.1 * severity * zone

    return data


def apply_seismic_damage(data, severity, rng, center_deg=None):
    """Apply seismic damage: transient acceleration + oscillatory strain.

    Physics: Earthquake ground motion causes high-frequency acceleration
    and oscillatory strain throughout the structure. Damage concentrates
    at structural weak points.
    """
    params = SEVERITY_AMPLITUDE['seismic']
    center = center_deg if center_deg is not None else rng.uniform(60, N_POSITIONS - 60)

    pga = params['pga_gal'][severity]

    # Acceleration channel: high-amplitude oscillatory signal
    n = N_POSITIONS
    # Multi-frequency seismic signal (1-10 Hz scaled to spatial domain)
    t = np.arange(n) / n
    seismic_signal = np.zeros(n)
    freqs = [2.0, 4.5, 7.0]  # dominant frequencies
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi)
        seismic_signal += np.sin(2 * np.pi * f * t + phase) / len(freqs)
    # Envelope: strong shaking near epicenter, decay with distance
    zone = _gkern(params['zone_width_deg'][severity], center)
    envelope = zone + 0.05  # low-level shaking everywhere
    data[:, 8] = pga * seismic_signal * envelope + rng.randn(n) * pga * 0.05

    # Strain channels: oscillatory response
    for ch in [0, 1, 2]:
        osc = BASELINE['strain'] * 0.08 * severity * seismic_signal * envelope
        data[:, ch] += osc

    # Displacement: transient
    data[:, 5] += BASELINE['displacement'] * 0.5 * severity * np.abs(seismic_signal) * envelope
    data[:, 6] += BASELINE['displacement'] * 0.3 * severity * np.abs(seismic_signal) * envelope

    # Pore pressure: may spike due to soil liquefaction effect
    if severity >= 2:
        data[:, 3] += BASELINE['pore_pressure'] * 0.4 * severity * zone

    return data


def generate_multi_hazard_dataset(n_samples=2000, seed=42):
    """Generate a complete multi-hazard training dataset.

    Returns:
        positions:  np.array [N_POSITIONS]
        data:       np.array [n_samples, N_POSITIONS, N_FEATURES]
        hazard_labels: np.array [n_samples] — 0=seepage, 1=deformation, 2=settlement, 3=seismic
        severity_labels: np.array [n_samples] — 0=normal, 1=mild, 2=moderate, 3=severe
        damage_spans: np.array [n_samples] — damage zone width in degrees
    """
    rng = np.random.RandomState(seed)
    positions = np.linspace(0, _get_profile()['crest_length_m'], N_POSITIONS)

    data_list = []
    hazard_list = []
    severity_list = []
    span_list = []

    # Distribution: ~10% normal, ~90% damaged (balanced across 4 hazards × 3 severities)
    samples_per_class = n_samples // 13  # 1 normal + 4 hazards × 3 severities
    samples_per_class = max(samples_per_class, 10)

    hazard_appliers = {
        0: apply_seepage_damage,      # seepage
        1: apply_deformation_damage,   # deformation
        2: apply_settlement_damage,    # settlement
        3: apply_seismic_damage,       # seismic
    }

    # Normal samples (severity=0)
    n_normal = samples_per_class
    for _ in range(n_normal):
        _, base_data = generate_base_state(rng)
        h_type = rng.choice(4)
        data_list.append(base_data)
        hazard_list.append(h_type)
        severity_list.append(0)
        span_list.append(0.0)

    # Damaged samples: 4 hazards × 3 severities × samples_per_class
    for h_type in range(4):
        for severity in range(1, 4):
            for _ in range(samples_per_class):
                _, base_data = generate_base_state(rng)
                # Random damage center for variety
                center = rng.uniform(50, N_POSITIONS - 50)
                damaged_data = hazard_appliers[h_type](
                    base_data.copy(), severity, rng, center
                )
                data_list.append(damaged_data)
                hazard_list.append(h_type)
                severity_list.append(severity)

                # Record approximate damage span
                zone_w = SEVERITY_AMPLITUDE[HAZARD_NAMES[h_type]]['zone_width_deg'][severity]
                span_list.append(zone_w + rng.uniform(-zone_w * 0.2, zone_w * 0.2))

    data = np.array(data_list, dtype=np.float32)
    hazard_labels = np.array(hazard_list, dtype=np.int64)
    severity_labels = np.array(severity_list, dtype=np.int64)
    damage_spans = np.array(span_list, dtype=np.float32)

    # Shuffle
    idx = rng.permutation(len(data_list))
    data = data[idx]
    hazard_labels = hazard_labels[idx]
    severity_labels = severity_labels[idx]
    damage_spans = damage_spans[idx]

    return positions, data, hazard_labels, severity_labels, damage_spans


def generate_sliding_window_dataset(positions, full_data, hazard_labels, severity_labels,
                                    damage_spans, seq_length=30, pred_horizon=10, stride=10):
    """Convert full-position data into sliding-window training samples.

    Each full sample [N_POSITIONS, N_FEATURES] is sliced into windows of
    [seq_length, N_FEATURES] as input and [pred_horizon, N_FEATURES] as target.
    Hazard/severity labels are inherited from the parent sample.

    Returns:
        X:          np.array [n_windows, seq_length, N_FEATURES]
        y_sensor:   np.array [n_windows, pred_horizon, N_FEATURES]
        y_hazard:   np.array [n_windows] — hazard type label
        y_severity: np.array [n_windows] — severity label
        y_span:     np.array [n_windows] — damage span regression target
    """
    n_samples, n_positions, n_features = full_data.shape

    X_list, y_sensor_list = [], []
    y_hazard_list, y_severity_list, y_span_list = [], [], []

    for i in range(n_samples):
        sample = full_data[i]
        for start in range(0, n_positions - seq_length - pred_horizon, stride):
            end_in = start + seq_length
            end_out = end_in + pred_horizon
            X_list.append(sample[start:end_in])
            y_sensor_list.append(sample[end_in:end_out])
            y_hazard_list.append(hazard_labels[i])
            y_severity_list.append(severity_labels[i])

            # Scale span to local window context
            local_span_fraction = damage_spans[i] / n_positions
            y_span_list.append(local_span_fraction * pred_horizon)

    X = np.array(X_list, dtype=np.float32)
    y_sensor = np.array(y_sensor_list, dtype=np.float32)
    y_hazard = np.array(y_hazard_list, dtype=np.int64)
    y_severity = np.array(y_severity_list, dtype=np.int64)
    y_span = np.array(y_span_list, dtype=np.float32)

    print(f"Window dataset: X={X.shape}, y_sensor={y_sensor.shape}, "
          f"y_hazard={y_hazard.shape}, y_severity={y_severity.shape}, y_span={y_span.shape}")

    # Class distribution
    for name, labels in [('Hazard', y_hazard), ('Severity', y_severity)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = ', '.join(f'{u}({HAZARD_NAMES[u] if name == "Hazard" else SEVERITY_NAMES[u]}):{c}'
                        for u, c in zip(unique, counts))
        print(f"  {name} distribution: {dist}")

    return X, y_sensor, y_hazard, y_severity, y_span


if __name__ == '__main__':
    profile = _get_profile()
    bl = _baseline_params()
    print(f"Active profile: {profile['name']}")
    print(f"  Height: {profile['height_m']}m | Crest: {profile['crest_length_m']}m")
    print(f"  E = {profile['elastic_modulus_gpa']} GPa | rho = {profile['density_kgm3']} kg/m3 | nu = {profile['poisson_ratio']}")
    print(f"  Source: {profile['source']}")
    print(f"  Computed baselines: strain={bl['strain']:.1f} ue, PP={bl['pore_pressure']:.1f} kPa, "
          f"disp={bl['displacement']:.2f} mm, sett={bl['settlement']:.2f} mm, acc={bl['acceleration']:.1f} gal")
    print()

    positions, data, h_labels, s_labels, spans = generate_multi_hazard_dataset(n_samples=1300)
    print(f"Dataset: {data.shape}, hazards={h_labels.shape}, severities={s_labels.shape}")
    print(f"Physical range check -- strain: [{data[:,:,0].min():.0f}, {data[:,:,0].max():.0f}] ue")

    X, ys, yh, ysev, ysp = generate_sliding_window_dataset(
        positions, data, h_labels, s_labels, spans
    )
    print()
    print("Available profiles:")
    for k, p in DAM_PROFILES.items():
        print(f"  {k}: {p['name']} -- {p['height_m']}m -- {p['source']}")
    print()
    print("Tip: Set ACTIVE_PROFILE to switch dam type.")
