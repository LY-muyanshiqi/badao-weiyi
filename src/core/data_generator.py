"""
Physics-based multi-hazard dam monitoring data generator.

Generates realistic 9-channel sensor data for arch dam health monitoring
with 4 hazard types (seepage, deformation, settlement, seismic) and 4 severity
levels each, grounded in structural mechanics first principles.

Sensor channels (9-D feature vector):
  0-2: Distributed fiber optic strain      [με]   — 3 spatial zones
  3-4: Piezometer pore pressure            [kPa]  — upstream/downstream
  5-6: Displacement meters                 [mm]   — radial/tangential
  7:   Settlement gauge                    [mm]   — vertical
  8:   Accelerometer                       [gal]  — seismic (1 gal ≈ 1 cm/s²)

Hazard physics:
  Seepage:     pore pressure ↑↑, strain ↓ (effective stress loss), displacement ↑
  Deformation: strain ↓↓ localized (relaxation zone), displacement ↑↑
  Settlement:  settlement ↑↑ (differential), strain dipole pattern
  Seismic:     acceleration ↑↑↑ transient, all channels oscillatory
"""
import numpy as np
from pathlib import Path


# ── Physical constants for an arch dam (拱坝) ──
DAM_HEIGHT = 80.0          # m — typical medium arch dam
DAM_CREST_LENGTH = 360.0   # m — crest arc length
N_POSITIONS = 360          # 1° resolution along crest
N_FEATURES = 9

# Baseline sensor values at normal state
BASELINE = {
    'strain':       120.0,   # με — typical working strain in concrete arch dam
    'pore_pressure': 80.0,   # kPa — normal uplift pressure at monitoring points
    'displacement':   3.0,   # mm — normal radial displacement under hydrostatic load
    'settlement':     0.5,   # mm — negligible settlement in normal operation
    'acceleration':   0.5,   # gal — ambient microtremor
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
    sigma = width_deg / 2.355  # FWHM → sigma
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
    """
    rng = rng or np.random.RandomState()
    positions = np.linspace(0, DAM_CREST_LENGTH, N_POSITIONS)
    theta = positions / DAM_CREST_LENGTH * np.pi  # normalized arch angle

    # Arch action: higher loading at crown (center), lower at abutments
    arch_profile = 0.7 + 0.3 * np.sin(theta)  # [0.7, 1.0]

    data = np.zeros((N_POSITIONS, N_FEATURES))

    # Channels 0-2: Strain (3 spatial zones — left abutment, crown, right abutment)
    for i, center in enumerate([N_POSITIONS * 0.25, N_POSITIONS * 0.5, N_POSITIONS * 0.75]):
        zone_profile = 1.0 + 0.15 * np.sin(theta - np.pi * (center / N_POSITIONS - 0.5))
        data[:, i] = BASELINE['strain'] * arch_profile * zone_profile
        data[:, i] += _smooth_noise(rng, N_POSITIONS, scale=2.0, smoothness=5)

    # Channels 3-4: Pore pressure (upstream/downstream)
    data[:, 3] = BASELINE['pore_pressure'] * arch_profile  # upstream — higher
    data[:, 3] += _smooth_noise(rng, N_POSITIONS, scale=3.0, smoothness=8)
    data[:, 4] = BASELINE['pore_pressure'] * 0.6 * arch_profile  # downstream — lower
    data[:, 4] += _smooth_noise(rng, N_POSITIONS, scale=2.0, smoothness=8)

    # Channels 5-6: Displacement (radial, tangential)
    data[:, 5] = BASELINE['displacement'] * arch_profile  # radial — follows arch load
    data[:, 5] += _smooth_noise(rng, N_POSITIONS, scale=0.3, smoothness=6)
    data[:, 6] = BASELINE['displacement'] * 0.3  # tangential — small in symmetric loading
    data[:, 6] += _smooth_noise(rng, N_POSITIONS, scale=0.2, smoothness=6)

    # Channel 7: Settlement (near zero in normal state)
    data[:, 7] = BASELINE['settlement'] + _smooth_noise(rng, N_POSITIONS, scale=0.1, smoothness=10)

    # Channel 8: Acceleration (ambient microtremor)
    data[:, 8] = BASELINE['acceleration'] + rng.randn(N_POSITIONS) * 0.2

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
    positions = np.linspace(0, DAM_CREST_LENGTH, N_POSITIONS)

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
    positions, data, h_labels, s_labels, spans = generate_multi_hazard_dataset(n_samples=1300)
    print(f"Dataset: {data.shape}, hazards={h_labels.shape}, severities={s_labels.shape}")

    X, ys, yh, ysev, ysp = generate_sliding_window_dataset(
        positions, data, h_labels, s_labels, spans
    )
