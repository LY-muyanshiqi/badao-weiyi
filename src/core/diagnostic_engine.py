"""
多灾害耦合诊断引擎 — 渗流 + 变形 + 沉降 + 地震联合评估
基于 NSGA-II 多目标优化的最优监测方案 + 综合风险等级输出
"""
import numpy as np
import torch
from pathlib import Path

# 灾害类型定义
HAZARD_DEFINITIONS = {
    'seepage': {
        'name': '渗流灾害',
        'indicators': ['渗透压力', '渗流量', '浸润线高度'],
        'risk_levels': ['正常', '轻微渗漏', '集中渗漏', '管涌'],
        'threshold_ratio': [0.0, 0.3, 0.6, 0.85],
    },
    'deformation': {
        'name': '变形灾害',
        'indicators': ['应变', '裂缝宽度', '挠度'],
        'risk_levels': ['正常', '微变形', '显著变形', '结构失稳'],
        'threshold_ratio': [0.0, 0.25, 0.55, 0.8],
    },
    'settlement': {
        'name': '沉降灾害',
        'indicators': ['沉降量', '差异沉降', '沉降速率'],
        'risk_levels': ['正常', '轻微沉降', '不均匀沉降', '过量沉降'],
        'threshold_ratio': [0.0, 0.2, 0.5, 0.75],
    },
    'seismic': {
        'name': '地震灾害',
        'indicators': ['加速度峰值', '频谱特征', '阻尼比'],
        'risk_levels': ['正常', '微震', '中震', '强震'],
        'threshold_ratio': [0.0, 0.15, 0.4, 0.7],
    },
}

OVERALL_RISK_LEVELS = ['安全', '注意', '警告', '危险']


class DiagnosticEngine:
    """Multi-hazard coupled diagnostic engine."""

    def __init__(self, model_path=None, scaler_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.hazard_history = []

        if model_path and scaler_path:
            self.load_model(model_path, scaler_path)

    def load_model(self, model_path, scaler_path):
        from src.core.transformer_model import create_model
        import pickle
        self.model = create_model('multitask', n_features=9)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def diagnose(self, sensor_data, positions=None):
        """Run multi-hazard diagnosis on sensor data.

        Args:
            sensor_data: np.array [n_positions, n_features]
            positions: optional position array

        Returns:
            dict with per-hazard risk levels and overall assessment
        """
        # Window-based inference
        from src.core.data_pipeline import build_sliding_windows, normalize_data

        if positions is None:
            positions = np.arange(len(sensor_data))
        X, y, meta = build_sliding_windows(sensor_data, positions)

        if self.scaler:
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat).reshape(X.shape)
        else:
            X_scaled = X

        # Model inference
        if self.model:
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                sensor_pred, span_pred, hazard_logits, severity_logits = self.model(X_t)

            hazard_probs = torch.softmax(hazard_logits, dim=-1).cpu().numpy()
            severity_probs = torch.softmax(severity_logits, dim=-1).cpu().numpy()
        else:
            # Rule-based fallback when no trained model
            hazard_probs, severity_probs = self._rule_based_diagnosis(meta)

        # Aggregate results
        results = self._aggregate_diagnosis(meta, hazard_probs, severity_probs)

        self.hazard_history.append(results)
        return results

    def _rule_based_diagnosis(self, meta):
        """Rule-based diagnosis fallback using damage span thresholds."""
        n = len(meta)
        # hazard: [seepage, deformation, settlement, seismic]
        hazard_probs = np.zeros((n, 4))
        severity_probs = np.zeros((n, 4))

        for i, m in enumerate(meta):
            span = m.get('rz_span', 0)
            if span < 5:
                sev = 0
            elif span < 30:
                sev = 1
            elif span < 100:
                sev = 2
            else:
                sev = 3

            # When no data, assign most weight to deformation (most common hazard)
            hazard_probs[i] = [0.1, 0.6, 0.2, 0.1]
            severity_probs[i, sev] = 1.0

        return hazard_probs, severity_probs

    def _aggregate_diagnosis(self, meta, hazard_probs, severity_probs):
        """Aggregate window-level predictions into overall diagnosis."""
        conditions = [m['condition'] for m in meta]

        # Mean hazard distribution
        mean_hazard = hazard_probs.mean(axis=0)
        dominant_hazard_idx = int(np.argmax(mean_hazard))
        hazard_names = ['seepage', 'deformation', 'settlement', 'seismic']
        dominant_hazard = hazard_names[dominant_hazard_idx]

        # Mean severity (expected value of severity level)
        severity_levels = np.arange(4)
        mean_severity_dist = severity_probs.mean(axis=0)
        expected_severity = float(np.dot(mean_severity_dist, severity_levels))
        overall_severity_idx = int(np.argmax(mean_severity_dist))

        # Risk scores per hazard = hazard probability * overall severity
        risk_scores = {}
        for idx, hazard_key in enumerate(hazard_names):
            hazard_prob = float(mean_hazard[idx])
            risk_scores[hazard_key] = {
                'name': HAZARD_DEFINITIONS[hazard_key]['name'],
                'risk_score': hazard_prob * expected_severity,
                'risk_level': HAZARD_DEFINITIONS[hazard_key]['risk_levels'][
                    int(np.clip(round(hazard_prob * expected_severity), 0, 3))
                ],
            }

        # Overall risk
        overall_risk_score = expected_severity * float(mean_hazard.max())
        overall_risk_score = np.clip(overall_risk_score, 0, 3)
        overall_risk_level = OVERALL_RISK_LEVELS[int(np.clip(round(overall_risk_score), 0, 3))]

        # Condition distribution
        condition_dist = {}
        for cond in ['normal', 'mild_damage', 'moderate_damage', 'severe_damage']:
            cnt = conditions.count(cond)
            condition_dist[cond] = cnt / max(len(conditions), 1)

        return {
            'overall_risk_score': float(overall_risk_score),
            'overall_risk_level': overall_risk_level,
            'dominant_hazard': dominant_hazard,
            'per_hazard_scores': risk_scores,
            'condition_distribution': condition_dist,
            'n_windows': len(meta),
            'mean_damage_span': float(np.mean([m.get('rz_span', 0) for m in meta])),
        }

    def optimize_sensor_placement(self, sensor_data, n_sensors_budget=5, n_population=50, n_generations=100):
        """NSGA-II multi-objective sensor placement optimization.

        Objectives:
          1. Maximize detection coverage (minimize max distance to nearest sensor)
          2. Minimize redundancy (correlation between adjacent sensors)

        Returns:
            optimal_indices: best sensor positions
            pareto_front: all non-dominated solutions
        """
        n_positions = sensor_data.shape[0]
        n_features = sensor_data.shape[1]

        # Initialize population
        rng = np.random.RandomState(42)
        population = []
        for _ in range(n_population):
            individual = sorted(rng.choice(n_positions, size=n_sensors_budget, replace=False).tolist())
            population.append(individual)

        for gen in range(n_generations):
            # Evaluate objectives
            fitness = []
            for ind in population:
                # Objective 1: Coverage (max gap between sensors)
                gaps = [ind[i+1] - ind[i] for i in range(len(ind)-1)]
                max_gap = max(gaps) if gaps else n_positions
                f1 = max_gap

                # Objective 2: Redundancy (mean correlation)
                correlations = []
                for idx in ind:
                    for jdx in ind:
                        if idx < jdx:
                            corr = np.corrcoef(sensor_data[idx], sensor_data[jdx])[0, 1]
                            correlations.append(abs(corr))
                f2 = np.mean(correlations) if correlations else 0

                fitness.append((f1, f2))

            # Fast non-dominated sorting
            fronts = self._non_dominated_sort(fitness)
            pareto_front = fronts[0]

            # Selection, crossover, mutation
            new_pop = []
            for _ in range(n_population // 2):
                p1_idx = rng.choice(pareto_front)
                p2_idx = rng.choice(pareto_front)
                p1 = population[p1_idx]
                p2 = population[p2_idx]

                # Crossover
                all_pos = sorted(set(p1 + p2))
                c1 = sorted(rng.choice(all_pos, size=n_sensors_budget, replace=False).tolist())
                c2 = sorted(rng.choice(all_pos, size=n_sensors_budget, replace=False).tolist())

                # Mutation
                if rng.random() < 0.2:
                    c1[rng.randint(0, n_sensors_budget)] = rng.randint(0, n_positions - 1)
                if rng.random() < 0.2:
                    c2[rng.randint(0, n_sensors_budget)] = rng.randint(0, n_positions - 1)

                c1 = sorted(set(c1))
                c2 = sorted(set(c2))
                while len(c1) < n_sensors_budget:
                    v = rng.randint(0, n_positions - 1)
                    if v not in c1:
                        c1.append(v)
                while len(c2) < n_sensors_budget:
                    v = rng.randint(0, n_positions - 1)
                    if v not in c2:
                        c2.append(v)

                new_pop.extend([sorted(c1), sorted(c2)])

            population = new_pop[:n_population] + population[:n_population // 4]

        # Final evaluation and selection
        final_fitness = []
        for ind in population:
            gaps = [ind[i+1] - ind[i] for i in range(len(ind)-1)]
            f1 = max(gaps) if gaps else n_positions
            correlations = []
            for idx in ind:
                for jdx in ind:
                    if idx < jdx:
                        corr = abs(np.corrcoef(sensor_data[idx], sensor_data[jdx])[0, 1])
                        correlations.append(corr)
            f2 = np.mean(correlations) if correlations else 0
            final_fitness.append((f1, f2))

        fronts = self._non_dominated_sort(final_fitness)
        pareto_front = fronts[0]

        # Select best from Pareto front (knee point: closest to origin in normalized space)
        f1_vals = np.array([final_fitness[i][0] for i in pareto_front])
        f2_vals = np.array([final_fitness[i][1] for i in pareto_front])
        f1_norm = f1_vals / (f1_vals.max() + 1e-10)
        f2_norm = f2_vals / (f2_vals.max() + 1e-10)
        distances = np.sqrt(f1_norm**2 + f2_norm**2)
        best_pareto_idx = pareto_front[np.argmin(distances)]

        return {
            'optimal_indices': population[best_pareto_idx],
            'n_pareto_solutions': len(pareto_front),
            'coverage_gap': float(f1_vals[np.argmin(distances)]),
            'redundancy': float(f2_vals[np.argmin(distances)]),
        }

    def _non_dominated_sort(self, fitness):
        """Fast non-dominated sorting for NSGA-II."""
        n = len(fitness)
        domination_counts = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                fi1, fi2 = fitness[i]
                fj1, fj2 = fitness[j]
                if fi1 <= fj1 and fi2 <= fj2 and (fi1 < fj1 or fi2 < fj2):
                    dominated_by[i].append(j)
                elif fj1 <= fi1 and fj2 <= fi2 and (fj1 < fi1 or fj2 < fi2):
                    domination_counts[i] += 1

        fronts = [[]]
        for i in range(n):
            if domination_counts[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.data_pipeline import generate_sample_data
    positions, sensor_data = generate_sample_data()

    engine = DiagnosticEngine()
    results = engine.diagnose(sensor_data, positions)

    print(f"Overall Risk: {results['overall_risk_level']} (score={results['overall_risk_score']:.2f})")
    print(f"Dominant Hazard: {results['dominant_hazard']}")
    for hazard, info in results['per_hazard_scores'].items():
        print(f"  {info['name']}: {info['risk_level']} (score={info['risk_score']:.2f})")
    print(f"Damage Span: {results['mean_damage_span']:.1f}°")

    # Optimize sensor placement
    opt = engine.optimize_sensor_placement(sensor_data, n_sensors_budget=5)
    print(f"\nOptimal sensor positions: {opt['optimal_indices']}")
    print(f"Coverage gap: {opt['coverage_gap']:.1f}°, Redundancy: {opt['redundancy']:.3f}")
