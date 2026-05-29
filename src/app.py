"""
坝道微医 2.0 — Streamlit 主应用入口
集成：模型推理 + 多灾害诊断 + 数字孪生 + 知识库报告生成
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="坝道微医 2.0 — 水利工程智能诊断",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("坝道微医 2.0 — 水利工程多灾害智能诊断系统")
st.caption("水利部认定 | 国际专利 LU503818 | National-Level Safety Diagnosis Team")

# Sidebar
with st.sidebar:
    st.header("控制面板")

    mode = st.radio(
        "运行模式",
        ["演示模式 (示例数据)", "上传数据模式", "实时监测模式"],
        index=0,
    )

    st.divider()

    model_type = st.selectbox(
        "诊断模型",
        ["Spatial-Transformer (推荐)", "Inception-ResNet-LSTM", "Multi-Task Ensemble"],
        index=0,
    )

    st.divider()

    st.subheader("灾害类型")
    hazards = {
        'seepage': st.checkbox('渗流灾害', value=True),
        'deformation': st.checkbox('变形灾害', value=True),
        'settlement': st.checkbox('沉降灾害', value=True),
        'seismic': st.checkbox('地震灾害', value=False),
    }

    st.divider()

    st.subheader("传感器配置")
    n_sensors = st.slider("传感器数量", 5, 50, 9)
    seq_length = st.slider("输入序列长度", 10, 60, 30)
    pred_horizon = st.slider("预测范围", 5, 30, 10)

    st.divider()
    run_diagnosis = st.button("开始诊断", type="primary", use_container_width=True)


# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("结构健康状态")
    status_placeholder = st.empty()

with col2:
    st.subheader("风险指标")
    metrics_placeholder = st.empty()

# Results area
results_tab1, results_tab2, results_tab3 = st.tabs([
    "诊断报告", "传感器数据", "数字孪生"
])


def generate_demo_data():
    """Generate demo sensor data."""
    rng = np.random.RandomState(int(time.time() / 10))
    n_positions = 360
    positions = np.linspace(0, 360, n_positions)
    n_features = n_sensors

    strain_base = 100 + 15 * np.sin(positions * np.pi / 180)
    damage_zone = np.ones(n_positions)
    center = rng.uniform(120, 240)
    sigma = rng.uniform(25, 60)
    severity = rng.uniform(0.15, 0.55)
    damage_zone = 1 - severity * np.exp(-((positions - center) ** 2) / (2 * sigma ** 2))

    sensor_matrix = np.zeros((n_positions, n_features))
    for i in range(n_features):
        noise = rng.randn(n_positions) * 2
        sensor_matrix[:, i] = strain_base + noise + (i * 3) * damage_zone

    return positions, sensor_matrix


def run_diagnostic_pipeline(positions, sensor_data):
    """Run the full diagnostic pipeline."""
    from src.core.data_pipeline import build_sliding_windows
    from src.core.diagnostic_engine import DiagnosticEngine
    from src.knowledge_base.embeddings import StandardKnowledgeBase, generate_diagnosis_report

    # Build data windows
    X, y, meta = build_sliding_windows(sensor_data, positions, seq_length, pred_horizon)

    # Initialize diagnostic engine
    engine = DiagnosticEngine()

    # Run diagnosis
    results = engine.diagnose(sensor_data, positions)

    # Optimize sensor placement
    opt_results = engine.optimize_sensor_placement(sensor_data, n_sensors_budget=min(9, n_sensors))

    # Generate report
    kb = StandardKnowledgeBase()
    try:
        kb.load_standards()
    except Exception:
        pass
    report = generate_diagnosis_report(results, kb)

    return results, report, opt_results, meta


if run_diagnosis:
    with st.spinner("正在执行多灾害耦合诊断..."):
        if mode == "演示模式 (示例数据)":
            positions, sensor_data = generate_demo_data()
        elif mode == "上传数据模式":
            uploaded = st.file_uploader("上传传感器数据 (CSV/Excel)", type=['csv', 'xlsx'])
            if uploaded:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                positions = df.iloc[:, 0].values
                sensor_data = df.iloc[:, 1:].values
            else:
                st.warning("请上传数据文件")
                st.stop()
        else:
            st.info("实时监测模式已启动，等待数据...")
            positions, sensor_data = generate_demo_data()

        results, report, opt_results, meta = run_diagnostic_pipeline(positions, sensor_data)

        # Update status display
        risk_colors = {'安全': '#4caf50', '注意': '#ff9800', '警告': '#f44336', '危险': '#b71c1c'}
        status_color = risk_colors.get(results['overall_risk_level'], '#78909c')

        with col1:
            st.markdown(f"""
            <div style="background:#1a1a2e; border-radius:16px; padding:24px; text-align:center;">
                <div style="font-size:14px; color:#78909c; margin-bottom:8px;">整体风险等级</div>
                <div style="font-size:48px; font-weight:bold; color:{status_color}; margin:16px 0;">
                    {results['overall_risk_level']}
                </div>
                <div style="font-size:14px; color:#78909c;">
                    主要灾害类型：{report['dominant_hazard']} | 测点：{n_sensors} | 窗口：{results['n_windows']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            cols = st.columns(4)
            hazard_keys = list(results['per_hazard_scores'].keys())
            risk_labels = ['安全', '注意', '警告', '危险']
            for i, col in enumerate(cols):
                if i < len(hazard_keys):
                    hk = hazard_keys[i]
                    info = results['per_hazard_scores'][hk]
                    h_colors = {'正常': '#4caf50', '轻微渗漏': '#ff9800', '集中渗漏': '#f44336', '管涌': '#b71c1c',
                                '微变形': '#4caf50', '显著变形': '#ff9800', '结构失稳': '#b71c1c',
                                '轻微沉降': '#4caf50', '不均匀沉降': '#ff9800', '过量沉降': '#b71c1c',
                                '微震': '#4caf50', '中震': '#ff9800', '强震': '#b71c1c'}
                    h_color = h_colors.get(info['risk_level'], '#78909c')
                    col.metric(
                        label=info['name'][:4],
                        value=info['risk_level'],
                        delta=f"风险分 {info['risk_score']:.1f}",
                    )

        # Diagnosis report tab
        with results_tab1:
            st.markdown(f"""
            ### 诊断摘要
            {report['summary']}

            ### 处置建议
            """)
            for i, rec in enumerate(report['recommendations'], 1):
                st.markdown(f"{i}. {rec}")

            st.markdown("### 规范依据")
            for ref in report.get('references', []):
                st.info(f"**{ref['source']}** (相关性: {ref.get('relevance', 'N/A')})\n\n{ref['content'][:200]}...")

            st.markdown("### 传感器优化布置")
            st.write(f"Pareto前沿解数量: {opt_results['n_pareto_solutions']}")
            st.write(f"最优覆盖间隙: {opt_results['coverage_gap']:.1f}°")
            st.write(f"冗余度: {opt_results['redundancy']:.3f}")

        # Sensor data tab
        with results_tab2:
            st.subheader("传感器数据预览")
            df = pd.DataFrame(sensor_data, columns=[f'S{i+1}' for i in range(sensor_data.shape[1])])
            df.insert(0, '位置', positions)
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("数据统计")
            st.line_chart(df.set_index('位置').iloc[:, :min(5, sensor_data.shape[1])])

            st.subheader("损伤等级分布")
            cond_dist = results.get('condition_distribution', {})
            cond_df = pd.DataFrame({
                '等级': list(cond_dist.keys()),
                '占比': list(cond_dist.values()),
            })
            st.bar_chart(cond_df.set_index('等级'))

        # Digital twin tab
        with results_tab3:
            st.markdown(f"""
            <div style="background:#0a1628; border-radius:12px; padding:20px; text-align:center;">
                <p style="color:#4fc3f7;">数字孪生可视化需启动独立服务</p>
                <code style="color:#78909c;">python src/digital_twin/server.py</code>
                <p style="color:#78909c; margin-top:8px;">启动后访问 http://localhost:5000 查看3D大坝模型</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 当前风险热力图")
            # Simple heatmap
            if sensor_data.shape[1] > 1:
                st.write(f"损伤区域: {results.get('mean_damage_span', 0):.1f}° span")
                st.write(f"整体风险评分: {results.get('overall_risk_score', 0):.2f}/3.00")

else:
    with col1:
        st.markdown("""
        <div style="background:#1a1a2e; border-radius:16px; padding:24px; text-align:center;">
            <div style="font-size:14px; color:#78909c; margin-bottom:8px;">系统就绪</div>
            <div style="font-size:28px; color:#4fc3f7; margin:16px 0;">等待诊断指令</div>
            <div style="font-size:14px; color:#78909c;">请点击左侧 "开始诊断" 按钮</div>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("坝道微医 2.0 | 西安理工大学 | 李垚")
st.sidebar.caption("[GitHub](https://github.com/LY-muyanshiqi/badao-weiyi)")
