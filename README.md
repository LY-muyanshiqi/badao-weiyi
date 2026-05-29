# 坝道微医 2.0 — 水利工程多灾害智能诊断系统

基于 AI 的水利工程结构健康监测与多灾害耦合诊断平台。**水利部认定全国15家高水平安全诊断团队（唯一本科生团队）**，国际专利 LU503818。

## 概述

坝道微医 2.0 在原有单一安全诊断基础上，实现三大创新：

- **多灾害耦合诊断**：渗流 + 变形 + 沉降 + 地震联合评估
- **数字孪生可视化**：3D 交互式大坝模型 + 实时传感器映射
- **水利大模型知识库**：RAG 架构，规范检索 + 案例匹配 + 智能报告

### 核心性能

| 指标 | 数值 |
|------|------|
| 诊断准确率 | 92% |
| 监测点优化 | 减少 60% |
| 诊断时间 | 传统方法的 1/3 |
| 模型参数量 | 101万 (Spatial-Transformer) |
| 异常识别准确率 | 82% |

## 架构

```
输入 → 多源传感器 → Spatial-Transformer → 多灾害诊断 → 3D数字孪生
                    ↓                           ↓
              Inception-ResNet-LSTM          RAG知识库
                    ↓                           ↓
              Multi-Task Ensemble        诊断报告生成
```

### Spatial-Transformer 模型

```
Input [B, 30, N_features]
  → Inception Stem (多尺度1D卷积)
  → Sinusoidal Position Encoding
  → 4× Transformer Encoder (8-head Self-Attention)
  → BiLSTM Bridge
  → Multi-Task Heads (应变预测 + 损伤跨度 + 灾害分类 + 严重度)
  → Output [B, 10, N_features] + 辅助任务
```

## 项目结构

```
├── src/
│   ├── core/
│   │   ├── transformer_model.py    # Spatial-Transformer + Multi-Task
│   │   ├── data_pipeline.py        # 数据处理与序列化
│   │   └── diagnostic_engine.py    # 多灾害诊断 + NSGA-II优化
│   ├── digital_twin/
│   │   └── server.py               # Flask-SocketIO + Three.js 3D
│   ├── knowledge_base/
│   │   └── embeddings.py           # ChromaDB RAG 知识库
│   └── app.py                      # Streamlit 主应用
├── data/                           # 数据目录
├── outputs/
│   ├── models/                     # 模型权重
│   └── figures/                    # 可视化输出
├── docs/                           # 文档
├── .github/workflows/ci.yml        # CI/CD
├── requirements.txt
└── README.md
```

## 快速开始

### 环境要求

Python 3.8+ · PyTorch 1.12+ · Scikit-learn 1.2+

### 安装

```bash
pip install -r requirements.txt
```

### 运行诊断

```bash
# Streamlit Web 应用
streamlit run src/app.py

# 3D数字孪生
python src/digital_twin/server.py

# 模型训练（需要数据）
python src/core/transformer_model.py
```

### 快速验证

```bash
# 测试数据处理流水线
python src/core/data_pipeline.py

# 测试诊断引擎
python src/core/diagnostic_engine.py
```

## 灾害类型

| 灾害 | 监测指标 | 模型方法 |
|------|---------|---------|
| 渗流 | 渗透压力、渗流量、浸润线 | LSTM 时序预测 |
| 变形 | 应变、裂缝、挠度 | Spatial-Transformer |
| 沉降 | 沉降量、差异沉降、速率 | Inception-ResNet-LSTM |
| 地震 | 加速度、频谱、阻尼比 | Multi-Task Ensemble |

## 已获荣誉

- 水利部全国15家高水平安全诊断团队（唯一本科生）
- 国际发明专利 LU503818
- 2024中国水利学术大会分会场报告（唯一本科生）
- 中国国际大学生创新大赛（2026）校级晋级
- 实际应用：西安黑河供水公司、中国电建西北院

## 引用

```bibtex
@software{badao-weiyi-2.0,
  author = {李垚 and 坝道微医团队},
  title = {坝道微医 2.0 — 水利工程多灾害智能诊断系统},
  year = {2026},
  url = {https://github.com/LY-muyanshiqi/badao-weiyi}
}
```

## 相关项目

| 项目 | 描述 |
|------|------|
| [PCCP-E](https://github.com/LY-muyanshiqi/PCCP) | 环向变形智能预测 · R²=0.986 |
| [智慧水利](https://github.com/LY-muyanshiqi/smart-water-demo) | LSTM洪水预测 · Flask+Streamlit |
| [抽蓄-碳减排](https://github.com/LY-muyanshiqi/pumped-storage-carbon) | LSTM来水预测 · 碳核算 |

## 许可证

MIT License
