# 坝道微医 2.0 — 水利工程多灾害智能诊断系统

[![Status](https://img.shields.io/badge/赛事-中国国际大学生创新大赛2026-0891b2?style=flat-square)](https://github.com/LY-muyanshiqi)
[![Patent](https://img.shields.io/badge/国际专利-LU503818-green?style=flat-square)](https://github.com/LY-muyanshiqi)

> 水利部官方认定全国 15 家高水平安全诊断团队之一（**唯一本科生团队**）· 校级晋级

基于 AI 的水利工程结构健康监测与多灾害耦合诊断平台。国际专利 LU503818。

## 概述

坝道微医 2.0 在原有单一安全诊断基础上，实现三大创新：

- **多灾害耦合诊断**：渗流 + 变形 + 沉降 + 地震联合评估
- **数字孪生可视化**：3D 交互式大坝模型 + 实时传感器映射
- **水利大模型知识库**：RAG 架构，规范检索 + 案例匹配 + 智能报告

### 核心性能

| 指标 | 数值 |
|------|------|
| 诊断准确率 | 92%（国内最高） |
| 数据库规模 | 310 座小型水利工程 |
| AI 算法库 | 18 个优化算法 |
| 监测点优化 | 减少 60%（时空聚类） |
| 诊断时间 | 传统方法的 1/3 |
| 模型参数量 | 101万 (Spatial-Transformer) |
| 异常识别准确率 | 82% |
| 监控指标预警率 | 92% |

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
python src/core/data_pipeline.py
python src/core/diagnostic_engine.py
```

## 灾害类型

| 灾害 | 监测指标 | 模型方法 |
|------|---------|---------|
| 渗流 | 渗透压力、渗流量、浸润线 | LSTM 时序预测 |
| 变形 | 应变、裂缝、挠度 | Spatial-Transformer |
| 沉降 | 沉降量、差异沉降、速率 | Inception-ResNet-LSTM |
| 地震 | 加速度、频谱、阻尼比 | Multi-Task Ensemble |

## 关键里程碑

- **水利部官方认定**：全国 15 家高水平安全诊断团队（唯一本科生）
- **水利部副部长朱程清**高度肯定（2024.11 调研考察）
- **唯一受邀本科生**在 2024 中国水利学术大会作分会场报告
- **国际发明专利 1 项**（授权号 LU503818）
- **应用落地**：西安黑河供水公司 · 中国电建西北院
- 中国国际大学生创新大赛（2026）校级现场赛通过 · 进入下一轮

## 项目信息

- **赛事**：中国国际大学生创新大赛（2026）· 高教主赛道本科生创意组
- **项目负责人**：刘昱玚
- **成员**：李垚 等

## 相关项目

| 项目 | 描述 |
|------|------|
| [PCCP-E](https://github.com/LY-muyanshiqi/PCCP) | 环向变形智能预测 · R²=0.986 |
| [华中杯-VRP](https://github.com/LY-muyanshiqi/huazhong-cup-vrp) | Hybrid-ILS · 278页论文 |
| [统计建模-玉米](https://github.com/LY-muyanshiqi/statistical-modeling-corn) | LSTM+CNN+XGBoost · R²=0.82 |
| [抽蓄-碳减排](https://github.com/LY-muyanshiqi/pumped-storage-carbon) | LSTM来水预测 · 碳核算 |
| [智慧水利](https://github.com/LY-muyanshiqi/smart-water-demo) | LSTM洪水预测 · Flask+Streamlit |
| [个人主页](https://github.com/LY-muyanshiqi/LY-muyanshiqi) | GitHub Profile · 项目总览 |

## 许可证

MIT License
