"""
坝道微医 2.0 — 核心诊断模型
Spatial-Transformer + Inception-ResNet-LSTM + 多任务学习
用于多灾害（渗流、变形、沉降、地震）联合诊断与应变预测
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class InceptionStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=1),
            nn.BatchNorm1d(c), nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=1),
            nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm1d(c), nn.GELU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=1),
            nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, kernel_size=5, padding=2),
            nn.BatchNorm1d(c), nn.GELU()
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, c, kernel_size=1),
            nn.BatchNorm1d(c), nn.GELU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class SpatialTransformerModel(nn.Module):
    """Spatial-Transformer for structural health monitoring.

    1. Inception Stem: Multi-scale 1D conv feature extraction
    2. Position Encoding: Sinusoidal position encoding
    3. Transformer Encoder: Multi-head self-attention layers
    4. LSTM Bridge: Temporal/sequential feature refinement
    5. Prediction Head: Output future sensor sequence
    """
    def __init__(self, n_features, d_model=128, n_heads=8, n_layers=4,
                 d_ff=512, pred_horizon=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_horizon = pred_horizon
        self.n_features = n_features

        self.inception = InceptionStem(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_horizon * n_features),
        )

    def forward(self, x):
        B, L, F = x.shape
        x_conv = x.permute(0, 2, 1)
        x_conv = self.inception(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x_enc = self.pos_encoder(x_conv)
        for layer in self.encoder_layers:
            x_enc = layer(x_enc)
        lstm_out, _ = self.lstm(x_enc)
        pooled = lstm_out.mean(dim=1)
        pred = self.pred_head(pooled)
        return pred.view(B, self.pred_horizon, self.n_features)


class InceptionResNetLSTM(nn.Module):
    """Baseline model for comparison."""
    def __init__(self, n_features, d_model=128, pred_horizon=10, dropout=0.1):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.n_features = n_features
        self.inception1 = InceptionStem(n_features, 64)
        self.inception2 = InceptionStem(64, 128)
        self.inception3 = InceptionStem(128, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_horizon * n_features),
        )

    def forward(self, x):
        B, L, F = x.shape
        x_c = x.permute(0, 2, 1)
        x_c = self.inception1(x_c)
        x_c = self.inception2(x_c)
        x_c = self.inception3(x_c)
        x_c = x_c.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_c)
        pooled = lstm_out[:, -1, :]
        pred = self.pred_head(pooled)
        return pred.view(B, self.pred_horizon, self.n_features)


class MultiTaskDiagnosticModel(nn.Module):
    """Multi-task model for multi-hazard diagnostic.

    Task 1 (primary): Sensor sequence prediction
    Task 2 (auxiliary): Damage span regression
    Task 3 (auxiliary): Hazard type & severity classification
    """
    def __init__(self, n_features, d_model=128, n_heads=8, n_layers=4,
                 d_ff=512, pred_horizon=10, dropout=0.1, n_hazard_types=4,
                 n_severity_levels=4):
        super().__init__()
        self.d_model = d_model
        self.pred_horizon = pred_horizon

        self.inception = InceptionStem(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=dropout)

        # Task 1: Sensor prediction head
        self.sensor_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_horizon * n_features),
        )

        # Task 2: Damage span regression
        self.span_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Task 3: Hazard type classification
        self.hazard_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_hazard_types),
        )

        # Task 4: Severity classification
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_severity_levels),
        )

    def forward(self, x):
        B = x.shape[0]
        h = self.inception(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.pos_encoder(h)
        for layer in self.encoder_layers:
            h = layer(h)
        h, _ = self.lstm(h)
        h_pooled = h.mean(dim=1)

        sensor_pred = self.sensor_head(h_pooled)
        sensor_pred = sensor_pred.view(B, self.pred_horizon, -1)

        span_pred = self.span_head(h_pooled).squeeze(-1)
        hazard_logits = self.hazard_head(h_pooled)
        severity_logits = self.severity_head(h_pooled)

        return sensor_pred, span_pred, hazard_logits, severity_logits


class PhysicsInformedLoss(nn.Module):
    """MSE + smoothness + boundary + zone structure constraints."""
    def __init__(self, lambda_smooth=0.01, lambda_boundary=0.05, lambda_zone=0.02):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.lambda_zone = lambda_zone

    def forward(self, pred, target, input_seq=None):
        loss_mse = self.mse(pred, target)
        second_diff = pred[:, :-2] - 2 * pred[:, 1:-1] + pred[:, 2:]
        loss_smooth = torch.mean(second_diff ** 2)
        loss_boundary = 0.0
        if input_seq is not None:
            loss_boundary = torch.mean((pred[:, :1] - input_seq[:, -1:]) ** 2)
        edge_mean = (pred[:, :1] + pred[:, -1:]) / 2.0
        center_max = pred[:, 1:-1].max(dim=1).values.unsqueeze(-1)
        zone_violation = torch.clamp(center_max - edge_mean, min=0)
        loss_zone = torch.mean(zone_violation ** 2)
        return (loss_mse + self.lambda_smooth * loss_smooth +
                self.lambda_boundary * loss_boundary + self.lambda_zone * loss_zone)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with Kendall uncertainty weighting."""
    def __init__(self, lambda_smooth=0.01, lambda_boundary=0.05, lambda_zone=0.02,
                 learnable_weights=True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.lambda_zone = lambda_zone
        self.learnable_weights = learnable_weights

        if learnable_weights:
            self.log_sigma_sensor = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_span = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_hazard = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_severity = nn.Parameter(torch.tensor(0.0))

    def forward(self, outputs, targets, input_seq=None):
        pred_sensor, pred_span, pred_hazard, pred_severity = outputs
        true_sensor, true_span, true_hazard, true_severity = targets

        # Sensor loss with physics
        loss_sensor = self.mse(pred_sensor, true_sensor)
        second_diff = (pred_sensor[:, :-2] - 2 * pred_sensor[:, 1:-1] + pred_sensor[:, 2:])
        loss_smooth = torch.mean(second_diff ** 2)
        loss_boundary = 0.0
        if input_seq is not None:
            loss_boundary = torch.mean((pred_sensor[:, :1] - input_seq[:, -1:]) ** 2)
        edge_mean = (pred_sensor[:, :1] + pred_sensor[:, -1:]) / 2.0
        center_max = pred_sensor[:, 1:-1].max(dim=1).values.unsqueeze(-1)
        zone_violation = torch.clamp(center_max - edge_mean, min=0)
        loss_zone = torch.mean(zone_violation ** 2)
        loss_sensor_total = (loss_sensor + self.lambda_smooth * loss_smooth +
                             self.lambda_boundary * loss_boundary + self.lambda_zone * loss_zone)

        loss_span = self.mse(pred_span, true_span)
        loss_hazard = self.ce(pred_hazard, true_hazard)
        loss_severity = self.ce(pred_severity, true_severity)

        if self.learnable_weights:
            sigma_s = torch.exp(self.log_sigma_sensor)
            sigma_sp = torch.exp(self.log_sigma_span)
            sigma_h = torch.exp(self.log_sigma_hazard)
            sigma_sev = torch.exp(self.log_sigma_severity)
            total = (loss_sensor_total / (2 * sigma_s**2) + self.log_sigma_sensor +
                     loss_span / (2 * sigma_sp**2) + self.log_sigma_span +
                     loss_hazard / (2 * sigma_h**2) + self.log_sigma_hazard +
                     loss_severity / (2 * sigma_sev**2) + self.log_sigma_severity)
        else:
            total = loss_sensor_total + 0.3 * loss_span + 0.2 * loss_hazard + 0.2 * loss_severity
        return total


def create_model(model_type='transformer', **kwargs):
    default_kwargs = {
        'n_features': 9, 'd_model': 128, 'n_heads': 8, 'n_layers': 4,
        'd_ff': 512, 'pred_horizon': 10, 'dropout': 0.1,
    }
    default_kwargs.update(kwargs)

    if model_type == 'transformer':
        return SpatialTransformerModel(
            n_features=default_kwargs['n_features'],
            d_model=default_kwargs['d_model'],
            n_heads=default_kwargs['n_heads'],
            n_layers=default_kwargs['n_layers'],
            d_ff=default_kwargs['d_ff'],
            pred_horizon=default_kwargs['pred_horizon'],
            dropout=default_kwargs['dropout'],
        )
    elif model_type == 'inception_resnet_lstm':
        return InceptionResNetLSTM(
            n_features=default_kwargs['n_features'],
            d_model=default_kwargs['d_model'],
            pred_horizon=default_kwargs['pred_horizon'],
            dropout=default_kwargs['dropout'],
        )
    elif model_type == 'multitask':
        return MultiTaskDiagnosticModel(
            n_features=default_kwargs['n_features'],
            d_model=default_kwargs['d_model'],
            n_heads=default_kwargs['n_heads'],
            n_layers=default_kwargs['n_layers'],
            d_ff=default_kwargs['d_ff'],
            pred_horizon=default_kwargs['pred_horizon'],
            dropout=default_kwargs['dropout'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    for mtype in ['transformer', 'inception_resnet_lstm', 'multitask']:
        model = create_model(mtype, n_features=9, pred_horizon=10)
        n_params = count_parameters(model)
        print(f"{mtype}: {n_params:,} parameters")
        x = torch.randn(4, 30, 9)
        if mtype == 'multitask':
            s, sp, h, sev = model(x)
            print(f"  Input: {x.shape} -> Sensor:{s.shape}, Span:{sp.shape}, Hazard:{h.shape}, Severity:{sev.shape}")
        else:
            y = model(x)
            print(f"  Input: {x.shape} -> Output: {y.shape}")
