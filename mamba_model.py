import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


class MambaLayer(nn.Module):
    """簡素化されたMambaレイヤー実装"""
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        
        # 入力投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D畳み込み
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )
        
        # 状態空間パラメータ
        self.state_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.A = nn.Parameter(torch.randn(self.d_state))
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 入力を分割（メイン処理用とresidual用）
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # 1D畳み込み
        x_conv = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        
        # SiLU活性化
        x_conv = F.silu(x_conv)
        
        # 簡素化された状態空間モデル
        # 各時刻で独立してプロジェクション
        B = self.B_proj(x_conv)  # (B, L, d_state)
        C = self.C_proj(x_conv)  # (B, L, d_state)
        
        # 状態の更新（簡素化版）
        A_diag = -F.softplus(self.A)  # 安定性のため負の値
        y = torch.zeros_like(x_conv)
        
        for i in range(seq_len):
            if i == 0:
                state = B[:, i]  # 初期状態
            else:
                state = state * torch.exp(A_diag) + B[:, i]
            
            # 出力計算
            output = (state * C[:, i]).sum(dim=-1)  # (B,)
            y[:, i] = output.unsqueeze(-1).expand(-1, self.d_inner)  # (B, d_inner)
        
        # Residual connection
        y = y + x_conv * self.D
        
        # Gated connection
        y = y * F.silu(res)
        
        # 出力投影
        y = self.out_proj(y)
        
        return y


class MambaModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, d_state: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaLayer(d_model, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return {"logits": logits, "loss": loss}
    
    def generate(self, input_ids, max_length=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids