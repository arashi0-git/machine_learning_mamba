import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List


class MoEGate(nn.Module):
    """改良されたMixture of Experts ゲートネットワーク"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, noise_epsilon: float = 1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        
        # 複数層のゲートネットワーク
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts, bias=False)
        )
        
        # エキスパート使用量のバランス調整
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (B*L, d_model)
        
        gate_logits = self.gate(x_flat)  # (B*L, num_experts)
        
        # 訓練時にノイズを追加してバランスを促進
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
        
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Top-k選択
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # エキスパート使用量の追跡（訓練時）
        if self.training:
            unique_experts, counts = torch.unique(top_k_indices, return_counts=True)
            for expert_idx, count in zip(unique_experts, counts):
                self.expert_usage[expert_idx] += count.float()
        
        # 負荷バランス損失の計算
        gate_mean = gate_scores.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(gate_mean * torch.log(gate_mean + 1e-8))
        
        return top_k_scores, top_k_indices, balance_loss


class Expert(nn.Module):
    """改良された個別のエキスパートネットワーク"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'swiglu'):
        super().__init__()
        self.activation = activation
        
        if activation == 'swiglu':
            # SwiGLU活性化を使用した改良版
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_model, bias=False)
        else:
            # 従来のGELU版
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        if self.activation == 'swiglu':
            # SwiGLU: x -> w1(x) * silu(w2(x)) -> w3
            return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))
        else:
            # 従来版
            return self.w2(self.dropout(F.gelu(self.w1(x))))


class MoELayer(nn.Module):
    """改良された Mixture of Experts レイヤー"""
    def __init__(self, d_model: int, num_experts: int, d_ff: int, top_k: int = 2, 
                 dropout: float = 0.1, expert_activation: str = 'swiglu'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        self.gate = MoEGate(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout, expert_activation) for _ in range(num_experts)
        ])
        
        # 各エキスパートへの効率的なルーティング用のマスク
        self.register_buffer('routing_mask', torch.eye(num_experts))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (B*L, d_model)
        
        gate_scores, gate_indices, balance_loss = self.gate(x)  # (B*L, top_k), (B*L, top_k), scalar
        
        # 効率的なエキスパート処理
        final_output = torch.zeros_like(x_flat)
        expert_outputs = torch.zeros(x_flat.size(0), self.num_experts, d_model, device=x.device)
        
        # 各エキスパートを必要な時だけ実行
        for expert_idx in range(self.num_experts):
            # このエキスパートが選択されているトークンのマスク
            expert_mask = (gate_indices == expert_idx).any(dim=1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                expert_outputs[expert_mask, expert_idx] = expert_output
        
        # Top-kエキスパートの出力を重み付き合成
        for i in range(self.top_k):
            expert_idx = gate_indices[:, i]  # (B*L,)
            expert_weight = gate_scores[:, i].unsqueeze(-1)  # (B*L, 1)
            
            # 各サンプルに対応するエキスパートの出力を選択
            selected_expert_output = expert_outputs[torch.arange(x_flat.size(0)), expert_idx]
            final_output += expert_weight * selected_expert_output
        
        output = final_output.view(batch_size, seq_len, d_model)
        
        return output, balance_loss


class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        
        # 事前計算された位置エンコーディング
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(-2)
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb

def apply_rope(x, cos, sin):
    """RoPEを適用"""
    def rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)
    
    return x * cos + rotate_half(x) * sin

class MultiHeadAttention(nn.Module):
    """改良されたマルチヘッドアテンション"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rope: bool = True,
                 use_flash_attn: bool = False, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn
        
        # より効率的な投影
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        if use_rope:
            self.rope = RoPEEmbedding(self.d_head, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
        # アテンション重みの学習可能な調整
        self.attn_scale = nn.Parameter(torch.ones(n_heads))
        
    def forward(self, x, mask=None, kv_cache=None):
        batch_size, seq_len, d_model = x.shape
        
        # QKVを一度に計算
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # ヘッドに分割
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # RoPE適用
        if self.use_rope:
            cos, sin = self.rope(x, seq_len)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        
        # スケーリング
        q = q * self.scale
        
        # アテンションスコア計算
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # ヘッドごとのスケーリング
        scores = scores * self.attn_scale.view(1, -1, 1, 1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # アテンション適用
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(out)


class DropPath(nn.Module):
    """確率的深度（DropPath）実装 - 精度向上のため"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class RMSNorm(nn.Module):
    """RMSNorm - LayerNormより効率的で精度向上に寄与"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class AdaptiveLossWeighting(nn.Module):
    """アダプティブ損失重み付け - 精度向上のため"""
    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i, l in enumerate(losses):
            loss += precision[i] * l + self.log_vars[i]
        return loss


class EnhancedMambaLayer(nn.Module):
    """精度向上のため大幅に改良されたMambaレイヤー"""
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, n_heads: int = 8, 
                 num_experts: int = 4, d_ff: int = None, dropout: float = 0.1, 
                 max_seq_len: int = 2048, drop_path: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        # RMSNormを使用（LayerNormより効率的）
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        
        # Mambaコンポーネント（改良版）
        self.mamba = self._create_enhanced_mamba_block()
        
        # マルチヘッドアテンション
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, use_rope=True, 
            use_flash_attn=True, max_seq_len=max_seq_len
        )
        
        # MoE FFN
        if num_experts > 1:
            self.moe = MoELayer(d_model, num_experts, d_ff, top_k=2, 
                              dropout=dropout, expert_activation='swiglu')
        else:
            # 単一エキスパート（SwiGLU FFN）
            self.moe = Expert(d_model, d_ff, dropout, 'swiglu')
        
        # レイヤースケール（精度向上のため）
        self.layer_scale1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.layer_scale2 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.layer_scale3 = nn.Parameter(torch.ones(d_model) * 1e-4)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_enhanced_mamba_block(self):
        """改良されたMambaブロック"""
        # nn.Parameterは別途定義が必要
        self.mamba_A = nn.Parameter(-torch.rand(self.d_state) - 1)  # 改良された初期化
        self.mamba_D = nn.Parameter(torch.ones(self.d_inner))
        
        return nn.ModuleDict({
            'in_proj': nn.Linear(self.d_model, self.d_inner * 2, bias=False),
            'conv1d': nn.Conv1d(
                self.d_inner, self.d_inner, 
                kernel_size=4,  # カーネルサイズ増加
                padding=2, groups=self.d_inner, bias=True
            ),
            'state_proj': nn.Linear(self.d_inner, self.d_state, bias=False),
            'B_proj': nn.Linear(self.d_inner, self.d_state, bias=False),
            'C_proj': nn.Linear(self.d_inner, self.d_state, bias=False),
            'out_proj': nn.Linear(self.d_inner, self.d_model, bias=False),
            'dt_proj': nn.Linear(self.d_inner, self.d_state, bias=True),  # 時間ステップ学習
        })

    def mamba_forward(self, x):
        """精度向上のため改良されたMamba forward"""
        batch_size, seq_len, _ = x.shape
        
        # 入力投影
        x_and_res = self.mamba['in_proj'](x)
        x_main, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # 1D畳み込み（改良版）
        x_conv = x_main.transpose(1, 2)
        x_conv = self.mamba['conv1d'](x_conv)[:, :, :seq_len]  # パディング調整
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 適応的時間ステップ学習
        dt = self.mamba['dt_proj'](x_conv)
        dt = F.softplus(dt)  # 正の値を保証
        
        # 状態空間パラメータ
        B = self.mamba['B_proj'](x_conv)
        C = self.mamba['C_proj'](x_conv)
        
        # 改良された状態更新
        A = self.mamba_A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_state)
        dA = dt * A  # (B, L, d_state)
        dB = dt * B  # (B, L, d_state)
        
        # 効率的なスキャン操作（精度向上版）
        states = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(seq_len):
            states = states * torch.exp(dA[:, i]) + dB[:, i]
            y = (states * C[:, i]).sum(dim=-1, keepdim=True)  # (B, 1)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (B, L, 1)
        y = y.expand(-1, -1, self.d_inner)  # (B, L, d_inner)
        
        # スキップコネクション
        y = y + x_conv * self.mamba_D
        
        # ゲート機構
        y = y * F.silu(res)
        
        # 出力投影
        y = self.mamba['out_proj'](y)
        
        return y

    def forward(self, x, mask=None):
        # Pre-normalization + Residual connection + Layer scale
        
        # Mamba branch
        mamba_out = self.mamba_forward(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1 * mamba_out)
        
        # Attention branch
        attn_out, _ = self.attention(self.norm2(x), mask)
        x = x + self.drop_path(self.layer_scale2 * attn_out)
        
        # MoE/FFN branch
        if isinstance(self.moe, MoELayer):
            # MoE case
            moe_out, balance_loss = self.moe(self.norm3(x))
            self.balance_loss = balance_loss
        else:
            # Single expert case
            moe_out = self.moe(self.norm3(x))
            self.balance_loss = torch.tensor(0.0, device=x.device)
        
        x = x + self.drop_path(self.layer_scale3 * moe_out)
        
        return x


class HybridMambaModel(nn.Module):
    """精度向上のため大幅改良されたHybrid Mambaモデル"""
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 8, 
                 d_state: int = 32, n_heads: int = 16, num_experts: int = 8, 
                 max_seq_len: int = 1024, dropout: float = 0.15, 
                 use_gradient_checkpointing: bool = True, drop_path_rate: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # トークン埋め込み（改良版）
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 学習可能な位置埋め込み（RoPEと併用）
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # DropPath率の線形増加
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        
        # トランスフォーマーレイヤー
        self.layers = nn.ModuleList([
            EnhancedMambaLayer(
                d_model=d_model,
                d_state=d_state,
                n_heads=n_heads,
                num_experts=num_experts,
                dropout=dropout,
                max_seq_len=max_seq_len,
                drop_path=dpr[i]
            ) for i in range(n_layers)
        ])
        
        # 最終正規化（RMSNorm）
        self.norm = RMSNorm(d_model)
        
        # 言語モデリングヘッド
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 重み共有（embeddings and lm_head）
        self.lm_head.weight = self.embedding.weight
        
        # アダプティブ損失重み付け
        self.adaptive_loss = AdaptiveLossWeighting(num_tasks=2)
        
        # パラメータ初期化
        self.apply(self._init_weights)
        
        # 特別な初期化
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def _init_weights(self, module):
        """改良されたパラメータ初期化"""
        if isinstance(module, nn.Linear):
            # Xavier uniform初期化 with scaling
            std = math.sqrt(2.0 / (module.in_features + module.out_features))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len):
        """因果マスクの作成"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # トークン埋め込み + 位置埋め込み
        x = self.embedding(input_ids)
        if seq_len <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :seq_len]
        x = self.embed_dropout(x)
        
        # Causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # レイヤー処理
        balance_losses = []
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
            
            # バランス損失を収集
            if hasattr(layer, 'balance_loss'):
                balance_losses.append(layer.balance_loss)
        
        # 最終正規化
        x = self.norm(x)
        
        # 言語モデリングヘッド
        logits = self.lm_head(x)
        
        # 損失計算
        loss = None
        if labels is not None:
            # メイン損失（クロスエントロピー）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size), 
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # バランス損失
            balance_loss = torch.stack(balance_losses).mean() if balance_losses else torch.tensor(0.0)
            
            # アダプティブ損失重み付け
            losses = [main_loss, balance_loss]
            loss = self.adaptive_loss(losses)
        
        return {
            "logits": logits, 
            "loss": loss,
            "balance_loss": balance_loss if labels is not None else None
        }
    
    def generate(self, input_ids, max_length=50, temperature=1.0, do_sample=True):
        """テキスト生成"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                if do_sample:
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if input_ids.size(1) >= self.max_seq_len:
                    break
                    
        return input_ids
    
    def get_model_info(self):
        """改良されたモデル情報の取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # コンポーネント別のパラメータ数を計算
        embedding_params = sum(p.numel() for p in [self.embedding.weight, self.pos_embedding])
        layer_params = sum(p.numel() for p in self.layers.parameters())
        output_params = sum(p.numel() for p in [self.norm.weight, self.norm.bias, self.lm_head.weight])
        
        # 各レイヤーのエキスパート使用統計
        expert_usage_stats = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer.moe.gate, 'expert_usage'):
                usage = layer.moe.gate.expert_usage.cpu().numpy()
                expert_usage_stats[f'layer_{i}'] = {
                    'mean_usage': float(usage.mean()),
                    'std_usage': float(usage.std()),
                    'usage_distribution': usage.tolist()
                }
        
        return {
            "model_type": "HybridMambaModel (Mamba + Transformer + MoE)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_breakdown": {
                "embedding": embedding_params,
                "layers": layer_params,
                "output": output_params
            },
            "architecture": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.layers[0].attention.n_heads if self.layers else 0,
                "num_experts": self.layers[0].moe.num_experts if self.layers else 0,
                "max_seq_len": self.max_seq_len
            },
            "expert_usage_stats": expert_usage_stats,
            "memory_efficiency": {
                "gradient_checkpointing": self.use_gradient_checkpointing,
                "parameter_sharing": "lm_head and embedding weights separate"  # 現在は分離
            }
        }