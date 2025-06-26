#!/usr/bin/env python3
"""
精度向上のための改善されたHybrid Mamba設定
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ImprovedModelConfig:
    """改善されたモデル設定"""
    # モデルアーキテクチャ
    d_model: int = 512  # 大幅増加
    n_layers: int = 8   # レイヤー数増加
    d_state: int = 32   # 状態次元増加
    n_heads: int = 16   # ヘッド数増加
    num_experts: int = 8  # エキスパート数増加
    expert_capacity: float = 1.25  # エキスパート容量
    
    # MoE設定
    top_k: int = 2
    balance_loss_weight: float = 0.01  # バランス損失の重み
    expert_dropout: float = 0.1
    expert_activation: str = 'swiglu'  # より効果的な活性化
    
    # Attention設定
    use_rope: bool = True
    use_flash_attention: bool = True
    attention_dropout: float = 0.1
    
    # 正則化
    dropout: float = 0.15
    weight_decay: float = 0.1
    label_smoothing: float = 0.1
    
    # シーケンス処理
    max_seq_len: int = 512  # 長いシーケンス対応
    
    # 語彙とトークン化
    vocab_size: int = 32000  # より大きな語彙サイズ
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2


@dataclass 
class ImprovedTrainingConfig:
    """改善されたトレーニング設定"""
    # 学習率スケジューリング
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 1000
    max_steps: int = 10000
    
    # バッチ処理
    batch_size: int = 16  # 増加
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 最適化
    optimizer: str = 'adamw'
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # 評価とチェックポイント
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # データ拡張
    use_data_augmentation: bool = True
    max_position_embeddings: int = 512


def get_improved_model_args() -> Dict[str, Any]:
    """改善されたモデル引数を取得"""
    config = ImprovedModelConfig()
    return {
        'vocab_size': config.vocab_size,
        'd_model': config.d_model, 
        'n_layers': config.n_layers,
        'd_state': config.d_state,
        'n_heads': config.n_heads,
        'num_experts': config.num_experts,
        'max_seq_len': config.max_seq_len,
        'dropout': config.dropout,
        'use_gradient_checkpointing': True,  # メモリ効率化
    }


def get_improved_training_args() -> Dict[str, Any]:
    """改善されたトレーニング引数を取得"""
    config = ImprovedTrainingConfig()
    return {
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'max_steps': config.max_steps,
        'warmup_steps': config.warmup_steps,
        'weight_decay': 0.1,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'max_grad_norm': config.max_grad_norm,
        'eval_steps': config.eval_steps,
        'save_steps': config.save_steps,
        'logging_steps': config.logging_steps,
    }


class ImprovedLearningRateScheduler:
    """改善された学習率スケジューラー"""
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, 
                 min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # Warmup期間：線形増加
            return self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # コサインアニーリング
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return self.initial_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)


def print_model_config():
    """モデル設定を表示"""
    model_config = ImprovedModelConfig()
    training_config = ImprovedTrainingConfig()
    
    print("=" * 60)
    print("改善されたHybrid Mambaモデル設定")
    print("=" * 60)
    print("📊 モデルアーキテクチャ:")
    print(f"  - モデル次元: {model_config.d_model}")
    print(f"  - レイヤー数: {model_config.n_layers}")
    print(f"  - 状態次元: {model_config.d_state}")
    print(f"  - アテンションヘッド: {model_config.n_heads}")
    print(f"  - MoEエキスパート数: {model_config.num_experts}")
    print(f"  - 語彙サイズ: {model_config.vocab_size}")
    print(f"  - 最大シーケンス長: {model_config.max_seq_len}")
    
    print("\n🎯 トレーニング設定:")
    print(f"  - 学習率: {training_config.learning_rate}")
    print(f"  - バッチサイズ: {training_config.batch_size}")
    print(f"  - 勾配蓄積ステップ: {training_config.gradient_accumulation_steps}")
    print(f"  - 最大ステップ数: {training_config.max_steps}")
    print(f"  - ウォームアップステップ: {training_config.warmup_steps}")
    
    # パラメータ数の推定
    estimated_params = estimate_parameters(model_config)
    print(f"\n💾 推定パラメータ数: {estimated_params:,}")
    print("=" * 60)


def estimate_parameters(config: ImprovedModelConfig) -> int:
    """モデルのパラメータ数を推定"""
    vocab_params = config.vocab_size * config.d_model * 2  # embedding + lm_head
    
    # Mambaレイヤーのパラメータ
    mamba_params_per_layer = (
        config.d_model * config.d_model * 6 +  # 投影層
        config.d_model * config.d_state * 3 +  # 状態パラメータ
        config.d_state * 2  # A, D パラメータ
    )
    
    # Attentionのパラメータ
    attn_params_per_layer = config.d_model * config.d_model * 4  # Q, K, V, O
    
    # MoEのパラメータ
    expert_params_per_layer = config.num_experts * config.d_model * config.d_model * 3
    gate_params_per_layer = config.d_model * config.num_experts
    
    layer_params = (mamba_params_per_layer + attn_params_per_layer + 
                   expert_params_per_layer + gate_params_per_layer)
    
    total_params = vocab_params + (layer_params * config.n_layers)
    return total_params


if __name__ == "__main__":
    print_model_config() 