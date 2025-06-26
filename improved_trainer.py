#!/usr/bin/env python3
"""
精度向上のためのImproved Mamba Trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from improved_config import ImprovedLearningRateScheduler


class AdvancedTrainingMetrics:
    """高度な訓練メトリクス"""
    def __init__(self):
        self.train_losses = []
        self.balance_losses = []
        self.learning_rates = []
        self.perplexities = []
        self.gradient_norms = []
        self.expert_usage = []  # エキスパート使用統計
        self.attention_entropy = []  # アテンションエントロピー
        self.step_times = []
        self.memory_usage = []
        
        # 評価メトリクス
        self.eval_losses = []
        self.eval_perplexities = []
        self.bleu_scores = []
        
    def add_training_step(self, loss: float, balance_loss: float, lr: float, 
                         grad_norm: float, step_time: float, memory_mb: float = 0.0,
                         expert_stats: Optional[Dict] = None):
        self.train_losses.append(loss)
        self.balance_losses.append(balance_loss)
        self.learning_rates.append(lr)
        self.perplexities.append(math.exp(min(loss, 10)))
        self.gradient_norms.append(grad_norm)
        self.step_times.append(step_time)
        self.memory_usage.append(memory_mb)
        
        if expert_stats:
            self.expert_usage.append(expert_stats)
    
    def add_eval_step(self, eval_loss: float, eval_perplexity: float, bleu_score: float = 0.0):
        self.eval_losses.append(eval_loss)
        self.eval_perplexities.append(eval_perplexity)
        self.bleu_scores.append(bleu_score)
    
    def get_smoothed_metric(self, metric_list: List[float], window: int = 100) -> float:
        """スムーズ化されたメトリクス"""
        if not metric_list:
            return 0.0
        return np.mean(metric_list[-window:])
    
    def get_expert_balance_score(self) -> float:
        """エキスパート使用バランススコア"""
        if not self.expert_usage:
            return 0.0
        
        latest_usage = self.expert_usage[-1]
        if not latest_usage:
            return 0.0
            
        usage_values = list(latest_usage.values())
        if not usage_values:
            return 0.0
            
        # ジニ係数を計算してバランスを評価
        sorted_usage = sorted(usage_values)
        n = len(sorted_usage)
        cumsum = np.cumsum(sorted_usage)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0


class ImprovedMambaTrainer:
    """改良されたMambaトレーナー"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 use_mixed_precision: bool = True, use_compilation: bool = True):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.use_compilation = use_compilation
        
        # モデルをデバイスに移動
        self.model.to(device)
        
        # 最適化設定
        if use_compilation and hasattr(torch, 'compile'):
            print("🚀 モデルコンパイルを使用します")
            self.model = torch.compile(self.model)
        
        # Mixed precision設定
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
        
        self.optimizer = None
        self.scheduler = None
        self.metrics = AdvancedTrainingMetrics()
        
        # 早期停止設定
        self.best_eval_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        
    def setup_optimizer(self, learning_rate: float = 5e-4, weight_decay: float = 0.1,
                       warmup_steps: int = 1000, max_steps: int = 10000):
        """最適化器の設定"""
        # パラメータをグループ化（重み減衰の対象を分ける）
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'LayerNorm', 'layernorm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # AdamWオプティマイザー（推奨設定）
        self.optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 学習率スケジューラー
        self.scheduler = ImprovedLearningRateScheduler(
            self.optimizer, warmup_steps, max_steps
        )
        
        print(f"📈 最適化器設定完了:")
        print(f"  - 学習率: {learning_rate}")
        print(f"  - 重み減衰パラメータ: {len(decay_params)}")
        print(f"  - 重み減衰なしパラメータ: {len(no_decay_params)}")
    
    def compute_loss_with_label_smoothing(self, logits: torch.Tensor, labels: torch.Tensor, 
                                        smoothing: float = 0.1) -> torch.Tensor:
        """ラベルスムージング付き損失計算"""
        vocab_size = logits.size(-1)
        confidence = 1.0 - smoothing
        
        # シフト済みロジットとラベル
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # ログ確率の計算
        log_probs = F.log_softmax(shift_logits.view(-1, vocab_size), dim=-1)
        
        # 正解ラベルの確率
        nll_loss = F.nll_loss(log_probs, shift_labels.view(-1), 
                              ignore_index=-100, reduction='mean')
        
        # スムージング項
        smooth_loss = -log_probs.mean()
        
        # 組み合わせ
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   gradient_accumulation_steps: int = 1,
                   max_grad_norm: float = 1.0,
                   label_smoothing: float = 0.1) -> Dict[str, float]:
        """改良された訓練ステップ"""
        step_start_time = time.time()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Mixed precision forward pass
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, labels=labels)
                main_loss = outputs['loss']
                balance_loss = outputs.get('balance_loss', 0.0)
                
                # ラベルスムージング付き損失
                if label_smoothing > 0:
                    logits = outputs['logits']
                    main_loss = self.compute_loss_with_label_smoothing(
                        logits, labels, label_smoothing
                    )
                
                total_loss = main_loss + 0.01 * balance_loss
                total_loss = total_loss / gradient_accumulation_steps
        else:
            outputs = self.model(input_ids, labels=labels)
            main_loss = outputs['loss']
            balance_loss = outputs.get('balance_loss', 0.0)
            total_loss = main_loss + 0.01 * balance_loss
            total_loss = total_loss / gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # 勾配ノルムの計算
        grad_norm = 0.0
        if max_grad_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                ).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                ).item()
        
        # エキスパート使用統計（MoE対応）
        expert_stats = {}
        if hasattr(self.model, 'get_expert_usage'):
            expert_stats = self.model.get_expert_usage()
        
        step_time = time.time() - step_start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # メモリ使用量
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return {
            'loss': main_loss.item(),
            'balance_loss': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss,
            'total_loss': total_loss.item() * gradient_accumulation_steps,
            'grad_norm': grad_norm,
            'lr': current_lr,
            'step_time': step_time,
            'memory_mb': memory_mb,
            'expert_stats': expert_stats
        }
    
    def optimizer_step(self):
        """最適化ステップの実行"""
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if self.scheduler:
            self.scheduler.step()
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """モデル評価"""
        self.model.eval()
        total_loss = 0.0
        total_balance_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="評価中", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, labels=labels)
                else:
                    outputs = self.model(input_ids, labels=labels)
                
                total_loss += outputs['loss'].item()
                total_balance_loss += outputs.get('balance_loss', 0.0)
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_balance_loss = total_balance_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(min(avg_loss, 10))
        
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_balance_loss': avg_balance_loss,
            'eval_perplexity': perplexity
        }
    
    def train(self, train_dataloaders: Dict[str, DataLoader], 
              eval_dataloaders: Optional[Dict[str, DataLoader]] = None,
              max_steps: int = 10000,
              gradient_accumulation_steps: int = 4,
              eval_steps: int = 500,
              save_steps: int = 1000,
              logging_steps: int = 100,
              save_dir: str = 'checkpoints',
              max_grad_norm: float = 1.0,
              label_smoothing: float = 0.1):
        """改良された訓練ループ"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("🚀 改良された訓練を開始します")
        print(f"  - 最大ステップ数: {max_steps}")
        print(f"  - 勾配蓄積ステップ: {gradient_accumulation_steps}")
        print(f"  - 評価間隔: {eval_steps}")
        print(f"  - 保存間隔: {save_steps}")
        print(f"  - デバイス: {self.device}")
        print(f"  - Mixed Precision: {self.scaler is not None}")
        print("-" * 80)
        
        global_step = 0
        best_eval_loss = float('inf')
        
        # データローダーの組み合わせ
        all_dataloaders = list(train_dataloaders.values())
        
        while global_step < max_steps:
            for dataloader_idx, dataloader in enumerate(all_dataloaders):
                for batch_idx, batch in enumerate(dataloader):
                    if global_step >= max_steps:
                        break
                    
                    # 訓練ステップ
                    step_metrics = self.train_step(
                        batch, gradient_accumulation_steps, 
                        max_grad_norm, label_smoothing
                    )
                    
                    # メトリクス記録
                    self.metrics.add_training_step(
                        step_metrics['loss'],
                        step_metrics['balance_loss'],
                        step_metrics['lr'],
                        step_metrics['grad_norm'],
                        step_metrics['step_time'],
                        step_metrics['memory_mb'],
                        step_metrics['expert_stats']
                    )
                    
                    # 勾配蓄積が完了した時のみ最適化
                    if (global_step + 1) % gradient_accumulation_steps == 0:
                        self.optimizer_step()
                    
                    global_step += 1
                    
                    # ログ出力
                    if global_step % logging_steps == 0:
                        self.log_training_progress(global_step, max_steps)
                    
                    # 評価
                    if eval_dataloaders and global_step % eval_steps == 0:
                        eval_results = self.evaluate_all(eval_dataloaders)
                        
                        # 早期停止チェック
                        if eval_results['avg_eval_loss'] < best_eval_loss:
                            best_eval_loss = eval_results['avg_eval_loss']
                            self.patience_counter = 0
                            self.save_checkpoint(
                                os.path.join(save_dir, f'best_model_step_{global_step}.pt'),
                                global_step, eval_results
                            )
                        else:
                            self.patience_counter += 1
                        
                        self.metrics.add_eval_step(
                            eval_results['avg_eval_loss'],
                            eval_results['avg_eval_perplexity']
                        )
                    
                    # 定期保存
                    if global_step % save_steps == 0:
                        self.save_checkpoint(
                            os.path.join(save_dir, f'checkpoint_step_{global_step}.pt'),
                            global_step, {}
                        )
                    
                    # 早期停止
                    if self.patience_counter >= self.patience:
                        print(f"🛑 早期停止: {self.patience}回連続で改善なし")
                        return
                
                if global_step >= max_steps:
                    break
        
        print("✅ 訓練完了!")
        
        # 最終チェックポイントの保存
        final_checkpoint = os.path.join(save_dir, 'final_model.pt')
        self.save_checkpoint(final_checkpoint, global_step, {})
        
        # 訓練曲線のプロット
        self.plot_advanced_training_curves(save_dir)
    
    def evaluate_all(self, eval_dataloaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """全ての評価データローダーで評価"""
        all_results = {}
        total_loss = 0.0
        total_perplexity = 0.0
        num_datasets = len(eval_dataloaders)
        
        for name, dataloader in eval_dataloaders.items():
            results = self.evaluate(dataloader)
            all_results[f'{name}_eval_loss'] = results['eval_loss']
            all_results[f'{name}_eval_perplexity'] = results['eval_perplexity']
            
            total_loss += results['eval_loss']
            total_perplexity += results['eval_perplexity']
        
        all_results['avg_eval_loss'] = total_loss / num_datasets
        all_results['avg_eval_perplexity'] = total_perplexity / num_datasets
        
        return all_results
    
    def log_training_progress(self, step: int, max_steps: int):
        """訓練進捗のログ出力"""
        if not self.metrics.train_losses:
            return
        
        # スムーズ化されたメトリクス
        smooth_loss = self.metrics.get_smoothed_metric(self.metrics.train_losses, 50)
        smooth_balance_loss = self.metrics.get_smoothed_metric(self.metrics.balance_losses, 50)
        smooth_perplexity = self.metrics.get_smoothed_metric(self.metrics.perplexities, 50)
        smooth_grad_norm = self.metrics.get_smoothed_metric(self.metrics.gradient_norms, 50)
        
        current_lr = self.metrics.learning_rates[-1]
        current_memory = self.metrics.memory_usage[-1]
        expert_balance = self.metrics.get_expert_balance_score()
        
        progress_pct = (step / max_steps) * 100
        
        print(f"ステップ {step:6d}/{max_steps} ({progress_pct:5.1f}%) | "
              f"Loss: {smooth_loss:.4f} | "
              f"BLoss: {smooth_balance_loss:.4f} | "
              f"PPL: {smooth_perplexity:.2f} | "
              f"GradNorm: {smooth_grad_norm:.3f} | "
              f"LR: {current_lr:.2e} | "
              f"Mem: {current_memory:.0f}MB | "
              f"ExpertBal: {expert_balance:.3f}")
    
    def save_checkpoint(self, path: str, step: int, metrics: Dict):
        """チェックポイントの保存"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.__dict__ if self.scheduler else None,
            'metrics': metrics,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(checkpoint, path)
        print(f"💾 チェックポイント保存: {path}")
    
    def plot_advanced_training_curves(self, save_dir: str):
        """高度な訓練曲線のプロット"""
        if not self.metrics.train_losses:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('改良されたMamba訓練曲線', fontsize=16)
        
        # 訓練損失
        axes[0, 0].plot(self.metrics.train_losses, alpha=0.7, label='訓練損失')
        if self.metrics.eval_losses:
            eval_steps = np.linspace(0, len(self.metrics.train_losses), len(self.metrics.eval_losses))
            axes[0, 0].plot(eval_steps, self.metrics.eval_losses, 'r-', label='評価損失')
        axes[0, 0].set_title('損失曲線')
        axes[0, 0].set_xlabel('ステップ')
        axes[0, 0].set_ylabel('損失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # バランス損失
        axes[0, 1].plot(self.metrics.balance_losses, 'g-', alpha=0.7)
        axes[0, 1].set_title('MoEバランス損失')
        axes[0, 1].set_xlabel('ステップ')
        axes[0, 1].set_ylabel('バランス損失')
        axes[0, 1].grid(True)
        
        # パープレキシティ
        axes[0, 2].plot(self.metrics.perplexities, 'purple', alpha=0.7, label='訓練PPL')
        if self.metrics.eval_perplexities:
            eval_steps = np.linspace(0, len(self.metrics.perplexities), len(self.metrics.eval_perplexities))
            axes[0, 2].plot(eval_steps, self.metrics.eval_perplexities, 'r-', label='評価PPL')
        axes[0, 2].set_title('パープレキシティ')
        axes[0, 2].set_xlabel('ステップ')
        axes[0, 2].set_ylabel('パープレキシティ')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 学習率
        axes[1, 0].plot(self.metrics.learning_rates, 'orange')
        axes[1, 0].set_title('学習率スケジュール')
        axes[1, 0].set_xlabel('ステップ')
        axes[1, 0].set_ylabel('学習率')
        axes[1, 0].grid(True)
        
        # 勾配ノルム
        axes[1, 1].plot(self.metrics.gradient_norms, 'brown', alpha=0.7)
        axes[1, 1].set_title('勾配ノルム')
        axes[1, 1].set_xlabel('ステップ')
        axes[1, 1].set_ylabel('勾配ノルム')
        axes[1, 1].grid(True)
        
        # メモリ使用量
        axes[1, 2].plot(self.metrics.memory_usage, 'cyan')
        axes[1, 2].set_title('メモリ使用量')
        axes[1, 2].set_xlabel('ステップ')
        axes[1, 2].set_ylabel('メモリ (MB)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'advanced_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 高度な訓練曲線を保存しました: {save_dir}/advanced_training_curves.png") 