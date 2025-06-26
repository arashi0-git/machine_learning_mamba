import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from typing import Dict, List, Optional
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.balance_losses = []  # MoE負荷バランス損失
        self.learning_rates = []
        self.perplexities = []
        self.epoch_times = []
        self.step_times = []
        self.memory_usage = []  # メモリ使用量
        
    def add_step(self, loss: float, balance_loss: float, lr: float, step_time: float, memory_mb: float = 0.0):
        self.train_losses.append(loss)
        self.balance_losses.append(balance_loss)
        self.learning_rates.append(lr)
        self.perplexities.append(math.exp(min(loss, 10)))
        self.step_times.append(step_time)
        self.memory_usage.append(memory_mb)
    
    def add_epoch_time(self, epoch_time: float):
        self.epoch_times.append(epoch_time)
    
    def get_average_loss(self, last_n: int = 100) -> float:
        if not self.train_losses:
            return 0.0
        return sum(self.train_losses[-last_n:]) / min(len(self.train_losses), last_n)
    
    def get_average_perplexity(self, last_n: int = 100) -> float:
        if not self.perplexities:
            return 0.0
        return sum(self.perplexities[-last_n:]) / min(len(self.perplexities), last_n)
    
    def get_average_balance_loss(self, last_n: int = 100) -> float:
        if not self.balance_losses:
            return 0.0
        return sum(self.balance_losses[-last_n:]) / min(len(self.balance_losses), last_n)


class ProgressTracker:
    def __init__(self, total_epochs: int, steps_per_epoch: int):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        
    def update(self, epoch: int, step: int):
        self.current_epoch = epoch
        self.current_step = step
        self.global_step = epoch * self.steps_per_epoch + step
    
    def get_progress_percentage(self) -> float:
        return (self.global_step / self.total_steps) * 100
    
    def get_epoch_progress_percentage(self) -> float:
        return (self.current_step / self.steps_per_epoch) * 100
    
    def get_eta(self, avg_step_time: float) -> str:
        remaining_steps = self.total_steps - self.global_step
        eta_seconds = remaining_steps * avg_step_time
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class MambaTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = None
        self.scheduler = None
        self.metrics = TrainingMetrics()
        self.progress_tracker = None
        
    def setup_training(self, learning_rate: float = 1e-4, weight_decay: float = 0.01):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def get_average_balance_loss(self, last_n: int = 100) -> float:
        """負荷バランス損失の平均を取得"""
        if not self.metrics.balance_losses:
            return 0.0
        return sum(self.metrics.balance_losses[-last_n:]) / min(len(self.metrics.balance_losses), last_n)
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        if self.progress_tracker is None:
            self.progress_tracker = ProgressTracker(1, num_batches)
        
        epoch_start_time = time.time()
        
        pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f'Epoch {epoch+1}')
        
        for step, batch in pbar:
            step_start_time = time.time()
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']
            balance_loss = outputs.get('balance_loss', 0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            step_time = time.time() - step_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # メモリ使用量の取得
            memory_mb = 0.0
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            self.metrics.add_step(loss.item(), balance_loss, current_lr, step_time, memory_mb)
            self.progress_tracker.update(epoch, step)
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                avg_loss = self.metrics.get_average_loss(last_n=10)
                avg_balance_loss = self.get_average_balance_loss(last_n=10)
                avg_perplexity = self.metrics.get_average_perplexity(last_n=10)
                progress_pct = self.progress_tracker.get_progress_percentage()
                epoch_progress_pct = self.progress_tracker.get_epoch_progress_percentage()
                
                avg_step_time = sum(self.metrics.step_times[-100:]) / min(len(self.metrics.step_times), 100)
                eta = self.progress_tracker.get_eta(avg_step_time)
                
                # 現在のメモリ使用量
                current_memory = self.metrics.memory_usage[-1] if self.metrics.memory_usage else 0.0
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'BLoss': f'{avg_balance_loss:.4f}',
                    'PPL': f'{avg_perplexity:.2f}',
                    'Mem': f'{current_memory:.0f}MB',
                    'ETA': eta
                })
                
                self.print_training_status(epoch, step, avg_loss, avg_balance_loss, avg_perplexity, 
                                         current_memory, progress_pct, epoch_progress_pct, eta)
        
        epoch_time = time.time() - epoch_start_time
        self.metrics.add_epoch_time(epoch_time)
        
        avg_loss = total_loss / num_batches
        return {
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'perplexity': math.exp(min(avg_loss, 10))
        }
    
    def train(self, dataloaders: Dict[str, DataLoader], epochs: int = 5, save_dir: str = 'checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        all_dataloaders = list(dataloaders.values())
        total_steps = sum(len(dl) for dl in all_dataloaders) * epochs
        self.progress_tracker = ProgressTracker(epochs, sum(len(dl) for dl in all_dataloaders))
        
        print(f"開始訓練 - 総エポック数: {epochs}, 総ステップ数: {total_steps}")
        print(f"データセット: {list(dataloaders.keys())}")
        print(f"デバイス: {self.device}")
        print("-" * 80)
        
        for epoch in range(epochs):
            print(f"\nエポック {epoch + 1}/{epochs}")
            print("=" * 50)
            
            epoch_metrics = {}
            
            for lang, dataloader in dataloaders.items():
                print(f"\n{lang.upper()}データの学習中...")
                metrics = self.train_epoch(dataloader, epoch)
                epoch_metrics[lang] = metrics
                
                print(f"{lang} - 損失: {metrics['avg_loss']:.4f}, "
                      f"パープレキシティ: {metrics['perplexity']:.2f}, "
                      f"時間: {metrics['epoch_time']:.2f}秒")
            
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            self.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
            
            self.plot_training_progress(save_dir)
            
        print("\n" + "=" * 80)
        print("訓練完了！")
        final_metrics = self.get_final_metrics()
        print(f"最終損失: {final_metrics['final_loss']:.4f}")
        print(f"最終パープレキシティ: {final_metrics['final_perplexity']:.2f}")
        print(f"総訓練時間: {final_metrics['total_time']:.2f}秒")
    
    def print_training_status(self, epoch: int, step: int, loss: float, balance_loss: float, 
                            perplexity: float, memory_mb: float, progress_pct: float, 
                            epoch_progress_pct: float, eta: str):
        status = (
            f"エポック {epoch+1} | ステップ {step+1} | "
            f"損失: {loss:.4f} | バランス損失: {balance_loss:.4f} | "
            f"PPL: {perplexity:.2f} | メモリ: {memory_mb:.0f}MB | "
            f"進捗: {progress_pct:.1f}% | ETA: {eta}"
        )
        if step % 50 == 0:
            print(status)
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'training_metrics': {
                'train_losses': self.metrics.train_losses,
                'balance_losses': self.metrics.balance_losses,
                'perplexities': self.metrics.perplexities,
                'learning_rates': self.metrics.learning_rates,
                'memory_usage': self.metrics.memory_usage,
                'step_times': self.metrics.step_times
            }
        }
        torch.save(checkpoint, path)
        print(f"チェックポイント保存: {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    
    def plot_training_progress(self, save_dir: str):
        if len(self.metrics.train_losses) < 10:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # 訓練損失
        axes[0, 0].plot(self.metrics.train_losses, label='Train Loss')
        if self.metrics.balance_losses:
            axes[0, 0].plot(self.metrics.balance_losses, label='Balance Loss', alpha=0.7)
        axes[0, 0].set_title('訓練損失')
        axes[0, 0].set_xlabel('ステップ')
        axes[0, 0].set_ylabel('損失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # パープレキシティ
        axes[0, 1].plot(self.metrics.perplexities)
        axes[0, 1].set_title('パープレキシティ')
        axes[0, 1].set_xlabel('ステップ')
        axes[0, 1].set_ylabel('パープレキシティ')
        axes[0, 1].grid(True)
        
        # 学習率
        axes[1, 0].plot(self.metrics.learning_rates)
        axes[1, 0].set_title('学習率')
        axes[1, 0].set_xlabel('ステップ')
        axes[1, 0].set_ylabel('学習率')
        axes[1, 0].grid(True)
        
        # メモリ使用量
        if self.metrics.memory_usage:
            axes[1, 1].plot(self.metrics.memory_usage)
            axes[1, 1].set_title('メモリ使用量')
            axes[1, 1].set_xlabel('ステップ')
            axes[1, 1].set_ylabel('メモリ (MB)')
            axes[1, 1].grid(True)
        
        # エポック時間
        if self.metrics.epoch_times:
            axes[2, 0].bar(range(len(self.metrics.epoch_times)), self.metrics.epoch_times)
            axes[2, 0].set_title('エポック時間')
            axes[2, 0].set_xlabel('エポック')
            axes[2, 0].set_ylabel('時間 (秒)')
        
        # ステップ時間の移動平均
        if len(self.metrics.step_times) > 100:
            step_times_ma = np.convolve(self.metrics.step_times, np.ones(50)/50, mode='valid')
            axes[2, 1].plot(step_times_ma)
            axes[2, 1].set_title('ステップ時間 (移動平均)')
            axes[2, 1].set_xlabel('ステップ')
            axes[2, 1].set_ylabel('時間 (秒)')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_final_metrics(self) -> Dict[str, float]:
        return {
            'final_loss': self.metrics.get_average_loss(last_n=100),
            'final_balance_loss': self.get_average_balance_loss(last_n=100),
            'final_perplexity': self.metrics.get_average_perplexity(last_n=100),
            'total_time': sum(self.metrics.epoch_times),
            'avg_step_time': sum(self.metrics.step_times) / len(self.metrics.step_times) if self.metrics.step_times else 0.0,
            'peak_memory_mb': max(self.metrics.memory_usage) if self.metrics.memory_usage else 0.0,
            'avg_memory_mb': sum(self.metrics.memory_usage) / len(self.metrics.memory_usage) if self.metrics.memory_usage else 0.0
        }
    
    def evaluate_model(self, test_text: str, max_length: int = 50) -> str:
        self.model.eval()
        
        from data_loader import TextTokenizer
        tokenizer = TextTokenizer()
        
        input_ids = torch.tensor([tokenizer.encode(test_text, max_length=20)], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(input_ids, max_length=max_length, temperature=0.8)
            
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        return generated_text