#!/usr/bin/env python3
"""
精度向上のための改良されたMambaトレーニングスクリプト
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import random
from typing import Dict, Any
import json

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_mamba_model import HybridMambaModel
from improved_trainer import ImprovedMambaTrainer
from improved_config import (
    get_improved_model_args, 
    get_improved_training_args,
    print_model_config
)
from data_loader import (
    ImprovedTextDataset, 
    create_balanced_dataloader, 
    create_sample_data
)


def set_random_seeds(seed: int = 42):
    """再現性のためのシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device() -> str:
    """最適なデバイスを選択"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 CUDA利用可能: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("💻 CPUを使用します")
    
    return device


def print_system_info():
    """システム情報を表示"""
    print("=" * 80)
    print("🔧 システム情報")
    print("=" * 80)
    print(f"Python バージョン: {sys.version}")
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"CUDA 利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA バージョン: {torch.version.cuda}")
        print(f"GPU デバイス数: {torch.cuda.device_count()}")
    print("=" * 80)


def create_experiment_config(args) -> Dict[str, Any]:
    """実験設定を作成"""
    model_args = get_improved_model_args()
    training_args = get_improved_training_args()
    
    # コマンドライン引数で上書き
    if args.d_model:
        model_args['d_model'] = args.d_model
    if args.n_layers:
        model_args['n_layers'] = args.n_layers
    if args.max_seq_len:
        model_args['max_seq_len'] = args.max_seq_len
    if args.num_experts:
        model_args['num_experts'] = args.num_experts
    
    if args.learning_rate:
        training_args['learning_rate'] = args.learning_rate
    if args.batch_size:
        training_args['batch_size'] = args.batch_size
    if args.max_steps:
        training_args['max_steps'] = args.max_steps
    
    return {
        'model': model_args,
        'training': training_args,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'experiment_name': args.experiment_name,
        'device': args.device
    }


def main():
    parser = argparse.ArgumentParser(description='改良されたMambaモデル学習')
    
    # データ関連
    parser.add_argument('--data_dir', type=str, default='sample_data',
                       help='学習データディレクトリ')
    parser.add_argument('--create_sample', action='store_true',
                       help='サンプルデータを作成')
    
    # モデル関連
    parser.add_argument('--d_model', type=int, default=None,
                       help='モデル次元数（デフォルト: 512）')
    parser.add_argument('--n_layers', type=int, default=None,
                       help='レイヤー数（デフォルト: 8）')
    parser.add_argument('--max_seq_len', type=int, default=None,
                       help='最大シーケンス長（デフォルト: 512）')
    parser.add_argument('--num_experts', type=int, default=None,
                       help='エキスパート数（デフォルト: 8）')
    
    # 学習関連
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学習率（デフォルト: 5e-4）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='バッチサイズ（デフォルト: 16）')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='最大ステップ数（デフォルト: 10000）')
    
    # システム関連
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='使用デバイス')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='チェックポイント保存ディレクトリ')
    parser.add_argument('--experiment_name', type=str, default='improved_mamba',
                       help='実験名')
    parser.add_argument('--seed', type=int, default=42,
                       help='ランダムシード')
    
    # その他
    parser.add_argument('--test_text', type=str, default='機械学習は',
                       help='生成テスト用テキスト')
    parser.add_argument('--no_compile', action='store_true',
                       help='PyTorchコンパイルを無効化')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Mixed Precisionを無効化')
    
    args = parser.parse_args()
    
    # システム情報表示
    print_system_info()
    
    # シード設定
    set_random_seeds(args.seed)
    print(f"🎲 ランダムシード設定: {args.seed}")
    
    # サンプルデータ作成
    if args.create_sample:
        create_sample_data(args.data_dir)
        if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
            print("❌ サンプルデータの作成に失敗しました")
            return
    
    # デバイス設定
    if args.device == 'auto':
        device = setup_device()
    else:
        device = args.device
    
    # 実験設定
    config = create_experiment_config(args)
    config['device'] = device
    
    # 設定表示
    print_model_config()
    
    print("\n" + "=" * 80)
    print("📊 実験設定")
    print("=" * 80)
    print(f"実験名: {config['experiment_name']}")
    print(f"データディレクトリ: {config['data_dir']}")
    print(f"保存ディレクトリ: {config['save_dir']}")
    print(f"デバイス: {device}")
    print("=" * 80)
    
    try:
        # データセット作成
        print("\n📁 データセット作成中...")
        dataset = ImprovedTextDataset(
            data_dir=config['data_dir'],
            max_length=config['model']['max_seq_len'],
            use_augmentation=True,
            balance_languages=True
        )
        
        if len(dataset) == 0:
            print("❌ データセットが空です。データディレクトリを確認してください。")
            return
        
        # データローダー作成
        train_dataloader = create_balanced_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            use_weighted_sampling=True
        )
        
        # 語彙サイズを設定
        vocab_size = len(dataset.processor.char_to_id)
        config['model']['vocab_size'] = vocab_size
        
        print(f"📚 語彙サイズ: {vocab_size}")
        
        # モデル作成
        print("\n🏗️  モデル作成中...")
        model = HybridMambaModel(**config['model'])
        
        # パラメータ数表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"💼 総パラメータ数: {total_params:,}")
        print(f"🎯 学習可能パラメータ数: {trainable_params:,}")
        
        # トレーナー作成
        print("\n🚀 トレーナー設定中...")
        trainer = ImprovedMambaTrainer(
            model=model,
            device=device,
            use_mixed_precision=not args.no_mixed_precision,
            use_compilation=not args.no_compile
        )
        
        # 最適化器設定
        trainer.setup_optimizer(
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            warmup_steps=config['training']['warmup_steps'],
            max_steps=config['training']['max_steps']
        )
        
        # 保存ディレクトリ作成
        os.makedirs(config['save_dir'], exist_ok=True)
        
        # 設定を保存
        config_path = os.path.join(config['save_dir'], 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 学習開始
        print("\n" + "=" * 80)
        print("🎯 学習開始")
        print("=" * 80)
        
        start_time = time.time()
        
        # 単一データローダーを辞書形式に変換
        train_dataloaders = {'main': train_dataloader}
        
        trainer.train(
            train_dataloaders=train_dataloaders,
            eval_dataloaders=None,  # 評価データがない場合
            max_steps=config['training']['max_steps'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            eval_steps=config['training']['eval_steps'],
            save_steps=config['training']['save_steps'],
            logging_steps=config['training']['logging_steps'],
            save_dir=config['save_dir'],
            max_grad_norm=config['training']['max_grad_norm'],
            label_smoothing=0.1
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("✅ 学習完了")
        print("=" * 80)
        print(f"⏰ 総学習時間: {training_time:.2f}秒 ({training_time/60:.1f}分)")
        
        # 生成テスト
        print("\n🎪 生成テスト")
        print("-" * 40)
        
        model.eval()
        test_input = dataset.processor.encode(args.test_text, add_special_tokens=True)
        test_input = torch.tensor([test_input[:50]], dtype=torch.long).to(device)  # 長さ制限
        
        with torch.no_grad():
            generated = model.generate(
                test_input,
                max_length=min(100, config['model']['max_seq_len']),
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = dataset.processor.decode(generated[0].cpu().tolist())
        print(f"入力: {args.test_text}")
        print(f"生成: {generated_text}")
        
        # 学習曲線をプロット
        print("\n📈 学習曲線を保存中...")
        trainer.plot_advanced_training_curves(config['save_dir'])
        
        print(f"\n💾 全ての結果が保存されました: {config['save_dir']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  学習が中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 プログラム終了")


if __name__ == "__main__":
    main() 