#!/usr/bin/env python3
import os
import argparse
import torch
import json
from pathlib import Path

from hybrid_mamba_model import HybridMambaModel
from data_loader import ImprovedTextDataset, create_balanced_dataloader
from trainer import MambaTrainer


def create_sample_data():
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    japanese_text = """
こんにちは、世界！これは日本語のサンプルテキストです。
機械学習は人工知能の重要な分野です。
深層学習とニューラルネットワークが注目されています。
自然言語処理は言語理解の技術です。
"""
    
    english_text = """
Hello world! This is a sample English text.
Machine learning is an important field of artificial intelligence.
Deep learning and neural networks are getting attention.
Natural language processing is a technology for language understanding.
"""
    
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y

print("Hello, Python!")
for i in range(5):
    print(f"Fibonacci({i}) = {fibonacci(i)}")
"""
    
    with open(sample_dir / "japanese.txt", "w", encoding="utf-8") as f:
        f.write(japanese_text)
    
    with open(sample_dir / "english.txt", "w", encoding="utf-8") as f:
        f.write(english_text)
    
    with open(sample_dir / "sample_code.py", "w", encoding="utf-8") as f:
        f.write(python_code)
    
    print(f"サンプルデータを作成しました: {sample_dir}")


def main():
    parser = argparse.ArgumentParser(description="Mamba機械学習システム")
    parser.add_argument("--data_dir", type=str, default="sample_data", 
                       help="学習データディレクトリ")
    parser.add_argument("--epochs", type=int, default=15, 
                       help="エポック数")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="バッチサイズ")
    parser.add_argument("--max_length", type=int, default=128, 
                       help="最大シーケンス長")
    parser.add_argument("--d_model", type=int, default=128, 
                       help="モデルの次元数")
    parser.add_argument("--n_layers", type=int, default=2, 
                       help="レイヤー数")
    parser.add_argument("--n_heads", type=int, default=8, 
                       help="アテンションヘッド数")
    parser.add_argument("--num_experts", type=int, default=4, 
                       help="MoEエキスパート数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, 
                       help="学習率")
    parser.add_argument("--create_sample", action="store_true", 
                       help="サンプルデータを作成")
    parser.add_argument("--device", type=str, default="auto", 
                       help="使用デバイス (cpu/cuda/auto)")
    parser.add_argument("--test_text", type=str, default="機械学習", 
                       help="テスト用テキスト")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("Mamba機械学習システム")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"データディレクトリ: {args.data_dir}")
    print(f"エポック数: {args.epochs}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"モデル次元: {args.d_model}")
    print(f"レイヤー数: {args.n_layers}")
    print(f"アテンションヘッド数: {args.n_heads}")
    print(f"MoEエキスパート数: {args.num_experts}")
    print("-" * 80)
    
    if not os.path.exists(args.data_dir):
        print(f"エラー: データディレクトリが見つかりません: {args.data_dir}")
        print("--create_sample オプションでサンプルデータを作成できます")
        return
    
    print("ステップ 1: データセットの準備")
    dataset = ImprovedTextDataset(args.data_dir, max_length=args.max_length)
    dataloader = create_balanced_dataloader(dataset, batch_size=args.batch_size)
    
    vocab_size = dataset.processor.vocab_size
    
    print(f"  総テキスト数: {len(dataset)}")
    print(f"  言語分布: {dataset.languages}")
    print(f"  語彙サイズ: {vocab_size}")
    
    print("\nステップ 3: モデルの初期化")
    model = HybridMambaModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_experts=args.num_experts,
        max_seq_len=args.max_length
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  総パラメータ数: {total_params:,}")
    print(f"  学習対象パラメータ数: {trainable_params:,}")
    
    print("\nステップ 4: 訓練の開始")
    trainer = MambaTrainer(model, device=device, processor=dataset.processor)
    trainer.setup_training(learning_rate=args.learning_rate)
    
    try:
        trainer.train(dataloader, epochs=args.epochs)
        
        print("\nステップ 5: モデルのテスト")
        test_output = trainer.evaluate_model(args.test_text, max_length=30)
        print(f"入力: {args.test_text}")
        print(f"生成: {test_output}")
        
        print("\n" + "=" * 80)
        print("訓練完了！")
        
        final_metrics = trainer.get_final_metrics()
        print(f"最終メトリクス:")
        print(f"  損失: {final_metrics['final_loss']:.4f}")
        print(f"  パープレキシティ: {final_metrics['final_perplexity']:.2f}")
        print(f"  総時間: {final_metrics['total_time']:.2f}秒")
        
        # resultsディレクトリを作成
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "training_results.json"
        results = {
            "model_config": {
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "num_experts": args.num_experts,
                "total_params": total_params
            },
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length
            },
            "final_metrics": final_metrics,
            "data_stats": {"languages": dataset.languages}
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"結果を保存しました: {results_file}")
        
    except KeyboardInterrupt:
        print("\n訓練が中断されました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()