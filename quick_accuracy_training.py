#!/usr/bin/env python3
"""
精度向上のためのクイック学習スクリプト
簡単に精度向上された学習を実行できます
"""

import os
import subprocess
import sys

def main():
    print("🚀 精度向上Mamba学習システム")
    print("=" * 50)
    
    print("以下の改良点が実装されています:")
    print("✅ DropPath正則化")
    print("✅ RMSNorm（LayerNormより効率的）")
    print("✅ アダプティブ損失重み付け")
    print("✅ データ拡張（ノイズ・マスキング）")
    print("✅ 言語バランス調整")
    print("✅ 改良されたMambaブロック")
    print("✅ SwiGLU活性化")
    print("✅ レイヤースケール")
    print("✅ 動的バッチサイズ調整")
    print("✅ Mixed Precision学習")
    print("✅ モデルコンパイル")
    print("")
    
    # デフォルト設定
    default_config = {
        "d_model": 512,
        "n_layers": 8,
        "num_experts": 8,
        "max_seq_len": 512,
        "batch_size": 16,
        "max_steps": 5000,
        "learning_rate": 5e-4
    }
    
    print("📊 推奨設定:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")
    print("")
    
    response = input("この設定で学習を開始しますか？ (y/n): ").lower().strip()
    
    if response != 'y':
        print("学習をキャンセルしました。")
        return
    
    # サンプルデータ作成
    print("\n📁 サンプルデータを作成中...")
    
    # 学習実行
    cmd = [
        sys.executable, "run_improved_training.py",
        "--create_sample",
        "--d_model", str(default_config["d_model"]),
        "--n_layers", str(default_config["n_layers"]),
        "--num_experts", str(default_config["num_experts"]),
        "--max_seq_len", str(default_config["max_seq_len"]),
        "--batch_size", str(default_config["batch_size"]),
        "--max_steps", str(default_config["max_steps"]),
        "--learning_rate", str(default_config["learning_rate"]),
        "--experiment_name", "accuracy_improved"
    ]
    
    print(f"\n🎯 実行コマンド:")
    print(" ".join(cmd))
    print("")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 学習が完了しました！")
        print("📈 学習曲線とチェックポイントは 'checkpoints' ディレクトリに保存されています。")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 学習中にエラーが発生しました: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  学習が中断されました")

if __name__ == "__main__":
    main() 