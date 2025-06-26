#!/usr/bin/env python3
"""
Hybrid Mamba Model の基本テスト
"""
import torch
import sys
import traceback

try:
    from hybrid_mamba_model import HybridMambaModel
    print("✓ HybridMambaModelのインポート成功")
    
    # 基本的な設定
    vocab_size = 1000
    d_model = 128
    n_layers = 2
    n_heads = 4
    num_experts = 2
    max_seq_len = 64
    
    print(f"モデル設定:")
    print(f"  語彙サイズ: {vocab_size}")
    print(f"  モデル次元: {d_model}")
    print(f"  レイヤー数: {n_layers}")
    print(f"  ヘッド数: {n_heads}")
    print(f"  エキスパート数: {num_experts}")
    print(f"  最大シーケンス長: {max_seq_len}")
    
    # モデルのインスタンス化
    model = HybridMambaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    print("✓ HybridMambaModelのインスタンス化成功")
    
    # モデル情報の表示
    model_info = model.get_model_info()
    print(f"\nモデル情報:")
    print(f"  総パラメータ数: {model_info['total_parameters']:,}")
    print(f"  学習可能パラメータ数: {model_info['trainable_parameters']:,}")
    print(f"  モデルタイプ: {model_info['model_type']}")
    
    # 簡単な前向き計算テスト
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n前向き計算テスト:")
    print(f"  入力形状: {input_ids.shape}")
    
    # 評価モード
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        
    print(f"✓ 前向き計算成功")
    print(f"  出力logits形状: {outputs['logits'].shape}")
    print(f"  損失: {outputs['loss']:.4f}")
    print(f"  負荷バランス損失: {outputs['balance_loss']:.4f}")
    
    # 生成テスト
    print(f"\n生成テスト:")
    test_input = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(test_input, max_length=20, temperature=1.0)
    print(f"  入力長: {test_input.size(1)}")
    print(f"  生成長: {generated.size(1)}")
    print(f"✓ 生成テスト成功")
    
    print(f"\n🎉 すべてのテストが成功しました！")
    print(f"ハイブリッドモデル (Mamba × MoE × Transformer) は正常に動作しています。")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    print(f"詳細:")
    traceback.print_exc()
    sys.exit(1)