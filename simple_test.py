#!/usr/bin/env python3
"""
簡単なモデル比較テスト
"""
import torch
import torch.nn.functional as F
import time
import json

def test_model_basic(model_class, model_name, **kwargs):
    """基本的なモデルテスト"""
    print(f"\n=== {model_name} テスト ===")
    
    try:
        # モデル作成
        model = model_class(**kwargs)
        
        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"パラメータ数: {total_params:,}")
        
        # 小さなテストデータ
        batch_size = 2
        seq_len = 16
        vocab_size = kwargs.get('vocab_size', 100)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 推論速度テスト
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # 10回実行
                outputs = model(input_ids, labels=labels)
        
        inference_time = (time.time() - start_time) / 10
        print(f"推論時間: {inference_time:.4f}秒")
        
        # 損失確認
        loss = outputs['loss'].item()
        print(f"初期損失: {loss:.4f}")
        print(f"初期パープレキシティ: {torch.exp(outputs['loss']):.2f}")
        
        # 生成テスト
        test_input = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(test_input, max_length=15, temperature=1.0)
        print(f"生成長: {test_input.size(1)} → {generated.size(1)}")
        
        return {
            'model_name': model_name,
            'params': total_params,
            'inference_time': inference_time,
            'initial_loss': loss,
            'initial_perplexity': torch.exp(outputs['loss']).item()
        }
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def main():
    print("🔍 モデル性能比較テスト")
    
    # 共通設定
    config = {
        'vocab_size': 100,
        'd_model': 128,
        'n_layers': 2,
        'max_seq_len': 64
    }
    
    results = []
    
    # 元のMambaモデルテスト
    try:
        from mamba_model import MambaModel
        result = test_model_basic(MambaModel, "Original Mamba", **config)
        if result:
            results.append(result)
    except ImportError as e:
        print(f"Original Mambaモデルをインポートできません: {e}")
    
    # ハイブリッドモデルテスト
    try:
        from hybrid_mamba_model import HybridMambaModel
        hybrid_config = {**config, 'n_heads': 4, 'num_experts': 2}
        result = test_model_basic(HybridMambaModel, "Hybrid Mamba (Mamba+MoE+Transformer)", **hybrid_config)
        if result:
            results.append(result)
    except ImportError as e:
        print(f"Hybrid Mambaモデルをインポートできません: {e}")
    
    # 結果比較
    if len(results) >= 2:
        print(f"\n📊 性能比較結果")
        print("=" * 60)
        
        orig = results[0]
        hybrid = results[1]
        
        print(f"モデル               | パラメータ数    | 推論時間(秒) | 初期損失  | 初期PPL")
        print("-" * 60)
        print(f"{orig['model_name']:<20} | {orig['params']:>10,} | {orig['inference_time']:>8.4f} | {orig['initial_loss']:>7.4f} | {orig['initial_perplexity']:>7.2f}")
        print(f"{hybrid['model_name']:<20} | {hybrid['params']:>10,} | {hybrid['inference_time']:>8.4f} | {hybrid['initial_loss']:>7.4f} | {hybrid['initial_perplexity']:>7.2f}")
        
        # 改善率計算
        param_ratio = hybrid['params'] / orig['params']
        time_ratio = hybrid['inference_time'] / orig['inference_time']
        
        print(f"\n📈 比較分析:")
        print(f"パラメータ数比: {param_ratio:.2f}x ({param_ratio-1:+.1%})")
        print(f"推論時間比: {time_ratio:.2f}x ({time_ratio-1:+.1%})")
        print(f"※初期損失とパープレキシティは未訓練状態での値")
        
    # 結果保存
    with open('model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ テスト完了！結果はmodel_comparison.jsonに保存されました")

if __name__ == "__main__":
    main()