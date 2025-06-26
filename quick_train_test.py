#!/usr/bin/env python3
"""
ハイブリッドモデルの簡単な訓練テスト
"""
import torch
import torch.optim as optim
import time
import json
from pathlib import Path

def create_simple_dataset():
    """簡単なテスト用データセット作成"""
    # 小さな語彙での簡単な文章
    sentences = [
        "hello world this is a test",
        "machine learning is amazing",
        "deep learning with transformers",
        "attention is all you need",
        "mamba model is efficient"
    ]
    
    # 語彙作成
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence.split())
    vocab = sorted(list(vocab))
    vocab_size = len(vocab)
    
    # トークン化
    word_to_id = {word: i for i, word in enumerate(vocab)}
    
    tokenized_data = []
    for sentence in sentences:
        tokens = [word_to_id[word] for word in sentence.split()]
        tokenized_data.append(tokens)
    
    return tokenized_data, vocab_size, word_to_id

def simple_training_test():
    """簡単な訓練テスト"""
    print("🚀 ハイブリッドモデル簡単訓練テスト")
    
    # データ準備
    data, vocab_size, word_to_id = create_simple_dataset()
    print(f"語彙サイズ: {vocab_size}")
    print(f"データサンプル数: {len(data)}")
    
    try:
        from hybrid_mamba_model import HybridMambaModel
        
        # 小さなモデル設定
        model = HybridMambaModel(
            vocab_size=vocab_size,
            d_model=64,        # 小さく
            n_layers=2,        # 少なく
            n_heads=4,         # 少なく
            num_experts=2,     # 少なく
            max_seq_len=32,
            dropout=0.1
        )
        
        model_info = model.get_model_info()
        print(f"モデルパラメータ数: {model_info['total_parameters']:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        # 訓練データ準備
        max_len = 8
        train_data = []
        for tokens in data:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [0] * (max_len - len(tokens))  # パディング
            train_data.append(tokens)
        
        # 訓練ループ
        model.train()
        losses = []
        
        print(f"\n📚 訓練開始 (10ステップ)")
        for step in range(10):
            total_loss = 0
            for tokens in train_data:
                input_ids = torch.tensor([tokens], dtype=torch.long)
                labels = input_ids.clone()
                
                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            losses.append(avg_loss)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            print(f"ステップ {step+1}: 損失={avg_loss:.4f}, PPL={perplexity:.2f}")
        
        # 生成テスト
        print(f"\n🎯 生成テスト")
        model.eval()
        test_input = torch.tensor([[word_to_id.get('hello', 0)]], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(test_input, max_length=8, temperature=0.8)
        
        # 逆トークン化
        id_to_word = {i: word for word, i in word_to_id.items()}
        generated_words = [id_to_word.get(token.item(), '<UNK>') for token in generated[0]]
        print(f"生成文: {' '.join(generated_words)}")
        
        # 結果まとめ
        results = {
            'model_type': 'HybridMambaModel',
            'vocab_size': vocab_size,
            'model_params': model_info['total_parameters'],
            'training_steps': 10,
            'final_loss': losses[-1],
            'final_perplexity': torch.exp(torch.tensor(losses[-1])).item(),
            'loss_reduction': (losses[0] - losses[-1]) / losses[0] * 100,
            'generated_text': ' '.join(generated_words)
        }
        
        print(f"\n📊 結果サマリー:")
        print(f"初期損失: {losses[0]:.4f} → 最終損失: {losses[-1]:.4f}")
        print(f"損失減少率: {results['loss_reduction']:.1f}%")
        print(f"最終パープレキシティ: {results['final_perplexity']:.2f}")
        
        # 結果保存
        with open('hybrid_training_test.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = simple_training_test()
    if results:
        print(f"\n✅ テスト完了！")
    else:
        print(f"\n❌ テスト失敗")