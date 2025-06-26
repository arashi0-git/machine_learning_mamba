import os
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from typing import List, Dict, Tuple, Optional, Union
import json
import random
import numpy as np
from torch.utils.data import WeightedRandomSampler
import unicodedata


class AdvancedTextProcessor:
    """精度向上のための高度なテキスト処理"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.char_freqs = Counter()
        
        # 特殊トークン
        self.special_tokens = {
            '<pad>': 0,
            '<eos>': 1, 
            '<bos>': 2,
            '<unk>': 3,
            '<mask>': 4,
            '<sep>': 5,
        }
        
        # 言語検出用パターン
        self.language_patterns = {
            'japanese': re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'),
            'english': re.compile(r'[a-zA-Z]'),
            'python': re.compile(r'(def |class |import |from |if |for |while |try |except)')
        }
        
    def normalize_text(self, text: str) -> str:
        """テキスト正規化（精度向上のため）"""
        # Unicode正規化
        text = unicodedata.normalize('NFKC', text)
        
        # 空白文字の統一
        text = re.sub(r'\s+', ' ', text)
        
        # 制御文字の除去
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """言語検出"""
        scores = {}
        for lang, pattern in self.language_patterns.items():
            matches = len(pattern.findall(text))
            scores[lang] = matches / len(text) if text else 0
        
        return max(scores, key=scores.get) if scores else 'unknown'
    
    def build_vocab(self, texts: List[str]) -> None:
        """改良された語彙構築"""
        # 文字頻度をカウント
        for text in texts:
            normalized_text = self.normalize_text(text)
            self.char_freqs.update(normalized_text)
        
        # 特殊トークンを最初に追加
        self.char_to_id.update(self.special_tokens)
        self.id_to_char.update({v: k for k, v in self.special_tokens.items()})
        
        # 頻度順でソート
        sorted_chars = sorted(self.char_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # 語彙サイズまで追加
        current_id = len(self.special_tokens)
        for char, freq in sorted_chars:
            if current_id >= self.vocab_size:
                break
            if char not in self.char_to_id:
                self.char_to_id[char] = current_id
                self.id_to_char[current_id] = char
                current_id += 1
        
        print(f"🔤 語彙構築完了: {len(self.char_to_id)} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """テキストをトークンIDに変換"""
        text = self.normalize_text(text)
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        
        for char in text:
            tokens.append(self.char_to_id.get(char, self.special_tokens['<unk>']))
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
            
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """トークンIDをテキストに変換"""
        chars = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, '<unk>')
            if char not in ['<pad>', '<bos>', '<eos>']:
                chars.append(char)
        return ''.join(chars)


class DataAugmentation:
    """精度向上のためのデータ拡張"""
    
    def __init__(self, noise_prob: float = 0.1, mask_prob: float = 0.15):
        self.noise_prob = noise_prob
        self.mask_prob = mask_prob
    
    def add_noise(self, text: str, processor: AdvancedTextProcessor) -> str:
        """ノイズ追加"""
        if random.random() > self.noise_prob:
            return text
            
        chars = list(text)
        num_noise = max(1, int(len(chars) * 0.05))  # 5%の文字にノイズ
        
        for _ in range(num_noise):
            if not chars:
                break
                
            pos = random.randint(0, len(chars) - 1)
            operation = random.choice(['replace', 'delete', 'insert'])
            
            if operation == 'replace' and chars:
                # ランダムな文字で置換
                random_char = random.choice(list(processor.char_to_id.keys()))
                chars[pos] = random_char
            elif operation == 'delete' and len(chars) > 1:
                # 削除
                chars.pop(pos)
            elif operation == 'insert':
                # 挿入
                random_char = random.choice(list(processor.char_to_id.keys()))
                chars.insert(pos, random_char)
        
        return ''.join(chars)
    
    def mask_tokens(self, token_ids: List[int], mask_token_id: int) -> Tuple[List[int], List[int]]:
        """トークンマスキング（BERT style）"""
        if len(token_ids) < 2:
            return token_ids, token_ids
            
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)  # -100 = ignore in loss
        
        # マスク対象を選択
        mask_indices = []
        for i, token_id in enumerate(token_ids):
            if token_id not in [0, 1, 2] and random.random() < self.mask_prob:  # 特殊トークンを除く
                mask_indices.append(i)
        
        for i in mask_indices:
            labels[i] = token_ids[i]  # 正解ラベル
            
            prob = random.random()
            if prob < 0.8:
                # 80%の確率でマスクトークンに置換
                masked_ids[i] = mask_token_id
            elif prob < 0.9:
                # 10%の確率でランダムトークンに置換
                masked_ids[i] = random.randint(6, len(masked_ids) - 1)
            # 10%の確率でそのまま（何もしない）
        
        return masked_ids, labels


class ImprovedTextDataset(Dataset):
    """精度向上のための改良されたデータセット"""
    
    def __init__(self, data_dir: str, max_length: int = 512, 
                 use_augmentation: bool = True, balance_languages: bool = True):
        self.data_dir = data_dir
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        self.balance_languages = balance_languages
        
        # プロセッサーと拡張器
        self.processor = AdvancedTextProcessor()
        self.augmenter = DataAugmentation() if use_augmentation else None
        
        # データ読み込み
        self.texts, self.labels, self.languages = self._load_data()
        
        # 語彙構築
        self.processor.build_vocab(self.texts)
        
        # 言語バランス調整用の重み計算
        self.sample_weights = self._compute_sample_weights() if balance_languages else None
        
        print(f"📊 データセット統計:")
        print(f"  - 総テキスト数: {len(self.texts)}")
        print(f"  - 言語分布: {Counter(self.languages)}")
        print(f"  - 平均長: {np.mean([len(text) for text in self.texts]):.1f}")
        
    def _load_data(self) -> Tuple[List[str], List[str], List[str]]:
        """データ読み込み"""
        texts = []
        labels = []
        languages = []
        
        if not os.path.exists(self.data_dir):
            print(f"⚠️  データディレクトリが見つかりません: {self.data_dir}")
            return [], [], []
        
        # ファイル別にデータを読み込み
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.isfile(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # ファイル内容を適切な長さに分割
                chunks = self._split_text(content)
                
                for chunk in chunks:
                    if len(chunk.strip()) < 10:  # 短すぎるテキストをスキップ
                        continue
                        
                    language = self.processor.detect_language(chunk)
                    
                    texts.append(chunk)
                    labels.append(filename)
                    languages.append(language)
                    
            except Exception as e:
                print(f"⚠️  ファイル読み込みエラー: {filepath} - {e}")
                continue
        
        return texts, labels, languages
    
    def _split_text(self, text: str, overlap: int = 50) -> List[str]:
        """テキストを適切な長さに分割（オーバーラップ付き）"""
        if len(text) <= self.max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_length
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 文境界で分割を試行
            chunk = text[start:end]
            last_period = chunk.rfind('。')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            split_pos = max(last_period, last_newline, last_space)
            
            if split_pos > start + self.max_length // 2:
                end = start + split_pos + 1
            
            chunks.append(text[start:end])
            start = end - overlap  # オーバーラップ
        
        return chunks
    
    def _compute_sample_weights(self) -> List[float]:
        """言語バランス調整用の重み計算"""
        language_counts = Counter(self.languages)
        total_samples = len(self.languages)
        
        # 各言語の重みを計算（逆頻度重み付け）
        language_weights = {}
        for lang, count in language_counts.items():
            language_weights[lang] = total_samples / (len(language_counts) * count)
        
        # サンプルごとの重みを計算
        weights = [language_weights[lang] for lang in self.languages]
        return weights
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        language = self.languages[idx]
        
        # データ拡張（訓練時のみ）
        if self.augmenter and self.use_augmentation and random.random() < 0.3:
            text = self.augmenter.add_noise(text, self.processor)
        
        # トークン化
        token_ids = self.processor.encode(text)
        
        # 長さ調整
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            # パディング
            token_ids += [0] * (self.max_length - len(token_ids))
        
        # マスキング（確率的に適用）
        masked_ids = token_ids.copy()
        if self.augmenter and random.random() < 0.2:  # 20%の確率
            masked_ids, _ = self.augmenter.mask_tokens(token_ids, self.processor.special_tokens['<mask>'])
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long),  # 自己教師学習
            'language': language,
            'length': len([t for t in token_ids if t != 0])  # 実際の長さ
        }


def create_balanced_dataloader(dataset: ImprovedTextDataset, batch_size: int = 16, 
                             use_weighted_sampling: bool = True) -> DataLoader:
    """バランス調整されたデータローダー作成"""
    
    if use_weighted_sampling and dataset.sample_weights:
        # 重み付きサンプリング
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # 動的バッチ処理用のcollate関数
    def collate_fn(batch):
        # 長さでソート（効率的なパディングのため）
        batch = sorted(batch, key=lambda x: x['length'], reverse=True)
        
        # バッチ内の最大長に合わせてパディング調整
        max_len = max(item['length'] for item in batch)
        max_len = min(max_len, dataset.max_length)  # 最大長制限
        
        input_ids = []
        labels = []
        languages = []
        lengths = []
        
        for item in batch:
            # 必要に応じてトランケート
            input_id = item['input_ids'][:max_len]
            label = item['labels'][:max_len]
            
            # パディング
            if len(input_id) < max_len:
                padding_size = max_len - len(input_id)
                input_id = torch.cat([input_id, torch.zeros(padding_size, dtype=torch.long)])
                label = torch.cat([label, torch.full((padding_size,), -100, dtype=torch.long)])
            
            input_ids.append(input_id)
            labels.append(label)
            languages.append(item['language'])
            lengths.append(item['length'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'languages': languages,
            'lengths': torch.tensor(lengths)
        }
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Windows互換性のため0に設定
        pin_memory=False  # CPU使用時はFalseに設定
    )


def create_sample_data(data_dir: str = "sample_data"):
    """改良されたサンプルデータ作成"""
    os.makedirs(data_dir, exist_ok=True)
    
    # より豊富な日本語サンプル
    japanese_samples = [
        "機械学習は人工知能の一分野であり、データから自動的にパターンを学習する技術です。",
        "深層学習はニューラルネットワークを多層に積み重ねた手法で、画像認識や自然言語処理に優れた性能を発揮します。",
        "トランスフォーマーアーキテクチャは自然言語処理において革新的な進歩をもたらしました。",
        "Mambaモデルは状態空間モデルに基づく新しいアーキテクチャで、長いシーケンスの処理に効率的です。",
        "データサイエンスでは、データの収集、前処理、分析、可視化が重要なプロセスです。",
        "強化学習では、エージェントが環境との相互作用を通じて最適な行動を学習します。",
        "クロスバリデーションは機械学習モデルの性能を評価する重要な手法の一つです。",
        "オーバーフィッティングを防ぐために、正則化やドロップアウトなどの技術が使用されます。"
    ]
    
    # より豊富な英語サンプル
    english_samples = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Deep learning models have achieved remarkable success in computer vision and natural language processing tasks.",
        "The transformer architecture revolutionized the field of natural language processing with attention mechanisms.",
        "State space models like Mamba offer efficient alternatives to transformers for long sequence modeling.",
        "Data preprocessing is a crucial step that significantly impacts the performance of machine learning models.",
        "Regularization techniques help prevent overfitting and improve model generalization.",
        "Cross-validation provides robust estimates of model performance across different data splits.",
        "Feature engineering involves creating meaningful input variables that improve model accuracy."
    ]
    
    # より豊富なPythonサンプル
    python_samples = [
        '''
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
''',
        '''
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
''',
        '''
class DataProcessor:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
    
    def build_vocab(self, texts):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(sorted_words[:self.vocab_size]):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
'''
    ]
    
    # ファイルに保存
    with open(os.path.join(data_dir, "japanese.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(japanese_samples))
    
    with open(os.path.join(data_dir, "english.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(english_samples))
    
    with open(os.path.join(data_dir, "sample_code.py"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(python_samples))
    
    print(f"📁 改良されたサンプルデータを作成しました: {data_dir}")