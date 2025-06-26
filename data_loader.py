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
    
    def __init__(self, vocab_size: int = 2000):
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
        # バッチが辞書のリストであることを確認
        if not isinstance(batch, list) or not all(isinstance(item, dict) for item in batch):
            raise TypeError(f"バッチは辞書のリストである必要があります。現在の型: {type(batch)}")
        
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
    """大幅に強化されたサンプルデータ作成"""
    os.makedirs(data_dir, exist_ok=True)
    
    # 大幅に増強された日本語サンプル（技術・学術・日常会話など多様なジャンル）
    japanese_samples = [
        # 機械学習・AI関連
        "機械学習は人工知能の一分野であり、データから自動的にパターンを学習する技術です。大量のデータを用いて統計的なモデルを構築し、予測や分類を行います。",
        "深層学習はニューラルネットワークを多層に積み重ねた手法で、画像認識や自然言語処理に優れた性能を発揮します。畳み込みニューラルネットワークや再帰型ニューラルネットワークが代表的です。",
        "トランスフォーマーアーキテクチャは自然言語処理において革新的な進歩をもたらしました。アテンション機構により長距離依存関係を効率的に学習できます。",
        "Mambaモデルは状態空間モデルに基づく新しいアーキテクチャで、長いシーケンスの処理に効率的です。計算複雑度が線形であることが大きな特徴です。",
        "データサイエンスでは、データの収集、前処理、分析、可視化が重要なプロセスです。統計的手法と機械学習を組み合わせて知見を抽出します。",
        "強化学習では、エージェントが環境との相互作用を通じて最適な行動を学習します。報酬信号を最大化するような政策を学習することが目標です。",
        "クロスバリデーションは機械学習モデルの性能を評価する重要な手法の一つです。データを複数の部分に分割して訓練と検証を繰り返します。",
        "オーバーフィッティングを防ぐために、正則化やドロップアウトなどの技術が使用されます。汎化性能を向上させることが重要です。",
        
        # プログラミング・技術関連
        "Pythonは機械学習分野で最も人気のあるプログラミング言語です。NumPy、pandas、scikit-learnなどの豊富なライブラリが利用できます。",
        "Git は分散型バージョン管理システムで、ソフトウェア開発において欠かせないツールです。チーム開発では必須の技術となっています。",
        "データベース設計では正規化が重要な概念です。第一正規形、第二正規形、第三正規形などの段階的な正規化を行います。",
        "Web開発においてRESTful APIは標準的な設計手法です。HTTPメソッドを適切に使い分けてリソース指向の設計を行います。",
        
        # 日本文化・歴史
        "日本の四季は世界的にも美しいとされ、春の桜、夏の緑、秋の紅葉、冬の雪景色がそれぞれ独特の魅力を持っています。",
        "茶道は日本の伝統文化の一つで、おもてなしの心と精神的な修養を重視します。千利休が大成した侘び茶が特に有名です。",
        "江戸時代の鎖国政策は日本独自の文化発展をもたらしました。浮世絵や歌舞伎などの庶民文化が花開いた時代でもあります。",
        "明治維新は日本の近代化の出発点となりました。封建制度から近代国家への転換は世界史的にも注目される変化でした。",
        
        # 科学・自然
        "量子力学は物理学の基本理論の一つで、原子や電子などのミクロな世界の現象を記述します。不確定性原理が重要な概念です。",
        "地球温暖化は現代社会が直面する重要な課題です。二酸化炭素などの温室効果ガスの削減が急務となっています。",
        "生物多様性の保全は地球環境の持続可能性にとって不可欠です。生態系のバランスを維持することが重要です。",
        "再生可能エネルギーの普及が世界的に進んでいます。太陽光、風力、水力などのクリーンエネルギーが注目されています。",
        
        # 日常・社会
        "働き方改革により、テレワークやフレックスタイム制度が普及しています。ワークライフバランスの重要性が認識されています。",
        "高齢化社会において、医療・介護分野での人材不足が深刻な問題となっています。テクノロジーの活用による解決策が模索されています。",
        "教育現場でのデジタル化が急速に進んでいます。一人一台のタブレット端末配布により学習環境が大きく変化しています。",
        "都市部への人口集中が進む中、地方創生の取り組みが重要視されています。移住促進や起業支援などの施策が展開されています。"
    ]
    
    # 大幅に増強された英語サンプル（技術・学術・ビジネス・日常など多様なジャンル）
    english_samples = [
        # Machine Learning & AI
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models to enable computers to improve their performance on specific tasks through experience.",
        "Deep learning models have achieved remarkable success in computer vision and natural language processing tasks by utilizing multiple layers of neural networks to learn hierarchical representations.",
        "The transformer architecture revolutionized the field of natural language processing with attention mechanisms that allow models to focus on relevant parts of input sequences simultaneously.",
        "State space models like Mamba offer efficient alternatives to transformers for long sequence modeling with linear computational complexity and improved memory efficiency.",
        "Data preprocessing is a crucial step that significantly impacts the performance of machine learning models, including cleaning, normalization, and feature extraction processes.",
        "Regularization techniques help prevent overfitting and improve model generalization by adding penalty terms to the loss function or using dropout mechanisms.",
        "Cross-validation provides robust estimates of model performance across different data splits, helping to assess how well models generalize to unseen data.",
        "Feature engineering involves creating meaningful input variables that improve model accuracy through domain knowledge and statistical analysis of the data.",
        
        # Technology & Programming
        "Python has become the dominant programming language in data science and machine learning due to its simplicity, readability, and extensive ecosystem of libraries.",
        "Version control systems like Git enable collaborative software development by tracking changes to code and managing different versions of projects efficiently.",
        "Cloud computing platforms provide scalable infrastructure for machine learning workloads, allowing researchers and companies to access powerful computing resources on demand.",
        "DevOps practices integrate software development and operations to enable faster deployment cycles and more reliable systems through automation and monitoring.",
        "Cybersecurity has become increasingly important as digital transformation accelerates, requiring robust defenses against evolving threats and vulnerabilities.",
        "Blockchain technology offers decentralized solutions for various applications beyond cryptocurrency, including supply chain management and digital identity verification.",
        
        # Science & Research
        "Quantum computing represents a paradigm shift in computational power, utilizing quantum mechanical phenomena to solve certain problems exponentially faster than classical computers.",
        "Climate change research relies heavily on large-scale data analysis and predictive modeling to understand complex environmental systems and project future scenarios.",
        "Biotechnology advances are revolutionizing medicine through personalized treatments, gene therapy, and precision medicine approaches tailored to individual genetic profiles.",
        "Space exploration has entered a new era with private companies collaborating with government agencies to develop reusable rockets and establish permanent human presence beyond Earth.",
        "Renewable energy technologies are becoming increasingly cost-competitive with fossil fuels, driving global transitions toward sustainable energy systems.",
        
        # Business & Economics
        "Digital transformation is reshaping traditional business models across industries, requiring organizations to adapt to new technologies and changing customer expectations.",
        "E-commerce platforms have fundamentally changed retail landscapes, offering consumers unprecedented convenience while creating new challenges for traditional brick-and-mortar stores.",
        "Artificial intelligence is being integrated into business processes to automate routine tasks, enhance decision-making, and provide personalized customer experiences.",
        "Supply chain optimization has become critical for global businesses, especially in the wake of recent disruptions that highlighted the importance of resilience and flexibility.",
        "Financial technology innovations are democratizing access to financial services through mobile banking, peer-to-peer lending, and digital payment solutions.",
        
        # Society & Culture
        "Social media platforms have transformed how people communicate, share information, and form communities, while also raising concerns about privacy and misinformation.",
        "Remote work has become mainstream following global shifts in workplace culture, offering flexibility while presenting new challenges for collaboration and management.",
        "Educational technology is revolutionizing learning experiences through personalized curricula, virtual classrooms, and adaptive learning systems that cater to individual needs.",
        "Urban planning increasingly incorporates smart city technologies to improve efficiency, sustainability, and quality of life for growing metropolitan populations.",
        "Healthcare systems worldwide are adopting digital health solutions, including telemedicine, wearable devices, and AI-assisted diagnostics to improve patient outcomes."
    ]
    
    # 大幅に増強されたPythonサンプル（機械学習、データ分析、Web開発など）
    python_samples = [
        '''
# Advanced PyTorch Model with Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(output)
''',
        '''
# Data Science Pipeline with Pandas and Scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, df, target_column):
        # Handle missing values
        df_clean = df.dropna()
        
        # Separate features and target
        X = df_clean.drop(target_column, axis=1)
        y = df_clean[target_column]
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = self.label_encoder.fit_transform(X[col])
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
        
    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return y_test, y_pred
''',
        '''
# Advanced Web API with FastAPI
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import sqlite3
import hashlib

app = FastAPI(title="ML Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    timestamp: datetime

class MLModelService:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        # Load pre-trained models
        pass
        
    def predict(self, features: List[float], model_version: str = "latest"):
        if model_version not in self.models:
            raise HTTPException(status_code=404, detail="Model version not found")
            
        # Make prediction
        prediction = sum(features) / len(features)  # Dummy prediction
        confidence = 0.85
        
        return prediction, confidence

ml_service = MLModelService()

@app.get("/")
async def root():
    return {"message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        prediction, confidence = ml_service.predict(request.features, request.model_version)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=request.model_version,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return {"available_models": list(ml_service.models.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
        '''
# Advanced Data Processing and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns
        
    def exploratory_analysis(self):
        print("Dataset Shape:", self.df.shape)
        print("\\nMissing Values:")
        print(self.df.isnull().sum())
        print("\\nNumerical Statistics:")
        print(self.df[self.numerical_columns].describe())
        
    def create_visualizations(self):
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution Plot', 'Correlation Heatmap', 
                          'Box Plot', 'Scatter Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Distribution plot
        if len(self.numerical_columns) > 0:
            col = self.numerical_columns[0]
            fig.add_trace(go.Histogram(x=self.df[col], name=col), row=1, col=1)
            
        # Correlation heatmap data
        corr_matrix = self.df[self.numerical_columns].corr()
        
        # Box plot
        if len(self.numerical_columns) > 1:
            fig.add_trace(go.Box(y=self.df[self.numerical_columns[1]], 
                               name=self.numerical_columns[1]), row=2, col=1)
            
        # Scatter plot
        if len(self.numerical_columns) >= 2:
            fig.add_trace(go.Scatter(x=self.df[self.numerical_columns[0]], 
                                   y=self.df[self.numerical_columns[1]],
                                   mode='markers', name='Scatter'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Data Analysis Dashboard")
        fig.show()
        
    def statistical_tests(self):
        results = {}
        
        for col in self.numerical_columns:
            # Normality test
            stat, p_value = stats.normaltest(self.df[col].dropna())
            results[f'{col}_normality'] = {'statistic': stat, 'p_value': p_value}
            
        return results
''',
        '''
# Advanced Class Design with Decorators and Context Managers
import functools
import time
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List

def timing_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def retry_decorator(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def processing_context(self, operation_name: str):
        self.logger.info(f"Starting {operation_name}")
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {e}")
            raise
        finally:
            self.logger.info(f"Completed {operation_name}")
            
    @timing_decorator
    @retry_decorator(max_attempts=3)
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self.processing_context("data_processing"):
            processed_data = []
            
            for item in data:
                processed_item = self._transform_item(item)
                if self._validate_item(processed_item):
                    processed_data.append(processed_item)
                    
            return processed_data
            
    def _transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Apply transformations based on config
        transformed = item.copy()
        
        for key, value in item.items():
            if key in self.config.get('normalize_fields', []):
                transformed[key] = self._normalize_value(value)
                
        return transformed
        
    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().strip()
        elif isinstance(value, (int, float)):
            return (value - self.config.get('mean', 0)) / self.config.get('std', 1)
        return value
        
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        required_fields = self.config.get('required_fields', [])
        return all(field in item and item[field] is not None for field in required_fields)
'''
    ]
    
    # 新しいカテゴリ：技術ドキュメント
    technical_docs = [
        "API Documentation: The RESTful API follows standard HTTP methods and status codes. GET requests retrieve data, POST creates new resources, PUT updates existing resources, and DELETE removes resources.",
        "System Architecture: The microservices architecture consists of loosely coupled services that communicate through well-defined APIs. Each service has its own database and can be deployed independently.",
        "Performance Optimization: Database query optimization involves proper indexing, query plan analysis, and avoiding N+1 problems. Use connection pooling and caching strategies for better performance.",
        "Security Guidelines: Implement authentication using JWT tokens, authorize access based on user roles, and sanitize all user inputs to prevent SQL injection and XSS attacks.",
        "Deployment Process: Use containerization with Docker for consistent environments. Implement CI/CD pipelines with automated testing, staging deployments, and blue-green production releases.",
        "Monitoring and Logging: Set up comprehensive logging with structured formats. Monitor system metrics, application performance, and business KPIs using tools like Prometheus and Grafana."
    ]
    
    # 新しいカテゴリ：学術・研究風テキスト
    academic_samples = [
        "Abstract: This study investigates the application of deep learning techniques to natural language processing tasks. We propose a novel architecture that combines attention mechanisms with recurrent neural networks.",
        "Methodology: The experimental design includes a control group and treatment group. Data collection follows ethical guidelines with informed consent from all participants.",
        "Results: Statistical analysis reveals a significant correlation between variables X and Y (p < 0.05). The confidence interval provides robust evidence for the proposed hypothesis.",
        "Discussion: The findings suggest that machine learning models can effectively predict outcomes with 85% accuracy. However, limitations include dataset size and potential bias in the training data.",
        "Conclusion: Future research should explore larger datasets and more diverse populations. The proposed framework demonstrates promising results for real-world applications."
    ]
    
    # 多言語混合テキスト
    multilingual_samples = [
        "機械学習における cross-validation は重要な技術です。This technique helps evaluate model performance across different data splits.",
        "Deep learning models like transformers have revolutionized NLP. 自然言語処理の分野では transformer アーキテクチャが大きな変革をもたらしました。",
        "Python libraries such as pandas and numpy are essential for data science. データサイエンスでは pandas や numpy といったライブラリが必須です。"
    ]
    
    # より多様なファイルに保存
    file_data = {
        "japanese.txt": japanese_samples,
        "english.txt": english_samples,
        "sample_code.py": python_samples,
        "technical_docs.txt": technical_docs,
        "academic_papers.txt": academic_samples,
        "multilingual.txt": multilingual_samples
    }
    
    total_texts = 0
    for filename, samples in file_data.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(samples))
        total_texts += len(samples)
        print(f"📄 {filename}: {len(samples)} samples")
    
    print(f"📁 大幅に強化されたサンプルデータを作成しました: {data_dir}")
    print(f"📊 総テキスト数: {total_texts} (前回の9テキストから大幅増加)")