# Mamba機械学習システム

Mambaアーキテクチャを使用した多言語対応機械学習システムです。日本語、英語、Python言語の学習が可能で、学習進捗と精度をリアルタイムで確認できます。

## 概要

このプロジェクトは、State Space Model（SSM）ベースのMambaアーキテクチャを実装した言語モデルです。Transformerの代替として注目されているMambaは、長系列処理において効率的な性能を発揮します。

## 主要技術

### Mambaアーキテクチャ
- **State Space Models (SSM)**: 長期依存関係を効率的に学習
- **選択的状態更新**: 重要な情報のみを選択的に保持
- **線形計算量**: シーケンス長に対して線形の計算量
- **畳み込み演算**: 効率的な並列処理を実現

### 技術スタック
- **PyTorch**: 深層学習フレームワーク
- **NumPy**: 数値計算ライブラリ
- **Pandas**: データ処理ライブラリ
- **Matplotlib**: 可視化ライブラリ
- **tqdm**: 進捗表示ライブラリ

## システム特徴

### 多言語サポート
- **日本語**: ひらがな、カタカナ、漢字に対応
- **英語**: アルファベット、数字、記号に対応
- **Python**: コード構文とコメントの学習

### データ入力形式
- **テキストファイル (.txt)**: プレーンテキスト
- **CSVファイル (.csv)**: 構造化データ
- **Pythonファイル (.py)**: ソースコード

### 学習進捗表示
- **リアルタイム進捗**: エポック・ステップレベルの進捗表示
- **損失値監視**: 学習損失の推移を可視化
- **パープレキシティ**: モデルの予測精度指標
- **ETA表示**: 残り時間の推定

## ファイル構成

```
machine_learning_mamba/
├── main.py                 # メイン実行スクリプト
├── mamba_model.py          # Mambaモデル実装
├── data_loader.py          # データ読み込み・前処理
├── trainer.py              # 学習・評価システム
├── requirements.txt        # 依存関係
├── setup.py               # パッケージ設定
├── Dockerfile             # Docker設定
├── docker-compose.yml     # Docker Compose設定
├── .dockerignore          # Docker除外ファイル
└── README.md              # このファイル
```

## セットアップ

### 環境構築
```bash
# リポジトリのクローン
git clone <repository-url>
cd machine_learning_mamba

# 依存関係のインストール
pip install -r requirements.txt

# または開発環境用
pip install -e .[dev]
```

### GPU使用時
```bash
pip install -e .[gpu]
```

### Docker環境での実行

#### 前提条件
- Docker Engine 20.10+
- Docker Compose 2.0+
- GPU使用時: NVIDIA Docker (nvidia-container-toolkit)

#### Dockerイメージのビルド
```bash
# イメージをビルド
docker build -t mamba-ml .

# または Docker Compose使用
docker-compose build
```

#### 基本的な実行
```bash
# サンプルデータ作成
docker-compose run --rm mamba-ml python main.py --create_sample

# 学習実行
docker-compose run --rm mamba-ml python main.py

# カスタム設定での実行
docker-compose run --rm mamba-ml python main.py --epochs 5 --batch_size 8
```

#### 開発環境での使用
```bash
# 開発用コンテナ起動（対話モード）
docker-compose run --rm mamba-ml-dev

# コンテナ内でコマンド実行
root@container:/app# python main.py --create_sample
root@container:/app# python main.py --epochs 3
```

#### GPU使用時
```bash
# NVIDIA Docker設定確認
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# GPU使用でコンテナ実行
docker-compose run --rm --gpus all mamba-ml python main.py --device cuda
```

#### データの永続化
```bash
# ローカルディレクトリ作成
mkdir -p data checkpoints results

# データを配置
cp your_data.txt data/

# コンテナ実行（データが永続化される）
docker-compose run --rm mamba-ml python main.py --data_dir /app/data
```

## 使用方法

### サンプルデータの作成
```bash
python main.py --create_sample
```

### 基本的な学習
```bash
python main.py
```

### カスタム設定での学習
```bash
python main.py \
  --data_dir ./your_data \
  --epochs 10 \
  --batch_size 8 \
  --d_model 256 \
  --n_layers 4 \
  --learning_rate 1e-4
```

### 利用可能なオプション

| オプション | デフォルト | 説明 |
|-----------|------------|------|
| `--data_dir` | sample_data | 学習データディレクトリ |
| `--epochs` | 3 | エポック数 |
| `--batch_size` | 4 | バッチサイズ |
| `--max_length` | 128 | 最大シーケンス長 |
| `--d_model` | 128 | モデルの次元数 |
| `--n_layers` | 2 | レイヤー数 |
| `--learning_rate` | 1e-4 | 学習率 |
| `--device` | auto | 使用デバイス (cpu/cuda/auto) |
| `--test_text` | 機械学習 | テスト用テキスト |

## データ形式

### テキストファイル
```
データ/
├── japanese.txt     # 日本語テキスト
├── english.txt      # 英語テキスト
└── sample_code.py   # Pythonコード
```

### CSVファイル
```csv
text,label
"機械学習は人工知能の分野です",japanese
"Machine learning is a field of AI",english
```

## 出力情報

### 起動時情報
- デバイス情報（CPU/GPU）
- データ統計（言語別テキスト数）
- モデル情報（パラメータ数）

### 学習中情報
- エポック進捗（1/5など）
- ステップ進捗（バー表示）
- 損失値（リアルタイム更新）
- パープレキシティ（予測精度）
- 学習率（現在値）
- ETA（残り時間推定）

### 学習完了後
- 最終メトリクス（損失、精度）
- 総学習時間
- モデル生成例
- 結果保存（JSON形式）

## Mambaアーキテクチャ詳細

### State Space Model
```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t + D * x_t
```

### 選択的状態更新
- Δ（デルタ）: 状態更新の重要度制御
- B、C: 入力・出力ゲート
- 重要な情報のみを選択的に保持

### 効率性
- **線形計算量**: O(L)（Lはシーケンス長）
- **並列処理**: 畳み込み演算による高速化
- **メモリ効率**: 長系列でも一定メモリ使用量

## パフォーマンス

### 推奨環境
- **CPU**: Intel Core i5以上またはAMD Ryzen 5以上
- **RAM**: 8GB以上
- **GPU**: CUDA対応GPU（オプション）

### 処理速度
- **CPU**: 約100-200 tokens/秒
- **GPU**: 約500-1000 tokens/秒（CUDA利用時）

## カスタマイズ

### モデル設定
```python
model = MambaModel(
    vocab_size=10000,    # 語彙サイズ
    d_model=256,         # モデル次元
    n_layers=6,          # レイヤー数
    d_state=32           # 状態次元
)
```

### 学習設定
```python
trainer.setup_training(
    learning_rate=1e-4,
    weight_decay=0.01
)
```

## トラブルシューティング

### よくある問題

#### メモリ不足エラー
```bash
# バッチサイズを小さくする
python main.py --batch_size 2

# シーケンス長を短くする
python main.py --max_length 64
```

#### CUDA関連エラー
```bash
# CPU強制使用
python main.py --device cpu
```

#### データ読み込みエラー
- ファイルパスの確認
- エンコーディング（UTF-8）の確認
- ファイル権限の確認

#### Docker関連エラー
```bash
# コンテナのログ確認
docker-compose logs mamba-ml

# コンテナ内デバッグ
docker-compose run --rm mamba-ml-dev /bin/bash

# GPU使用時の問題
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# ポート権限エラー（Linux）
sudo chown -R $USER:$USER ./data ./checkpoints
```

## ライセンス

MIT License

## 貢献

プルリクエストや Issue の報告を歓迎します。

## 参考文献

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- State Space Models for Deep Learning
- Transformer alternatives and efficiency