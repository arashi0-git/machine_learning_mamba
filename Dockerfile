# Python 3.9-slim をベースイメージとして使用（代替レジストリ対応）
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# データディレクトリを作成
RUN mkdir -p /app/sample_data /app/checkpoints

# 実行時のデフォルトコマンド
CMD ["python", "main.py", "--help"]

# ポート設定（必要に応じて）
EXPOSE 8000

# 環境変数
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1