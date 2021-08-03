# AutoEncorder

- AutoEncorder のexample プログラム

## リポジトリ構成

```
.
├── Dockerfile
├── README.md
├── data
├── docker-compose.yml
├── docs
├── models
├── notebooks
│   ├── Convolutional_Autoencoder.ipynb
│   ├── Deep_Autoencoder.ipynb
│   ├── Denoising_Autoencoder.ipynb
│   └── Sparse_Autoencoder.ipynb
├── pyproject.toml
├── requirements.txt
├── setup.cfg
├── src
│   └── __init__.py
├── tests
│   └── __init__.py
└── work
```

## AutoEncoder とは

- オートエンコーダ（自己符号化器：AutoEncoder）とは、ニューラルネットワークの1つです。入力されたデータを一度圧縮し、重要な特徴量だけを残した後、再度もとの次元に復元処理をするアルゴリズムを意味します。このように、小さい次元に落とし込む作業を次元削減や特徴抽出と呼びますが、オートエンコーダはそれだけでなく、生成モデルとしても用いられます。
- オートエンコーダは、2006年にトロント大学のコンピュータ科学および認知心理学の研究者であるジェフリー・ヒントン氏らによって提唱されました。

## 環境詳細

- Google Colab
