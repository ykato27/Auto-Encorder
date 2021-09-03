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
│   ├── AutoEncoder_Pytorch_example.ipynb
│   ├── Convolutional_AutoEncoder_Pytorch_example.ipynb
│   ├── Google_Colab
│   │   ├── AE_CIFAR10.ipynb
│   │   ├── AutoEncoder_Keras.ipynb
│   │   ├── AutoEncoder_Pytorch.ipynb
│   │   ├── Convolutional_Autoencoder.ipynb
│   │   ├── Deep_Autoencoder.ipynb
│   │   ├── Denoising_Autoencoder.ipynb
│   │   ├── Sparse_Autoencoder.ipynb
│   │   ├── VAE_Pyro.ipynb
│   │   ├── VAE_Pytorch.ipynb
│   │   ├── anomaly_detection_autoencoder_2.ipynb
│   │   └── model_based_optimization_witth_vae.ipynb
│   └── data
│       └── MNIST
│           └── raw
│               ├── t10k-images-idx3-ubyte
│               ├── t10k-images-idx3-ubyte.gz
│               ├── t10k-labels-idx1-ubyte
│               ├── t10k-labels-idx1-ubyte.gz
│               ├── train-images-idx3-ubyte
│               ├── train-images-idx3-ubyte.gz
│               ├── train-labels-idx1-ubyte
│               └── train-labels-idx1-ubyte.gz
├── pyproject.toml
├── requirements.txt
├── setup.cfg
├── src
│   ├── README.md
│   ├── __init__.py
│   ├── ae.py
│   ├── model.py
│   └── result
│       ├── input_00.png
│       ├── input_01.png
│       ├── input_02.png
│       ├── input_03.png
│       ├── input_04.png
│       ├── input_05.png
│       ├── input_06.png
│       ├── input_07.png
│       ├── input_08.png
│       ├── input_09.png
│       ├── out_after_00.png
│       ├── out_after_01.png
│       ├── out_after_02.png
│       ├── out_after_03.png
│       ├── out_after_04.png
│       ├── out_after_05.png
│       ├── out_after_06.png
│       ├── out_after_07.png
│       ├── out_after_08.png
│       ├── out_after_09.png
│       ├── out_before_00.png
│       ├── out_before_01.png
│       ├── out_before_02.png
│       ├── out_before_03.png
│       ├── out_before_04.png
│       ├── out_before_05.png
│       ├── out_before_06.png
│       ├── out_before_07.png
│       ├── out_before_08.png
│       └── out_before_09.png
├── tests
│   └── __init__.py
└── work
```

## AutoEncoder とは

- オートエンコーダ（自己符号化器：AutoEncoder）とは、ニューラルネットワークの1つです。入力されたデータを一度圧縮し、重要な特徴量だけを残した後、再度もとの次元に復元処理をするアルゴリズムを意味します。このように、小さい次元に落とし込む作業を次元削減や特徴抽出と呼びますが、オートエンコーダはそれだけでなく、生成モデルとしても用いられます。
- オートエンコーダは、2006年にトロント大学のコンピュータ科学および認知心理学の研究者であるジェフリー・ヒントン氏らによって提唱されました。

## 環境詳細

- Google Colab
