import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.pipeline import Pipeline

# モデルのパスを定義
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/titanic_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")

@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    return pd.read_csv(DATA_PATH)

@pytest.fixture
def load_model():
    """学習済みモデルを読み込む"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def test_model_inference_time(load_model, sample_data):
    """モデルの推論時間を検証"""
    model = load_model

    # 入力データのサンプルを用意
    X_sample = sample_data.drop("Survived", axis=1).iloc[:10]  # 10件のデータで検証

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_sample)
    elapsed_time = time.time() - start_time

    # 推論時間の閾値（1秒未満）
    assert elapsed_time < 1.0, f"推論時間が長すぎます: {elapsed_time:.4f}秒"
