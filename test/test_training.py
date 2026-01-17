import os 
from src.train import train_model

def test_model_creation():
    train_model()
    assert os.path.exists("models/model.pkl")
