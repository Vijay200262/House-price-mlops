import pandas as pd 
import joblib
from sklearn.metrics import r2_score


MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/housing.csv"


def evaluate_model():
    df = pd.read_csv(DATA_PATH)

    x = df[["area","bedrooms","age"]]
    y = df["price"]

    model = joblib.load(MODEL_PATH)
    preds = model.predict(x)

    score = r2_score(y, preds)
    print(f"R2 Score: {score}")

    if score < 0.8:
        raise ValueError("MODEL ACCURACY NOT CORRECT")
    print("MODEL EVALUATION PASSED")


if __name__ == "__main__":
    evaluate_model()