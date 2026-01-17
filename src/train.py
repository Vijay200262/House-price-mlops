import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = "data/housing.csv"
MODEL_PATH = "models/model.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH)
    

    x = df[["area","bedrooms","age"]]
    y = df["price"]



    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )


    model = LinearRegression()
    model.fit(x_train, y_train)


    os.makedirs("models",exist_ok=True)
    joblib.dump(model,MODEL_PATH)

    print("model trained and saved")


if __name__ == "__main__":
    train_model()
