import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(BASE_DIR, "dataset", "air_quality_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = ['PM2.5','PM10','NO2','SO2','CO','O3']
TARGET = "AQI"

SEQ_LEN = 24
EPOCHS = 35
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)


def create_sequences(df):

    X = []
    y = []

    values = df[FEATURES + [TARGET]].values

    for i in range(len(values) - SEQ_LEN):
        X.append(values[i:i+SEQ_LEN, :len(FEATURES)])
        y.append(values[i+SEQ_LEN, -1])

    return np.array(X), np.array(y)


def build_model():

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),
        Dropout(0.2),

        LSTM(64),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_city(city, df):

    print("Training:", city)

    city_df = df[df["City"] == city].copy()

    if len(city_df) < 400:
        print("Skipping city due to insufficient data")
        return None

    X, y = create_sequences(city_df)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_rs = X_train.reshape(-1, len(FEATURES))
    X_val_rs = X_val.reshape(-1, len(FEATURES))

    feature_scaler.fit(X_train_rs)

    X_train_scaled = feature_scaler.transform(X_train_rs).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val_rs).reshape(X_val.shape)

    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1,1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1,1))

    model = build_model()

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    preds_scaled = model.predict(X_val_scaled)

    preds = target_scaler.inverse_transform(preds_scaled)

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    model_path = os.path.join(
        MODEL_DIR,
        f"lstm_model_{city.replace(' ','_')}.h5"
    )

    model.save(model_path)

    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "history": history.history
    }


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["City","Date"])

    cities = sorted(df["City"].unique())

    city_scalers = {}

    for city in cities:

        result = train_city(city, df)

        if result:
            city_scalers[city] = result

    joblib.dump(
        city_scalers,
        os.path.join(MODEL_DIR, "city_scalers.pkl")
    )

    print("Training complete")


if __name__ == "__main__":
    main()