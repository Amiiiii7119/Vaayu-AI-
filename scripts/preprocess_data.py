import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "dataset", "city_hour.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "air_quality_data.csv")

CITY_STATE_MAP = {
    "Ahmedabad":"Gujarat",
    "Aizawl":"Mizoram",
    "Amaravati":"Andhra Pradesh",
    "Amritsar":"Punjab",
    "Bengaluru":"Karnataka",
    "Bhopal":"Madhya Pradesh",
    "Brajrajnagar":"Odisha",
    "Chandigarh":"Chandigarh",
    "Chennai":"Tamil Nadu",
    "Coimbatore":"Tamil Nadu",
    "Delhi":"Delhi",
    "Ernakulam":"Kerala",
    "Gurugram":"Haryana",
    "Guwahati":"Assam",
    "Hyderabad":"Telangana",
    "Jaipur":"Rajasthan",
    "Jorapokhar":"Jharkhand",
    "Kochi":"Kerala",
    "Kolkata":"West Bengal",
    "Lucknow":"Uttar Pradesh",
    "Mumbai":"Maharashtra",
    "Patna":"Bihar",
    "Shillong":"Meghalaya",
    "Talcher":"Odisha",
    "Thiruvananthapuram":"Kerala",
    "Visakhapatnam":"Andhra Pradesh"
}

FEATURES = ['PM2.5','PM10','NO2','SO2','CO','O3','AQI']

def preprocess():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    df = df.rename(columns={"Datetime":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[['City','Date','PM2.5','PM10','NO2','SO2','CO','O3','AQI']]

    df.replace("None", np.nan, inplace=True)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df = df[df["City"].isin(CITY_STATE_MAP.keys())]

    df["State"] = df["City"].map(CITY_STATE_MAP)

    df = df.sort_values(["City","Date"])
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset prepared")
    print("Rows:", len(df))
    print("Cities:", df["City"].nunique())

if __name__ == "__main__":
    preprocess()