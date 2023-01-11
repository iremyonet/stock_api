import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

dataset = pd.read_csv("NVDA.csv")

data = dataset[['Close']].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data)

joblib.dump(scaler, "scaler")
