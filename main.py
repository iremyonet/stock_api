from fastapi import FastAPI
import numpy
from typing import List, Dict
import joblib
#from tensorflow.keras.models import Sequential
from statsmodels.regression.linear_model import OLSResults
from pandas import DataFrame

from pydantic import BaseModel
#from keras.models import load_model

app = FastAPI()

scaler = joblib.load("scaler")
#lstm_model: Sequential = load_model("lstm.hdf5")
#gru_model: Sequential = load_model("gru.hdf5")
arima_model = OLSResults.load("arima.pickle")
#sarimax_model = OLSResults.load("sarimax.pickle")


class TensorRequest(BaseModel):
    array: list[float]


class StatRequest(BaseModel):
    Low: Dict[str, float]
    High: Dict[str, float]
    Open: Dict[str, float]
    Close: Dict[str, float]
    Volume: Dict[str, float]
    Mean: Dict[str, float]


def prepare(request: TensorRequest):
    array = numpy.array(request.array)
    array = array.reshape((5, 1))
    array = scaler.transform(array)
    return numpy.expand_dims(array, 0)


@app.post("/lstm")
async def lstm(request: TensorRequest):
    array = prepare(request)

    result = lstm_model.predict(x=array)

    result = scaler.inverse_transform(result)

    value = result[0, 0]
    value = value.item()

    print(value)

    return value


@app.post("/gru")
async def gru(request: TensorRequest):
    array = prepare(request)

    result = gru_model.predict(x=array)

    result = scaler.inverse_transform(result)

    value = result[0, 0]
    value = value.item()

    print(value)

    return value


@app.post("/arima")
async def arima(request: StatRequest):
    dc = dict(request)
    df = DataFrame.from_dict(dc)
    df.set_index(df.iloc[:, 0], inplace=True)

    result = arima_model.predict(x=df)

    return result


@app.post("/sarimax")
async def sarimax(request: StatRequest):
    dc = dict(request)
    df = DataFrame.from_dict(dc)
    df.set_index(df.iloc[:, 0], inplace=True)

    result = sarimax_model.predict(x=df)

    return result

