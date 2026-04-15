#!/usr/bin/env python
# coding: utf-8

# In[2]:


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import statsmodels.api as sm

app = FastAPI()

# 모델 로드
with open("model.pkl", "rb") as f:
    ols_models = pickle.load(f)

Y_cols = ["chest", "waist", "thigh", "shoulder", "arm", "top", "bottom", "hem"]

class UserInput(BaseModel):
    weight: float
    height: float

@app.post("/predict")
def predict(data: UserInput):
    user_X = np.array([[data.weight, data.height]])
    user_X_sm = sm.add_constant(user_X, has_constant='add')

    result = {}

    for i, col in enumerate(Y_cols):
        pred = ols_models[i].predict(user_X_sm)
        result[col] = round(float(pred[0]), 2)

    return result


# In[ ]:




