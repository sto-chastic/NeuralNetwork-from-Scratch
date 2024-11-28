import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data):
    mean, std = data.mean(), data.std()
    normalized = (data - mean) / std
    return normalized, mean, std
