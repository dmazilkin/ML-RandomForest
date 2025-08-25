import pandas as pd
from sklearn.datasets import make_regression
from typing import Tuple

from random_forest.random_forest_reg import MyForestReg

def create_data_regression() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(n_samples=100, n_features=5, n_informative=4, noise=5, random_state=42)
    
    return pd.DataFrame(X), pd.Series(y)

def regression():
    X, y = create_data_regression()
    model = MyForestReg()
    model.fit(X, y)