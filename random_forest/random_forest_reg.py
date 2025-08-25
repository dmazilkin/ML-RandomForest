import pandas as pd
import numpy as np
import random 
from typing import List, Tuple

from decision_tree.decision_tree_reg import MyTreeReg

class MyForestReg:
    def __init__(self, 
                n_estimators: int = 10, 
                max_features: float = 0.5,
                max_samples: float = 0.5,
                random_state: int = 42,
                max_depth: int = 5,
                min_samples_split: int = 2,
                max_leafs: int = 20,
                bins: int = 16) -> None:
        self._n_estimators = n_estimators
        self._max_features = max_features
        self._max_samples = max_samples
        self._random_state = random_state
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._max_leafs = max_leafs
        self._bins = bins
        self.leafs_cnt = 0
        self._trees: List[MyTreeReg] = list()
        
    def __str__(self) -> str:
        return f'MyForestReg class: n_estimators={self._n_estimators}, max_features={self._max_features}, max_samples={self._max_samples}, max_depth={self._max_depth}, min_samples_split={self._min_samples_split}, max_leafs={self._max_leafs}, bins={self._bins}, random_state={self._random_state}'
    
    def _create_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        features_cnt = round(X.shape[1] * self._max_features)
        samples_cnt = round(X.shape[0] * self._max_samples)
        cols_idx = random.sample(list(X.columns), features_cnt)
        rows_idx = random.sample(range(X.shape[0]), samples_cnt)
        
        return X.loc[rows_idx, cols_idx], y.loc[rows_idx]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self._random_state)
        
        for _ in range(self._n_estimators):
            X_estimator, y_estimator = self._create_dataset(X, y)
            self._trees.append(MyTreeReg(max_depth=self._max_depth, min_samples_split=self._min_samples_split, max_leafs=self._max_leafs, bins=self._bins))
            self._trees[-1].fit(X_estimator, y_estimator)
            self.leafs_cnt += self._trees[-1]._leafs_cnt
            
    def predict(self, X: pd.DataFrame) -> pd.Series:
        predicts = np.zeros((X.shape[0],), dtype='float')
        
        for tree in self._trees:
            predicts += tree.predict(X)

        return predicts / self._n_estimators