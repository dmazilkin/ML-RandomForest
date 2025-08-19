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
        
    def __str__(self) -> str:
        return f'MyForestReg class: n_estimators={self._n_estimators}, max_features={self._max_features}, max_samples={self._max_samples}, max_depth={self._max_depth}, min_samples_split={self._min_samples_split}, max_leafs={self._max_leafs}, bins={self._bins}, random_state={self._random_state}'