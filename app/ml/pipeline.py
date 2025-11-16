from typing import Dict, List

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np

from app.services.training import fetch_training_window


class MLModels:
    def __init__(self):
        self.regime_model = None
        self.forecast_model = None
        self.anomaly_model = None

    def train(self):
        data = fetch_training_window()
        X = self._build_feature_matrix(data)
        if not len(X):
            return
        self.regime_model = KMeans(n_clusters=4).fit(X)
        self.forecast_model = LinearRegression().fit(X[:-1], X[1:, 0])
        self.anomaly_model = IsolationForest(contamination=0.1).fit(X)

    def _build_feature_matrix(self, data: Dict[str, List]) -> np.ndarray:
        series = sorted(data.keys())
        matrix = []
        for i in range(min(len(points) for points in data.values())):
            matrix.append([data[s][i].value for s in series])
        return np.array(matrix)
