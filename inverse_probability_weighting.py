"""Implementation of ATE estimation using inverse probability weighting with propensity score."""
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ate_estimators.ate_interface import AteEstimatorInterface


class InverseProbabilityWeightingAteEstimator(AteEstimatorInterface):
    def estimate_ate(self,
                     df: pd.DataFrame,
                     treatment: str,
                     target: str,
                     features: List[str],
                     **kwargs
                     ) -> float:

        df['propensity_score'] = calc_propensity_score(df, features, treatment)

        weight, weight_nt, weight_t = self._calc_weights(df, treatment)

        y1 = sum(df[df[treatment] == True][target] * weight_t) / len(df)
        y0 = sum(df[df[treatment] == False][target] * weight_nt) / len(df)

        ate = np.mean(weight * df[target])

        print("Y1:", y1)
        print("Y0:", y0)
        print("ATE", np.mean(weight * df[target]))
        return ate

    @staticmethod
    def _calc_weights(df, treatment):
        weight_t = 1 / df[df[treatment] == True]["propensity_score"]
        weight_nt = 1 / (1 - df[df[treatment] == False]["propensity_score"])
        weight = ((df[treatment] - df["propensity_score"]) /
                  (df["propensity_score"] * (1 - df["propensity_score"])))
        return weight, weight_nt, weight_t


def calc_propensity_score(df, propensity_cols, treatment):
    X = pd.get_dummies(df[propensity_cols])
    y = df[treatment]
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]
