from typing import List

import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def fit_model(df: pd.DataFrame, target: str, treatment: str, features: List[str]):
    estimator = CausalForestDML(
        max_depth=4,
        model_t=RandomForestClassifier(),
        model_y=RandomForestRegressor(),
        discrete_treatment=True,
        n_estimators=100,
        min_impurity_decrease=0.001,
        random_state=0)
    estimator.fit(Y=df[target],
                  T=df[treatment],
                  X=pd.get_dummies(df[features]),
                  inference='blb')
    return estimator


def estimate_hte(df: pd.DataFrame, target: str, treatment: str, features: List[str]):
    estimator = fit_model(df, target, treatment, features)
    effects_train = estimator.effect(pd.get_dummies(df[features]))
    conf_intrvl = estimator.effect_interval(pd.get_dummies(df[features]))
    return effects_train, conf_intrvl





