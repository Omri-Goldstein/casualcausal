"""Implementation of ATE estimation using a catboost model and SHAP values."""
from typing import List

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from ate_estimators.ate_interface import AteEstimatorInterface


class CatboostShapAteEstimator(AteEstimatorInterface):
    def estimate_ate(self,
                     df: pd.DataFrame,
                     features: List[str],
                     target: str,
                     treatment: str,
                     **kwargs
                     ) -> float:

        categorical_features = kwargs['categorical_features']
        monotone_constraints_dict = kwargs['monotone_constraints_dict']

        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=1)

        reg = CatBoostRegressor(cat_features=categorical_features,
                                learning_rate=0.05,
                                eval_metric='R2',
                                num_trees=250,
                                monotone_constraints=monotone_constraints_dict,
                                model_shrink_rate=0
                                )

        reg.fit(X_train, y_train, plot=True, eval_set=(X_test, y_test), verbose=10)

        shap_values = reg.get_feature_importance(catboost.Pool(df[features],
                                                               cat_features=categorical_features),
                                                 type="ShapValues")
        shap_values = shap_values[:, :-1]

        return np.mean(shap_values[:, features.index(treatment)])
