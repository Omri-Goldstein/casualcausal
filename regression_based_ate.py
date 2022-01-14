from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from ate_estimators.ate_interface import AteEstimatorInterface


class RegressionAteEstimator(AteEstimatorInterface):

    def estimate_ate(self, df: pd.DataFrame, features: List[str], treatment: str, target: str, **kwargs) -> float:
        reg = LinearRegression()
        dummies_df = pd.get_dummies(df[features])
        reg.fit(dummies_df, df[target])

        return reg.coef_[dummies_df.columns.get_loc(treatment)]


class OaxacaAteEstimator(AteEstimatorInterface):

    def estimate_ate(self, df: pd.DataFrame, features: List[str], treatment: str, target: str, plot=False) -> float:
        control_reg, treatment_reg, regs_df = self._train_regressions_for_control_and_treatment(df, features, target,
                                                                                                treatment)

        average_control, average_treatment = self._get_average_control_and_treatment(df, regs_df, treatment)
        decomposition_df = self._get_decomposition_df(average_control, average_treatment, control_reg, regs_df,
                                                      treatment_reg)

        if plot:
            self._plot(decomposition_df)

        ate = self._calc_ate(average_treatment, control_reg, df, target, treatment)
        return ate

    @staticmethod
    def _plot(decomposition_df: pd.DataFrame) -> None:
        decomposition_df[['diff_in_comp']].sort_values(by='diff_in_comp').plot(kind='barh')
        plt.show()

    @staticmethod
    def _calc_ate(average_treatment,
                  control_reg: LinearRegression,
                  df: pd.DataFrame,
                  target: str,
                  treatment: str
                  ) -> float:
        average_target_for_treatment = df[df[treatment] == True][target].mean()
        average_target_for_control = df[df[treatment] == False][target].mean()
        predicted_average_target_for_treatment = control_reg.predict(average_treatment.values.reshape(1, -1))[0]
        ate = average_target_for_treatment - predicted_average_target_for_treatment

        print(f'Gap: {average_target_for_treatment - average_target_for_control}')
        print(f'Explained Gap: {average_target_for_treatment - average_target_for_control - ate}')
        print(f'Unexplained Gap: {ate}')

        return ate

    @staticmethod
    def _get_average_control_and_treatment(df: pd.DataFrame, regs_df: pd.DataFrame, treatment: str) -> Tuple[Any, Any]:
        average_treatment = regs_df[df[treatment] == True].mean()
        average_control = regs_df[df[treatment] == False].mean()
        return average_control, average_treatment

    @staticmethod
    def _train_regressions_for_control_and_treatment(df: pd.DataFrame,
                                                     features: List[str],
                                                     target: str,
                                                     treatment: str
                                                     ) -> Tuple[LinearRegression, LinearRegression, pd.DataFrame]:
        control_reg = LinearRegression()
        treatment_reg = LinearRegression()

        regs_df = pd.get_dummies(df[features])

        control_reg.fit(regs_df[df[treatment] == False], df[df[treatment] == False][target])
        treatment_reg.fit(regs_df[df[treatment] == True], df[df[treatment] == True][target])
        return control_reg, treatment_reg, regs_df

    @staticmethod
    def _get_decomposition_df(average_control, average_treatment, control_reg, regs_df, treatment_reg):
        comparison_df = pd.DataFrame({'cols': regs_df.columns,
                                      'treatment_coefs': treatment_reg.coef_,
                                      'control_coef': control_reg.coef_,
                                      'average_treatment': average_treatment,
                                      'average_control': average_control})
        comparison_df['treatment_predicted_by_control'] = comparison_df.average_treatment * comparison_df.control_coef
        comparison_df['control_predicted_by_control'] = comparison_df.average_control * comparison_df.control_coef
        comparison_df['diff_in_comp'] = comparison_df.control_predicted_by_control - \
                                        comparison_df.treatment_predicted_by_control
        return comparison_df
