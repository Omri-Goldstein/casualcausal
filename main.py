"""This scripts contains examples of usage of several causal inference techniques for a specific project of mine."""
from typing import List, Dict, Tuple, Any

import pandas as pd
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression

from ate_estimators.catboost_shap_ate import CatboostShapAteEstimator
from ate_estimators.inverse_probability_weighting import InverseProbabilityWeightingAteEstimator
from ate_estimators.matching import MatchingATE
from ate_estimators.regression_based_ate import RegressionAteEstimator, OaxacaAteEstimator
from hte_estimators.causal_random_forest import estimate_hte


def create_matching_df(df: pd.DataFrame) -> pd.DataFrame:
    """This function is data-engineering specific to my project"""
    matching_df = df.copy()
    matching_df['Gender'] = matching_df.Gender == 'Female'
    matching_df["Gamfa"] = df.gamfa.apply(lambda x: 1 if x else 0)
    matching_df["Experience"] = df.experience
    matching_df["Education"] = df.education_level
    matching_df["Manager"] = df.manager
    matching_df["Seniority"] = df.seniority
    matching_df["Field"] = df.field
    matching_df["salary"] = df.salary
    matching_df['company_type'] = df.company

    for col in ["Age", "Education", "Experience", "Gamfa", "Manager", "Seniority"]:
        matching_df[col] = (matching_df[col] - matching_df[col].mean()) / matching_df[col].std()

    matching_df = matching_df[["Age", "Education", "Experience", "Gamfa", "Manager",
                               "Seniority", "salary", "Gender", "position", "Field", "company_type"]]

    matching_df.reset_index(inplace=True)
    return matching_df


def get_data_and_config(path: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """This function performs data-engineering and configuration specific to my project"""
    df = create_matching_df(pd.read_csv(path))
    config_dict = dict(cols_to_force=["Field", "position"],
                       relation_to_force=[True, True],
                       cols_to_calc=["Age", "Education", "Experience", "Gamfa", "Manager", "Seniority"],
                       target='salary',
                       treatment='Gender',
                       features=['Age', 'Education', 'Experience', 'Gamfa', 'Manager', 'Seniority', 'position', 'Field',
                                 'Gender',
                                 'company_type'],
                       categorical_features=['position', 'Field', 'company_type'],
                       monotone_constraints_dict={'Experience': 1, 'Education': 1,
                                                  'Seniority': 1, 'Manager': 1})
    return config_dict, df


def get_ate_with_in_house_code(df: pd.DataFrame, cols_to_calc: List[str],
                               features: List[str],
                               target: str,
                               treatment: str,
                               cols_to_force: List[str],  # For matching
                               relation_to_force: List[bool],  # For matching
                               categorical_features: List[str],  # For catboost
                               monotone_constraints_dict: Dict[str, int],  # For catboost
                               ) -> None:
    """
    Examples of matching, propensity score, catboost shap and other ML and causal inference techniques
    :param df: pandas dataframe with a treatment column, a target column and covariates
    :param features: list of covariates to be used
    :param target: the name of the target column
    :param treatment: the name of the treatment column
    :param cols_to_calc: for matching ATE estimation - which columns to use for distance calculation
    :param cols_to_force: for matching ATE - which columns must be equal or different for a valid match
    :param relation_to_force: for matching ATE - whether the values in the forced columns should be equal or not
    :param categorical_features: for Catboost ATE estimation - what columns are categorical
    :param monotone_constraints_dict: for Catboost ATE estimation - enforce monotonic behavior for some features
    """
    ate = MatchingATE(max_dist=6,
                      distance='euclidian',
                      cols_to_force=cols_to_force,
                      relation_to_force=relation_to_force,
                      cols_to_calc=cols_to_calc).estimate_ate(df, treatment=treatment, target=target)
    print('Matching ATE: ', ate)
    ate = InverseProbabilityWeightingAteEstimator().estimate_ate(df,
                                                                 target=target,
                                                                 treatment=treatment,
                                                                 propensity_cols=features)
    print('propensity ATE', ate)
    ate = MatchingATE(max_dist=0.1,
                      distance='euclidian',
                      cols_to_force=cols_to_force,
                      relation_to_force=relation_to_force,
                      cols_to_calc=cols_to_calc,
                      propensity=True,
                      propensity_cols=features).estimate_ate(df, treatment=treatment, target=target)
    print('Propensity Matching ATE: ', ate)
    ate = CatboostShapAteEstimator().estimate_ate(df=df,
                                                  features=features,
                                                  target=target,
                                                  treatment=treatment,
                                                  categorical_features=categorical_features,
                                                  monotone_constraints_dict=monotone_constraints_dict)
    print('Catboost Shap ATE', ate)
    ate = RegressionAteEstimator().estimate_ate(df=df,
                                                features=features,
                                                target=target,
                                                treatment=treatment)
    print('Regression ATE', ate)
    ate = OaxacaAteEstimator().estimate_ate(df=df,
                                            features=features,
                                            target=target,
                                            treatment=treatment,
                                            plot=True)
    print('Oaxaca ATE', ate)
    effects_train, conf_intrvl = estimate_hte(df,
                                              features=features,
                                              target=target,
                                              treatment=treatment)
    # Do something with effects_train and conf_intrvl


def get_ate_with_causal_inference_lib(df: pd.DataFrame, target: str, treatment: str, features: List[str]) -> None:
    """
    :param df: pandas dataframe with a treatment column, a target column and covariates
    :param features: list of covariates to be used
    :param target: the name of the target column
    :param treatment: the name of the treatment column
    """
    new_df = df[pd.notnull(df.salary)]
    new_df.dropna(inplace=True)
    cm = CausalModel(
        Y=new_df[target].values,
        D=new_df[treatment].values,
        X=pd.get_dummies(new_df[features], drop_first=False).values)
    cm.est_propensity()
    cm.stratify_s()
    cm.est_via_weighting()
    cm.est_via_matching(matches=1, bias_adj=True)
    cm.est_via_ols(adj=2)
    print('Use Matching, Regression and Inverse Probability Weighting')
    print(cm.estimates)
    df["propensity_score"] = LogisticRegression().fit(
        pd.get_dummies(new_df[features]), df.Gender).predict_proba(pd.get_dummies(new_df[features]))[:, 1]
    cm = CausalModel(
        Y=df[target].values,
        D=df[treatment].values,
        X=df[["propensity_score"]].values
    )
    print('Use Propensity Score Matching')
    cm.est_via_matching(matches=1, bias_adj=True)
    print(cm.estimates)


def main(path: str, use_in_house_code: bool = True, use_causal_inference_lib: bool = True) -> None:
    """
    :param path: path to the .csv datafile
    :param use_in_house_code: whether or not to use my implementations of several causal inference techniqes
    :param use_causal_inference_lib: whether or not to use the causalinference implementation
    """
    # The following line is specific to the problems I was working on
    config_dict, df = get_data_and_config(path)

    # This following lines are project agnostic and can be used on any appropriate data
    if use_in_house_code:
        get_ate_with_in_house_code(df=df,
                                   target=config_dict['target'],
                                   treatment=config_dict['treatment'],
                                   features=config_dict['features'],
                                   cols_to_calc=config_dict['cols_to_calc'],
                                   cols_to_force=config_dict['cols_to_force'],
                                   relation_to_force=config_dict['relation_to_force'],
                                   categorical_features=config_dict['categorical_features'],
                                   monotone_constraints_dict=config_dict['monotone_constraints_dict'])

    if use_causal_inference_lib:
        get_ate_with_causal_inference_lib(df=df,
                                          target=config_dict['target'],
                                          treatment=config_dict['treatment'],
                                          features=config_dict['features'])


if __name__ == '__main__':
    path_to_data_frame = ''
    main(path_to_data_frame)
