import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
import warnings

from ate_estimators.ate_interface import AteEstimatorInterface
from ate_estimators.inverse_probability_weighting import calc_propensity_score

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MatchingATE(AteEstimatorInterface):
    def __init__(self,
                 distance: str,
                 max_dist: float,
                 cols_to_force: List[str],
                 relation_to_force: List[bool],
                 cols_to_calc: List[str],
                 propensity: bool = False,
                 propensity_cols: List = list()):
        super(MatchingATE, self).__init__()
        self.distance = distance
        self.max_dist = max_dist
        self.dist = scipy.spatial.distance.euclidean if distance == 'euclidian' \
            else scipy.spatial.distance.mahalanobis
        self.cols_to_force = cols_to_force
        self.cols_to_force_indices = []  # To be filled on the fly

        self.relation_to_force = relation_to_force
        self.cols_to_calc = cols_to_calc
        self.cols_to_calc_indices = []  # To be filled on the fly
        self.propensity = propensity
        if propensity:
            self.cols_to_force = []
            self.relation_to_force = []
            self.cols_to_calc = []

        self.propensity_cols = propensity_cols

    def estimate_ate(self,
                     df: pd.DataFrame,
                     treatment: str,
                     target: str,
                     features: Optional[List[str]] = None,
                     **kwargs
                     ) -> float:

        if self.propensity:
            df['propensity_score'] = calc_propensity_score(df, self.propensity_cols, treatment)
            self.cols_to_calc = ['propensity_score']

        treatment_df, control_df = self._get_matching_dfs(df, treatment)
        treatment_df = self._prepare_dfs_for_matching(df, control_df, treatment_df)

        dist = self._calc_all_distances(control_df, treatment_df)
        merged = self._find_matches(control_df, treatment_df, dist)

        eta = self._calc_eta(merged, target)
        return eta

    @staticmethod
    def _calc_eta(merged, target):
        merged['eta'] = merged['{}_x'.format(target)] - merged['{}_y'.format(target)]
        eta = merged.eta.mean()
        logger.warning('Estimated ETA: ' + str(eta))
        return eta

    def _find_matches(self, control_df: pd.DataFrame, treatment_df: pd.DataFrame, dist: np.ndarray):
        merged = pd.merge(treatment_df, control_df, left_on='match', right_index=True)
        merged['ind'] = merged.ind.apply(int)
        merged['match'] = merged.match.apply(int)
        merged['dist'] = merged.apply(lambda x: dist[x.match, x.ind], axis=1)
        merged = merged[merged['dist'] < self.max_dist]
        return merged

    def _calc_all_distances(self, control_df: pd.DataFrame, treatment_df: pd.DataFrame):
        logger.warning('Calculating Distances...')
        dist = scipy.spatial.distance.cdist(control_df,
                                            treatment_df[control_df.columns],
                                            metric=self._calc_distance)
        logger.warning('Matching...')
        d, assigment = linear_sum_assignment(dist)
        logger.info('Matching Done!')
        treatment_df.loc[assigment, 'match'] = d
        treatment_df.loc[assigment, 'ind'] = assigment
        return dist

    def _prepare_dfs_for_matching(self,
                                  df: pd.DataFrame,
                                  control_df: pd.DataFrame,
                                  treatment_df: pd.DataFrame,
                                  ):
        logger.warning('Removing Samples without possible matches (this can take a while)...')
        self.cols_to_calc_indices = [df.columns.get_loc(x) for x in self.cols_to_calc]
        self.cols_to_force_indices = [df.columns.get_loc(x) for x in self.cols_to_force]
        treatment_df = self._remove_samples(treatment_df, control_df)
        return treatment_df

    @staticmethod
    def _get_matching_dfs(df: pd.DataFrame, col: str):
        """Split the data frame to contraol and treatment data frames"""
        control_df = df[df[col] == False]
        treatment_df = df[df[col] == True]

        treatment_df.index = range(len(treatment_df))
        control_df.index = range(len(control_df))
        return treatment_df, control_df

    def _calc_distance(self,
                       sample_1,
                       sample_2,
                       ) -> float:
        for col, relation in zip(self.cols_to_force_indices, self.relation_to_force):
            if (sample_1[col] == sample_2[col]) != relation:
                return self.max_dist

        return self.dist(sample_1[self.cols_to_calc_indices], sample_2[self.cols_to_calc_indices])

    def _remove_samples(self, treatment_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
        """To remove samples that have no close match in order to make the matching feasible."""
        treatment_df["closest"] = treatment_df.apply(lambda x: self._find_lowest_dist(x, control_df), axis=1)
        treatment_df = treatment_df[treatment_df.closest < self.max_dist]
        treatment_df.index = range(len(treatment_df))
        treatment_df.drop('closest', axis=1, inplace=True)
        return treatment_df

    def _find_lowest_dist(self,
                          row,
                          data_frame: pd.DataFrame) -> float:
        ans = []
        for i in range(len(data_frame)):
            ans.append(self._calc_distance(row, data_frame.loc[i]))
        return min(ans)
