from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class AteEstimatorInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate_ate(self,
                     df: pd.DataFrame,
                     treatment: str,
                     target: str,
                     features: List[str],
                     **kwargs
                     ) -> float:
        pass


