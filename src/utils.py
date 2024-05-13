from typing import List
import numpy as np


def vectorize_column_matrice(matrice: List[List[float]]) -> List[float]:
  return np.array([row[0] for row in matrice])
