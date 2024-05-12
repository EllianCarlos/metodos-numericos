from typing import List


def vectorize_column_matrice(matrice: List[List[float]]) -> List[float]:
  return [row[0] for row in matrice]
