import numpy as np
from utils import vectorize_column_matrice


def relaxation_method(A, b, vis, max_iter=10000, tol=1e-3, omega=1.4):
  RELAX_NAME = "Relaxamento"
  CONVERGENCE_NAME = "Convergence"

  vis.add_method(RELAX_NAME)

  x = np.zeros(b.shape)
  last_iter = np.copy(x)

  n_iter = 0
  error = 1e6

  n = A.shape[0]

  ## Duas condições de paradas
  ### número de iterações excede o número máximo de iterações
  ### error é suficientemente pequeno
  while n_iter < max_iter and error > tol:
    for i in range(n):
      sum_k = 0
      sum_k_next = 0
      for j in range(n):
        if j != i:
          if j < i:
            sum_k += A[i][j] * x[j]
          else:
            sum_k_next += A[i][j] * last_iter[j]
      x[i] = (1 - omega) * last_iter[i] + omega * (b[i] - sum_k -
                                                   sum_k_next) / A[i][i]
    error = np.linalg.norm(x - last_iter, 1) / np.linalg.norm(x, 1)
    vis.add_result(RELAX_NAME, CONVERGENCE_NAME, n_iter, error)
    last_iter = np.copy(x)
    n_iter += 1
  return vectorize_column_matrice(x)
