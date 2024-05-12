import numpy as np


def relaxation_method(A, b, vis, omega=1.3, max_iter=10000, tol=1e-3):
  RELAX_NAME = "Relaxamento"
  CONVERGENCE_NAME = "Convergence"

  vis.add_method(RELAX_NAME)

  x = np.copy(b)
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
      for j in range(n):
        if j != i:
          sum_k += A[i][j] * x[j]
      x[i] = (1 - omega) * last_iter[i] + omega * (b[i] - sum_k) / A[i][i]
    error = np.linalg.norm(x - last_iter, 1)
    vis.add_result(RELAX_NAME, CONVERGENCE_NAME, n_iter, error)
    last_iter = np.copy(x)
    n_iter += 1
  return x
