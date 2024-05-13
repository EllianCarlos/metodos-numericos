import numpy as np  ## usando para visualização, operações lineares e calculo de erros
from utils import vectorize_column_matrice


def jacobis_method(A, b, vis, max_iter=10000, tol=1e-3):
  JACOBI_NAME = "Jacobi"
  CONVERGENCE_NAME = "Convergence"

  vis.add_method(JACOBI_NAME)

  jacobis_x = np.zeros(b.shape)
  last_iter = np.copy(b)

  n_iter = 0
  error = 1e6

  n = A.shape[0]
  m = A.shape[1]

  ## Duas condições de paradas
  ### número de iterações excede o número máximo de iterações
  ### error é suficientemente pequeno
  while n_iter < max_iter and error > tol:
    for i in range(n):
      sum = 0
      for j in range(n):
        if i != j:
          sum += A[i][j] * jacobis_x[j]
      jacobis_x[i] = (b[i] - sum) / A[i][i]
    error = np.linalg.norm(jacobis_x - last_iter, 1)/np.linalg.norm(jacobis_x, 1)
    vis.add_result(JACOBI_NAME, CONVERGENCE_NAME, n_iter, error)
    last_iter = np.copy(jacobis_x)
    n_iter += 1
  return vectorize_column_matrice(jacobis_x)
