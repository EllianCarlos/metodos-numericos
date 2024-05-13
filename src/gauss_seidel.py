import numpy as np  ## usando para visualização, operações lineares e calculo de erros
from utils import vectorize_column_matrice


def gauss_seidel_method(A, b, vis, max_iter=10000, tol=1e-3):
  GS_NAME = "Gauss-Seidel"
  CONVERGENCE_NAME = "Convergence"

  vis.add_method(GS_NAME)

  gs_x = np.zeros(b.shape)
  last_iter = np.copy(gs_x)

  n = b.shape[0]

  n_iter = 0
  error = 1e6

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
            sum_k += A[i][j] * gs_x[j]
          else:
            sum_k_next += A[i][j] * last_iter[j]
      gs_x[i] = (b[i] - sum_k_next - sum_k) / A[i][i]
    error = np.linalg.norm(gs_x - last_iter, 1) / np.linalg.norm(gs_x, 1)
    vis.add_result(GS_NAME, CONVERGENCE_NAME, n_iter, error)
    last_iter = np.copy(gs_x)
    n_iter += 1
  return vectorize_column_matrice(gs_x)
