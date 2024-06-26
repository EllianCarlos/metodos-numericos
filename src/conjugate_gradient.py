import numpy as np  ## usando para visualização, operações lineares e calculo de erros
from utils import vectorize_column_matrice


def gradient_conj(A, b, vis, max_iter=10, tol=1e-3):
  GRAD_CONJ_NAME = "GC"
  CONVERGENCE_NAME = "Convergence"

  vis.add_method(GRAD_CONJ_NAME)

  x = np.zeros(b.shape)
  last_iter = np.copy(x)
  d = b - A * x
  r = np.copy(d)
  last_r = r

  n_iter = 0
  error = 1e6

  n = A.shape[0]
  m = A.shape[1]

  ## Duas condições de paradas
  ### número de iterações excede o número máximo de iterações
  ### error é suficientemente pequeno
  while n_iter < max_iter and error > tol:
    Ad = np.dot(A, d)
    alpha = np.dot(r.T, r) / np.dot(d.T, Ad)
    x = x + alpha * d
    last_r = np.copy(r)
    r = r - alpha * Ad
    beta = np.dot(r.T, r) / np.dot(last_r.T, last_r)
    d = r + beta * d
    error = np.linalg.norm(x - last_iter, ord=1) / np.linalg.norm(x, 1)
    vis.add_result(GRAD_CONJ_NAME, CONVERGENCE_NAME, n_iter, error)
    last_iter = np.copy(x)
    n_iter += 1
  print(n_iter)
  return vectorize_column_matrice(x)
