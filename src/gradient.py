import numpy as np  ## usando para visualização, operações lineares e calculo de erros
from utils import vectorize_column_matrice


def gradient_method(A, b, vis, max_iter=10000, tol=1e-6):
  GRAD_NAME = "Gradient"
  CONVERGENCE_NAME = "Convergence"

  vis.add_method(GRAD_NAME)

  x = np.zeros(b.shape)
  last_iter = np.copy(x)
  r_0 = b - np.dot(A, x)

  n_iter = 0
  error = 1e6

  ## Duas condições de paradas
  ### número de iterações excede o número máximo de iterações
  ### error é suficientemente pequeno
  while n_iter < max_iter and error > tol:
    gradient = np.dot(A, x) - b
    lamb = np.dot(gradient.T, gradient) / np.dot(np.dot(gradient.T, A),
                                                 gradient)
    x = x - lamb * gradient
    error = np.linalg.norm(x - last_iter, ord=1)
    vis.add_result(GRAD_NAME, CONVERGENCE_NAME, n_iter, error)
    last_iter = np.copy(x)
    n_iter += 1
  return vectorize_column_matrice(x)
