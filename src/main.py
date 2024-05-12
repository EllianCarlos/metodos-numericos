import numpy as np  ## usando para visualização, operações lineares e calculo de erros
from visualizer import Visualizer

from escalonamento import escalonamento
from jacobi import jacobis_method
from gauss_seidel import gauss_seidel_method
from relaxamento import relaxation_method
from gradient import gradient_method
from conjugate_gradient import gradient_conj_square

def run_method(name, func, vis, A, b):
  print(f'Running method {name}')

  max_iter = 1000
  tol = 1e-6

  x = func(A, b, vis, max_iter, tol)
  mse = (np.square(x - x_expected)).mean(axis=0)
  print(f'Mean Squared Error between exptected and found {mse}')
  
  print(f'Finished running method {name}') 


if __name__ == "__main__":
  A_input = np.array([[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4],
                      [1 / 3, 1 / 4, 1 / 5]]).astype(np.float64)

  b_input = np.array([[0.4167], [0.1167], [0.05]]).astype(np.float64)

  x_expected = np.array([10491 / 10000, -3987 / 2500,
                         99 / 200]).astype(np.float64)

  vis = Visualizer()

  run_method("Escalonamento", escalonamento, vis, A_input, b_input)
  run_method("Jacobi", jacobis_method, vis, A_input, b_input)
  run_method("Gauss-Seidel", gauss_seidel_method, vis, A_input, b_input)
  run_method("Relaxamento", relaxation_method, vis, A_input, b_input)
  run_method("Gradiente", gradient_method, vis, A_input, b_input)
  run_method("Gradiente Conjugado", gradient_conj_square, vis, A_input, b_input)

  vis.set_limits(-1, 80, -0.2, 2.5)
  vis.print_all_from_name("Convergence")
