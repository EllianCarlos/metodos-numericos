import numpy as np  ## usando para visualização, operações lineares e calculo de erros
from visualizer import Visualizer
from utils import vectorize_column_matrice

from escalonamento import escalonamento
from jacobi import jacobis_method
from gauss_seidel import gauss_seidel_method
from relaxamento import relaxation_method
from conjugate_squared_gradient import gradient_conj_square
from conjugate_gradient import gradient_conj

# TODO:
## - Adicionar Tempo
## - Corrigir critério de parada
## - Adicionar L, K e euclidiana norma da matriz A
## - Corrigir GCQ


def run_method(name, func, vis, A, b):
  print(f'Running method {name}')

  max_iter = 1000
  tol = 1e-4

  x = func(A, b, vis, max_iter, tol)
  #mse = (np.square(x - x_expected)).mean(axis=0)
  #print(f'Mean Squared Error between exptected and found {mse}')

  print(f'Finished running method {name}')

  return x


def transform_list_str_to_list_float(row):
  return [float(number.replace(",", ".")) for number in row]


def generate_matrice_from_text(txt):
  txt_split = txt.split("\n")
  return_txt = [
      transform_list_str_to_list_float(row.split("\t")) for row in txt_split
  ]
  return return_txt


if __name__ == "__main__":
  A_input = np.array([[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4],
                      [1 / 3, 1 / 4, 1 / 5]]).astype(np.float64)

  b_input = np.array([[0.4167], [0.1167], [0.05]]).astype(np.float64)

  x_expected = np.array([10491 / 10000, -3987 / 2500,
                         99 / 200]).astype(np.float64)

  A_input = np.array(
      generate_matrice_from_text(
          '''1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  -0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-0,3667	-0,4667	-0,3667	0	0	0	-0,3333	2,6667	-0,3333	0	0	0	-0,3	-0,2	-0,3
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0
  0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1'''))

  b_input = np.array(
      generate_matrice_from_text('''0
      0.2000
      0.4000
      0.6000
      0.8000
      1.0000
           0
           0
           0
           0
           0
      1.0000
           0
           0
           0
           0
           0
      1.0000
           0
           0
           0
           0
           0
      1.0000
           0
           0
           0
           0
           0
      1.0000
           0
      0.2000
      0.4000
      0.6000
      0.8000
      1.0000'''))

  x_expected = vectorize_column_matrice(np.linalg.solve(A_input, b_input))

  vis = Visualizer()

  esc_x = run_method("Escalonamento", escalonamento, vis, A_input, b_input)
  jacobi_x = run_method("Jacobi", jacobis_method, vis, A_input, b_input)
  gauss_x = run_method("Gauss-Seidel", gauss_seidel_method, vis, A_input,
                       b_input)
  relax_x = run_method("Relaxamento", relaxation_method, vis, A_input, b_input)
  grad_x = run_method("GC", gradient_conj, vis, A_input, b_input)
  conj_grad_x = run_method("CGS", gradient_conj_square, vis, A_input, b_input)

  ## Erro relativo em relação a resposta exata
  vis.set_limits(-1, 30, -0.2, 1.15)
  vis.print_all_from_name("Convergence")

  vis.print_errors(
      x_expected, [esc_x, jacobi_x, gauss_x, relax_x, grad_x, conj_grad_x], [
          "Escalonamento", "Jacobi", "Gauss-Seidel", "Relaxamento",
          "Gradiente\nConjugado", "Gradiente\nConjugado\nQuadrado"
      ])

  print(grad_x)
  print(x_expected)
