import numpy as np  ## usando para visualização, operações lineares e calculo de erros


def escalonamento(A, b, _vis, _max_iter=0, _tol=1e-6):
  # Gerar matrix aumentada para o escalonamento [A, b]
  augmentated_a = np.concatenate((np.copy(A), np.copy(b)), 1)

  N = augmentated_a.shape[0]

  def swap_row(mat, i, j):
    for k in range(N + 1):
      temp = mat[i][k]
      mat[i][k] = mat[j][k]
      mat[j][k] = temp

  def foward_elimination(mat):
    for k in range(N):
      i_max = k
      v_max = mat[i_max][k]

      for i in range(k + 1, N):
        if (abs(mat[i][k]) > v_max):
          v_max = mat[i][k]
          i_max = i
      if not mat[k][i_max]:
        return k
      if (i_max != k):
        swap_row(mat, k, i_max)
      for i in range(k + 1, N):
        f = mat[i][k] / mat[k][k]
        for j in range(k + 1, N + 1):
          mat[i][j] -= mat[k][j] * f
        mat[i][k] = 0
    return -1

  def back_substitution(mat):
    x = np.array([0.0 for _ in range(N)])
    for i in range(N - 1, -1, -1):
      x[i] = mat[i][N]
      for j in range(i + 1, N):
        x[i] -= mat[i][j] * x[j]
      x[i] = (x[i] / mat[i][i])
    return x

  def gaussianElimination(mat):
    singular_flag = foward_elimination(mat)
    if (singular_flag != -1):
      if (mat[singular_flag][N]):
        print("Inconsistent System.")
      else:
        print("May have infinitely many solutions.")

      return
    return back_substitution(mat)

  return gaussianElimination(augmentated_a)
