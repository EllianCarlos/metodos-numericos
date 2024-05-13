from matplotlib import pyplot as plt
import numpy as np


class Visualizer:

  def __init__(self):
    self.results = dict()
    self.methods = set()

  def add_method(self, method: str):
    if method in self.methods:
      return

    self.methods.add(method)
    self.results[method] = dict()

  def add_result(self, method: str, name, xv, yv):
    if method not in self.methods:
      raise Exception("method not defined in class")
    if name not in self.results[method]:
      self.results[method][name] = dict(x=[xv], y=[yv])
      return

    self.results[method][name]["x"].append(xv)
    self.results[method][name]["y"].append(yv)

  def print_result(self, method, name):
    handles = []
    plot_handle, = plt.plot(self.results[method][name]["x"],
                            self.results[method][name]["y"],
                            label=f'{method}')
    handles.append(plot_handle)
    plt.title("Comparação entre métodos")
    plt.xlabel("Número de Iterações")
    plt.ylabel("Erro absoluto ||x^(k+1) - x^(k)|| ")
    plt.legend(handles, self.results.keys(), loc='upper right')
    plt.show()

  def print_all_from_name(self, name):
    for method in self.results:
      plt.plot(self.results[method][name]["x"],
               self.results[method][name]["y"],
               label=f'{method}')
    plt.title("Gráficos de convergência")
    plt.legend(loc='upper right')
    plt.xlabel("Número de Iterações")
    plt.ylabel(
        r"Erro relativo $\dfrac{||x^{(k+1)} - x^{(k)}||}{||x^{(k+1)}||}$")
    plt.show()

  def set_limits(self, xmin=None, xmax=None, ymin=None, ymax=None):
    ax = plt.gca()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

  def print_errors(self, expected_x, given_x, labels):
    errors = [(np.linalg.norm(x - expected_x, ord=1) /
               np.linalg.norm(expected_x, ord=1)) for x in given_x]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    fig.set_dpi(80)
    ax.bar(labels, errors, width=0.1)
    ax.set_title("Comparação de erro entre os métodos")
    ax.set_ylabel(
        r"Erro relativo $\dfrac{||x_{real} - x_{encontrado}||}{||x_{real}||}$")
    plt.show()

  def reset(self):
    self.results = dict()
    self.methods = set()
