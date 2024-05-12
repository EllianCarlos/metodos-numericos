from matplotlib import pyplot as plt


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
    plt.legend(handles, self.results.keys(), loc='upper right')
    plt.show()

  def print_all_from_name(self, name):
    plt.legend(loc='upper right')
    for method in self.results:
      plt.plot(self.results[method][name]["x"],
               self.results[method][name]["y"],
               label=f'{method}')
    plt.show()

  def set_limits(self, xmin=None, xmax=None, ymin=None, ymax=None):
    ax = plt.gca()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

  def reset(self):
    self.results = dict()
    self.methods = set()
