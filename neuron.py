import numpy as np

def act_func(u: float) -> float:
  return max(0, u)

class perceptron:

  # input: size(int), output: 1
  def __init__(self, size: int):
    self.w_vec = np.random.rand(size)

  # def __init__(self, w_vec):
  #   self.w_vec = w_vec

  def calc(self, x_vec):
    u = np.sum(np.multiply(x_vec, self.w_vec))
    print(act_func(u))
    return act_func(u)

  def train(self, x_vec, d: float, kappa: float):
    # d: target value, kappa: learning coeff
    self.w_vec -= kappa * (self.calc(x_vec) - d) * x_vec
    print('trained')

class layer:
  perceptrons = []

  def __init__(self, prev_node_num: int, percep_num: int):
    for i in range(0, percep_num):
      self.perceptrons.append(perceptron(prev_node_num))
    self.y_vec = np.zeros(percep_num)
  
  def forward(self, prev_y_vec):
    for i in range(0, len(self.perceptrons)):
      self.y_vec[i] = self.perceptrons[i].calc(prev_y_vec)


layer1 = layer(5, 3)
layer0_y_vec = np.random.rand(5)
print(layer0_y_vec)
layer1.forward(layer0_y_vec)
print(layer1.y_vec)

# kappa = 0.5

# per1 = perceptron(5)
# for i in range(0, 5):
#   per1.train(np.array([0, 0.3, 0.3, 1, 0.1]), 0.5, kappa)
#   print(per1.calc(np.array([0, 0.3, 0.3, 1, 0.1])))