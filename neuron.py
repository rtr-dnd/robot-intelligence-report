import numpy as np

def act_func(u: float) -> float:
  return max(0, u)

class perceptron:
  def __init__(self, size: int):
    self.w_vec = np.zeros(size)

  # def __init__(self, w_vec):
  #   self.w_vec = w_vec

  def calc(self, x_vec):
    u = np.sum(np.multiply(x_vec, self.w_vec))
    return act_func(u)

  def train(self, x_vec, d: float, kappa: float):
    # d: target value, kappa: learning coeff
    self.w_vec -= kappa * (self.calc(x_vec) - d) * x_vec
    print('trained')

kappa = 0.5
per1 = perceptron(5)
for i in range(0, 5):
  per1.train(np.array([0, 0.3, 0.3, 1, 0.1]), 0.5, kappa)
print(per1.calc(np.array([0, 0.3, 0.3, 1, 0.1])))