import numpy as np
import copy
from tensorflow.keras.datasets import mnist

def softmax(x):
  max_x = np.max(x)
  exp_x = np.exp(x - max_x)
  return exp_x / np.sum(exp_x)

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))

def act_func(u: float) -> float:
  # return max(0, u)
  return sigmoid(u)

def act_func_prime(z_vec):
  # return_vec = np.zeros(len(z_vec))
  # return_vec[z_vec >= 0] = 1
  # return return_vec
  return_vec = np.zeros(len(z_vec))
  for i in range(0, len(return_vec)):
    return_vec[i] = sigmoid_prime(z_vec[i])
  return return_vec

def loss_func(res_vec, target_vec):
  return np.sum(np.power(res_vec - target_vec, 2)) / 2

def loss_derivative(y_vec, t_vec):
  return y_vec - t_vec

class perceptron:
  # input: size(int), output: 1
  def __init__(self, size: int):
    self.w_vec = np.random.rand(size) / size
    self.bias = 0

  def calc(self, x_vec):
    u = np.sum(np.multiply(x_vec, self.w_vec)) + self.bias
    return (u, act_func(u))

  def train(self, x_vec, d: float, kappa: float):
    # d: target value, kappa: learning coeff
    self.w_vec -= kappa * (self.calc(x_vec) - d) * x_vec
    print('trained')
  
  def set(self, w_vec, bias):
    self.w_vec = w_vec
    self.bias = bias

class layer:
  def __init__(self, prev_node_num: int, percep_num: int):
    self.perceptrons = []
    for i in range(0, percep_num):
      self.perceptrons.append(perceptron(prev_node_num))
    self.z_vec = np.zeros(percep_num) # before act_func
    self.y_vec = np.zeros(percep_num) # after act_func
  
  def forward(self, prev_y_vec):
    for i in range(0, len(self.perceptrons)):
      (this_z, this_y) = self.perceptrons[i].calc(prev_y_vec)
      self.z_vec[i] = this_z
      self.y_vec[i] = this_y
  
class input_layer:
  def __init__(self, size):
    self.y_vec = np.zeros(size)
  
  def set_y(self, input_vec):
    self.y_vec = input_vec

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = np.array(np.float32(test_images)) / 255
train_images = np.array(np.float32(train_images)) / 255
print('mnist loaded')
# print(test_images[0])

network_size = [len(test_images[0].flatten()), 100, 50, 10]
kappa = 0.5

layers = []
layers.append(input_layer(len(test_images[0].flatten())))
for i in range(1, len(network_size)):
  layers.append(layer(network_size[i-1], network_size[i]))
# layers.append(output_layer(network_size[-1]))

# train_images = train_images[0:10]
# train_labels = train_labels[0:10]
def train_network(epoch: int):
  for i_epoch in range(0, epoch):
    print('epoch ' + str(i_epoch))
    y_vec_arr = []
    delta = []
    for (cur_train_image, cur_train_label) in zip(train_images, train_labels):
      (cur_train_image, cur_train_label) = (train_images[0], train_labels[0])
      y_vec_arr.append([])
      layers[0].set_y(cur_train_image.flatten())
      y_vec_arr[-1].append(layers[0].y_vec)
      for i in range(1, len(layers)):
        layers[i].forward(layers[i-1].y_vec)
        y_vec_arr[-1].append(layers[i].y_vec)
      # print(layers[-1].y_vec)
      # print(layers[-1].z_vec)

      train_vector = np.zeros(10)
      train_vector[cur_train_label] = 1
      # print(train_vector)
      # delta = np.zeros(len(layers) + 1, 10) # last element: output
      delta.append([]) # delta[i][j][k]: ith train img, jth layer, kth perceptron(?)
      for i in range(0, len(layers)): # last element: output
        delta[-1].append([])
      delta[-1][-1] = loss_derivative(layers[-1].y_vec, train_vector) * act_func_prime(layers[-1].z_vec)

      for i in reversed(range(1, len(layers) - 1)):
        # print('layer ' + str(i))
        # print('perceptrons ' + str(len(layers[i+1].perceptrons)))
        # print('w ' + str(len(layers[i+1].perceptrons[0].w_vec)))
        # print('delta ' + str(len(delta[-1][i+1])))
        # print('z_vec ' + str(len(layers[i].z_vec)))
        w_mat = np.zeros((len(layers[i+1].perceptrons), len(layers[i+1].perceptrons[0].w_vec)))
        for i_p in range(0, len(layers[i+1].perceptrons)):
          # w_delta[i_p] = np.dot(layers[i+1].perceptrons[i_p].w_vec, delta[i+1])
          w_mat[i_p] = layers[i+1].perceptrons[i_p].w_vec
        w_delta = np.dot(np.transpose(w_mat), delta[-1][i+1])
        delta[-1][i] = w_delta * act_func_prime(layers[i].z_vec)

    # print('delta ' + str(delta[0][-1]))

    for i in range(1, len(layers)):
      w_mat = np.zeros((len(layers[i].perceptrons), len(layers[i].perceptrons[0].w_vec)))
      bias_vec = np.zeros(len(layers[i].perceptrons))
      for i_p in range(0, len(layers[i].perceptrons)):
        w_mat[i_p] = layers[i].perceptrons[i_p].w_vec
        bias_vec[i_p] = layers[i].perceptrons[i_p].bias
      delta_sum = np.zeros(len(delta[0][i]))
      delta_a_sum = np.zeros(w_mat.shape)
      for i_x in range(0, len(train_images)):
        delta_sum += delta[i_x][i]
        # print(w_mat.shape)
        # print(delta[i_x][i].shape)
        # print(y_vec_arr[i_x][i-1].shape)
        # print(np.outer(delta[i_x][i], y_vec_arr[i_x][i-1]))
        delta_a_sum += np.outer(delta[i_x][i], y_vec_arr[i_x][i-1])
      w_new = w_mat - kappa / len(train_images) * delta_a_sum # prev_perceptron * this_perceptronの行列
      # bias_new = bias_vec - kappa / len(train_images) * np.sum(delta, axis=0)
      bias_new = bias_vec - kappa / len(train_images) * delta_sum
      # print('w before ' + str(w_mat))
      # print('delta sum ' + str(delta_a_sum))
      # print('w new ' + str(w_new))
      # print(bias_new)
      for i_p in range(0, len(layers[i].perceptrons)):
        layers[i].perceptrons[i_p].set(w_new[i_p], bias_new[i_p])

train_network(3)
print('training done')
layers[0].set_y(train_images[0].flatten())
# print(layers[0].y_vec)
# layers[1].forward(layers[0].y_vec)
# print(layers[1].perceptrons[0].w_vec)
# print(np.dot(layers[1].perceptrons[0].w_vec, layers[0].y_vec))
# print(layers[1].perceptrons[0].bias)
# print(sigmoid(np.dot(layers[1].perceptrons[0].w_vec, layers[0].y_vec) + layers[1].perceptrons[0].bias))
# print(layers[1].y_vec)
for i in range(1, len(layers)):
  print('forwarding')
  # print(layers[i].y_vec)
  layers[i].forward(layers[i-1].y_vec)
  # print(layers[i].y_vec)
print(softmax(layers[-1].y_vec))
print(train_labels[0])
# for p in layers[2].perceptrons:
#   print(p.w_vec)


# network_size = [8, 5, 1] # 0番目: 入力の数
# layer0_y_vec = np.random.rand(network_size[0])
# layers = []
# layers.append(input_layer(layer0_y_vec))
# for i in range(1, len(network_size)):
#   layers.append(layer(network_size[i-1], network_size[i]))
# print(layer0_y_vec)
# for i in range(1, len(network_size)):
#   print('-----')
#   print(i)
#   layers[i].forward(layers[i-1].y_vec)
#   print(layers[i].y_vec)
# print(layers[-1].y_vec)

# layer1 = layer(5, 3)
# layer0_y_vec = np.random.rand(5)
# print(layer0_y_vec)
# layer1.forward(layer0_y_vec)
# print(layer1.y_vec)

# kappa = 0.5

# per1 = perceptron(5)
# for i in range(0, 5):
#   per1.train(np.array([0, 0.3, 0.3, 1, 0.1]), 0.5, kappa)
#   print(per1.calc(np.array([0, 0.3, 0.3, 1, 0.1])))