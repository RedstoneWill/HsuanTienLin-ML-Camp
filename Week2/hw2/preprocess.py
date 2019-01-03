# hw2
import numpy as np

# generate random data
def generate_input_data(time_seed):
    np.random.seed(time_seed)
    raw_X = np.sort(np.random.uniform(-1, 1, 20))
    # 加20%噪声
    noised_y = np.sign(raw_X) * np.where(np.random.random(raw_X.shape[0]) < 0.2, -1, 1)
    return raw_X, noised_y

# load the data
def read_input_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        items = line.strip().split(' ')
        tmp_x = []
        for i in range(0, len(items) - 1): tmp_x.append(float(items[i]))
        x.append(tmp_x)
        y.append(float(items[-1]))
    return np.array(x), np.array(y)

