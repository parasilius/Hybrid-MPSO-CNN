import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from math import e, randint

# Range of hyperparameters
range1 = [
    [1, 6], # Number of convolutional layers (nC)
    [1, 6], # Number of pooling layers (nP)
    [1, 6]  # Number of fully connected layers (nF)
]

range2 = [
    [1, 65],  # Number of ﬁlters (c_nf)
    [1, 14],  # Filter Size (c_fs) (odd)
    [0, 2],   # Padding pixels (c_pp)
    [1, 6],   # Stride Size (c_ss)(<c_fs)
    [1, 14],  # Filter Size (p_fs)(odd)
    [1, 6],   # Stride Size (p_ss)
    [0, 2],   # Padding pixels (p_pp) (<p_fs)
    [1, 1025] # Number of neurons (op)
]

# padding mapping
m_padding = {
    0: 'valid',
    1: 'same'
}

class HybridMPSOCNN:
    def __init__(self, m, n):
        self.m = m
        self.n = n
    
    def get_velocity(self, x):
        y = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = randint(range2[i][0] - x[i], range2[i][1] - x[i])
        return y
    
    def update_velocity(self, x, w, c1=2, c2=2):
        y = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = y[i] * w + c1 * r1

    def calculate_pbest(self, i, n, t_max): # algorithm 3
        swarm_sl2 = np.random.randint(low=range2[:, 0], high=range2[:, 1], size=n)
        v = swarm_sl2.apply(lambda x: self.get_velocity(x), axis=1)
        pbest = [float('-inf')] * n
        gbest = float('-inf')
        best_j = -1
        for t in range(t_max):
            for j in range(n):
                c_nf = swarm_sl2[j][0]
                c_fs = swarm_sl2[j][1] if swarm_sl2[j][1] % 2 == 1 else swarm_sl2[j][1] - 1
                c_pp = swarm_sl2[j][2]
                c_ss = swarm_sl2[j][3]
                p_fs = swarm_sl2[j][4]
                p_ss = swarm_sl2[j][5]
                p_pp = swarm_sl2[j][6]
                op   = swarm_sl2[j][7]
                cnn = Sequential()
                cnn.add(Conv2D(c_nf,
                            filters=c_fs,
                            padding=m_padding[c_pp],
                            strides=c_ss))
                cnn.add(MaxPooling2D(pool_size=(p_fs, p_fs),
                                    strides=(p_ss, p_ss),
                                    padding=m_padding[p_pp]))
                cnn.add(Dense(op, activation='softmax'))
                cnn.compile()
                loss, accuracy = cnn.evaluate()
                if accuracy > pbest[i]:
                    pbest[i] = accuracy
                if accuracy > gbest:
                    gbest = accuracy
                    best_j = j
                w = self.calculate_omega(t, t_max)
                v[j] = self.updateVelocity(v, j, w)
        return best_j, gbest
    
    def calculate_omega(self, t, t_max, a=0.2):
        if t < a * t_max:
            return 0.9
        return 1 / (1 + e ** ((10 * t - t_max) / t_max))
    
    def run(self, t_max_1, t_max_2): # algorithm 2
        swarm1 = np.random.randint(low=range1[:, 0], high=range1[:, 1], size=self.m)
        v = swarm1.apply(lambda x: self.get_velocity(x), axis=1)
        pbest = [float('-inf')] * self.m
        gbest = float('-inf')
        best_i = -1
        best_j = -1
        for t in range(t_max_1):
            w = self.calculate_omega(t, t_max_1)
            for i in range(self.m):
                j, fitness = self.calculate_pbest(i, self.n, t_max_2)
                if fitness > pbest[i]:
                    pbest[i] = fitness
                if fitness > gbest:
                    gbest = fitness
                    best_i = i
                    best_j = j