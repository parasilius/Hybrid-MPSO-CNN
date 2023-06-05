import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

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
    def __init__(self, m):
        self.m = m
        self.swarm1 = np.random.randint(low=range1[:, 0], high=range1[:, 1], size=self.m)

    def run(self):
        pass

    def calculate_pbest(self, i, n): # algorithm 3
        self.swarm_sl2s = []
        for _ in range(self.m):
            swarm_sl2 = np.random.randint(low=range2[:, 0], high=range2[:, 1], size=n)
            self.swarm_sl2s.append(swarm_sl2)
        pbest = [float('-inf')] * n
        gbest = float('-inf')
        best_i = -1
        for j in range(n):
            c_nf = self.swarm_sl2s[i][j][0]
            c_fs = self.swarm_sl2s[i][j][1] if self.swarm_sl2s[i][j][1] % 2 == 1 else self.swarm_sl2s[i][j][1] - 1
            c_pp = self.swarm_sl2s[i][j][2]
            c_ss = self.swarm_sl2s[i][j][3]
            p_fs = self.swarm_sl2s[i][j][4]
            p_ss = self.swarm_sl2s[i][j][5]
            p_pp = self.swarm_sl2s[i][j][6]
            op = self.swarm_sl2s[i][j][7]
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
            best_i = j
        return best_i, gbest