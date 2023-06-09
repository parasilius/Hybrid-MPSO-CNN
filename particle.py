from random import randint, random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from math import e
from tensorflow.nn import relu, softmax
import time

# padding mapping
m_padding = {
    0: 'valid',
    1: 'same'
}

def calculate_omega(t, t_max, a=0.2):
    if t < a * t_max:
        return 0.9
    return 1 / (1 + e ** ((10 * t - t_max) / t_max))

class Particle1:
    def __init__(self, n) -> None:
        self.n = n
        self.nC = randint(1, 5)       # Number of convolutional layers (nC)
        self.nP = randint(1, self.nC) # Number of pooling layers (nP)
        self.nF = randint(1, self.nC) # Number of fully connected layers (nF)
        self.v_nC = randint(1 - self.nC, 5 - self.nC)
        self.v_nP = randint(1 - self.nP, 5 - self.nP)
        self.v_nF = randint(1 - self.nF, 5 - self.nF)
        self.nC_best = self.nC
        self.nP_best = self.nP
        self.nF_best = self.nF
        self.best_particle = None
        self.swarm_sl2 = [Particle2()] * self.n
    
    def getFitness(self):
        if self.best_particle:
            return self.best_particle.fitness
        return float('-inf')
    
    def updatePosition(self):
        self.nC = int(self.nC + self.v_nC)
        self.nP = int(self.nP + self.v_nP)
        self.nF = int(self.nF + self.v_nF)
    
    def updateVelocity(self, w, c1=2, c2=2):
        r1 = random()
        r2 = random()
        # updating v_nC
        self.v_nC = w * self.v_nC + c1 * r1 * (self.nC_best - self.nC) + c2 * r2 * (self.nC_best - self.nC)
        v_nC_max = 5 - self.nC
        if self.v_nC > v_nC_max:
            self.v_nC = v_nC_max
        v_nC_min = 1 - self.nC
        if self.v_nC < v_nC_min:
            self.v_nC = v_nC_min
        # updating v_nP
        self.v_nP = w * self.v_nP + c1 * r1 * (self.nP_best - self.nP) + c2 * r2 * (self.nP_best - self.nP)
        v_nP_max = self.nC - self.nP
        if self.v_nP > v_nP_max:
            self.v_nP = v_nP_max
        v_nP_min = 1 - self.nP
        if self.v_nP < v_nP_min:
            self.v_nP = v_nP_min
        # updating v_nF
        self.v_nF = w * self.v_nF + c1 * r1 * (self.nF_best - self.nF) + c2 * r2 * (self.nF_best - self.nF)
        v_nF_max = self.nC - self.nF
        if self.v_nF > v_nF_max:
            self.v_nF = v_nF_max
        v_nF_min = 1 - self.nF
        if self.v_nF < v_nF_min:
            self.v_nF = v_nF_min

    def calculate_pbest(self, number_of_classes, x_train, y_train, x_test, y_test): # based on algorithm 3
        cnn_counter = 0
        cnns_trained_time = 0
        best_particle = None
        # t_max = 5
        t_max = 3
        for t in range(t_max):
            for particle in self.swarm_sl2:
                c_nf = particle.c_nf
                c_fs = particle.c_fs if particle.c_fs % 2 == 1 else particle.c_fs - 1
                c_pp = particle.c_pp
                c_ss = particle.c_ss
                p_fs = particle.p_fs
                p_ss = particle.p_ss
                p_pp = particle.p_pp
                op   = particle.op
                cnn = Sequential()
                nC_counter = 0
                nP_counter = 0
                try:
                    print('cnn')
                    while nP_counter != self.nP and nC_counter != self.nC:
                        if nC_counter != self.nC:
                            if nC_counter == 0:
                                cnn.add(Conv2D(c_nf,
                                               c_fs,
                                               padding=m_padding[c_pp],
                                               strides=c_ss,
                                               activation='relu',
                                               input_shape=(28, 28, 1)))
                            else:
                                cnn.add(Conv2D(c_nf * 2 ** (nC_counter),
                                               c_fs,
                                               padding=m_padding[c_pp],
                                               strides=c_ss,
                                               activation='relu')) 
                            nC_counter += 1
                        if nP_counter != self.nP:
                            cnn.add(MaxPooling2D(pool_size=p_fs,
                                                 strides=p_ss,
                                                 padding=m_padding[p_pp]))
                            nP_counter += 1
                    cnn.add(Flatten())
                    for _ in range(self.nF):
                        cnn.add(Dropout(0.2))
                        cnn.add(Dense(op, activation='relu'))
                    cnn.add(Dropout(0.2))
                    cnn.add(Dense(number_of_classes, activation='softmax'))
                    start_time = time.time()
                    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
                    cnn.fit(x_train, y_train, batch_size=128)
                    cnns_trained_time += time.time() - start_time
                    cnn_counter += 1
                    loss, accuracy = cnn.evaluate(x_test, y_test)
                except:
                    accuracy = float('-inf')
                if accuracy > particle.fitness:
                    particle.fitness = accuracy
                    particle.c_nf_best = c_nf
                    particle.c_fs_best = c_fs
                    particle.c_pp_best = c_pp
                    particle.c_ss_best = c_ss
                    particle.p_fs_best = p_fs
                    particle.p_ss_best = p_ss
                    particle.p_pp_best = p_pp
                    particle.op_best = op
                if not best_particle or accuracy > best_particle.fitness:
                    best_particle = particle
                w = calculate_omega(t, t_max)
                particle.updateVelocity(w, best_particle)
                particle.updatePosition()
        return best_particle, cnn_counter, cnns_trained_time

class Particle2:
    def __init__(self):
        self.c_nf   = randint(1, 64)                                            # Number of ﬁlters (c_nf)
        self.c_fs   = randint(1, 13)                                            # Filter Size (c_fs) (odd)
        self.c_pp   = randint(0, 1)                                             # Padding pixels (c_pp)
        self.c_ss   = randint(1, 5) if self.c_fs > 5 else randint(1, self.c_fs) # Stride Size (c_ss)(<c_fs)
        self.p_fs   = randint(1, 13)                                            # Filter Size (p_fs)(odd)
        self.p_ss   = randint(1, 5)                                             # Stride Size (p_ss)
        self.p_pp   = randint(0, 1)                                             # Padding pixels (p_pp) 
        self.op     = randint(1, 1024)                                          # Number of neurons (op)
        self.v_c_nf = randint(1 - self.c_nf, 64 - self.c_nf) 
        self.v_c_fs = randint(1 - self.c_fs, 13 - self.c_fs)
        self.v_c_pp = randint(0 - self.c_pp, 1 - self.c_pp)
        if self.c_fs > 5:
            self.v_c_ss = randint(1 - self.c_ss, 4 - self.c_ss)
        else:
            self.v_c_ss = randint(1 - self.c_ss, self.c_fs - self.c_ss)
        self.v_p_fs = randint(1 - self.p_fs, 13 - self.p_fs)
        self.v_p_ss = randint(1 - self.p_ss, 5 - self.p_ss)
        if self.p_fs > 2:
            self.v_p_pp = randint(0 - self.p_pp, 1 - self.p_pp)
        else:
            self.v_p_pp = randint(0 - self.p_pp, self.p_fs - self.p_pp)
        self.v_op = randint(1 - self.op, 1024 - self.op)
        self.c_nf_best = self.c_nf
        self.c_fs_best = self.c_fs
        self.c_pp_best = self.c_pp
        self.c_ss_best = self.c_ss
        self.p_fs_best = self.p_fs
        self.p_ss_best = self.p_ss
        self.p_pp_best = self.p_pp
        self.op_best = self.op
        self.fitness = float('-inf')
    
    def updatePosition(self):
        self.c_nf = int(self.c_nf + self.v_c_nf)
        self.c_fs = int(self.c_fs + self.v_c_fs)
        self.c_pp = int(self.c_pp + self.v_c_pp)
        self.c_ss = int(self.c_ss + self.v_c_ss)
        self.p_fs = int(self.p_fs + self.v_p_fs)
        self.p_ss = int(self.p_ss + self.v_p_ss)
        self.p_pp = int(self.p_pp + self.v_p_pp)
        self.op   = int(self.op + self.v_op)
    
    def updateVelocity(self, w, best_particle, c1=2, c2=2):
        r1 = random()
        r2 = random()
        # updating v_c_nf
        self.v_c_nf = w * self.v_c_nf + c1 * r1 * (self.c_nf_best - self.c_nf) + c2 * r2 * (best_particle.c_nf_best - self.c_nf)
        v_c_nf_max = 64 - self.c_nf
        if self.v_c_nf > v_c_nf_max:
            self.v_c_nf = v_c_nf_max
        v_c_nf_min = 1 - self.c_nf
        if self.v_c_nf < v_c_nf_min:
            self.v_c_nf = v_c_nf_min
        # updating v_c_fs
        self.v_c_fs = w * self.v_c_fs + c1 * r1 * (self.c_fs_best - self.c_fs) + c2 * r2 * (best_particle.c_fs_best - self.c_fs)
        v_c_fs_max = 13 - self.c_fs
        if self.v_c_fs > v_c_fs_max:
            self.v_c_fs = v_c_fs_max
        v_c_fs_min = 1 - self.c_fs
        if self.v_c_fs < v_c_fs_min:
            self.v_c_fs = v_c_fs_min
        # updating v_c_pp
        self.v_c_pp = w * self.v_c_pp + c1 * r1 * (self.c_pp_best - self.c_pp) + c2 * r2 * (best_particle.c_pp_best - self.c_pp)
        v_c_pp_max = 1 - self.c_pp
        if self.v_c_pp > v_c_pp_max:
            self.v_c_pp = v_c_pp_max
        v_c_pp_min = 0 - self.c_pp
        if self.v_c_pp < v_c_pp_min:
            self.v_c_pp = v_c_pp_min
        # updating v_c_ss
        self.v_c_ss = w * self.v_c_ss + c1 * r1 * (self.c_ss_best - self.c_ss) + c2 * r2 * (best_particle.c_ss_best - self.c_ss)
        v_c_ss_max = self.c_fs - 1 - self.c_ss
        if self.v_c_ss > v_c_ss_max:
            self.v_c_ss = v_c_ss_max
        v_c_ss_min = 1 - self.c_ss
        if self.v_c_ss < v_c_ss_min:
            self.v_c_ss = v_c_ss_min
        # updating v_p_fs
        self.v_p_fs = w * self.v_p_fs + c1 * r1 * (self.p_fs_best - self.p_fs) + c2 * r2 * (best_particle.p_fs_best - self.p_fs)
        v_p_fs_max = 13 - self.p_fs
        if self.v_p_fs > v_p_fs_max:
            self.v_p_fs = v_p_fs_max
        v_p_fs_min = 1 - self.p_fs
        if self.v_p_fs < v_p_fs_min:
            self.v_p_fs = v_p_fs_min
        # updating v_p_ss
        self.v_p_ss = w * self.v_p_ss + c1 * r1 * (self.p_ss_best - self.p_ss) + c2 * r2 * (best_particle.p_ss_best - self.p_ss)
        v_p_ss_max = self.p_fs - 1 - self.p_ss # right ?
        v_p_ss_max = 5 - self.p_ss
        if self.v_p_ss > v_p_ss_max:
            self.v_p_ss = v_p_ss_max
        v_p_ss_min = 1 - self.p_ss
        if self.v_p_ss < v_p_ss_min:
            self.v_p_ss = v_p_ss_min
        # updating v_p_pp
        self.v_p_pp = w * self.v_p_pp + c1 * r1 * (self.p_pp_best - self.p_pp) + c2 * r2 * (best_particle.p_pp_best - self.p_pp)
        v_p_pp_max = 1 - self.p_pp
        if self.v_p_pp > v_p_pp_max:
            self.v_p_pp = v_p_pp_max
        v_p_pp_min = 0 - self.p_pp
        if self.v_p_pp < v_p_pp_min:
            self.v_p_pp = v_p_pp_min
        # updating v_op
        self.v_op = w * self.v_op + c1 * r1 * (self.op_best - self.op) + c2 * r2 * (best_particle.op_best - self.op)
        v_op_max = 1024 - self.op
        if self.v_op > v_op_max:
            self.v_op = v_op_max
        v_op_min = 1 - self.op
        if self.v_op < v_op_min:
            self.v_op = v_op_min