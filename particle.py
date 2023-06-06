from random import randint, random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from math import e

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
    def __init__(self, n, nC=randint(1, 6), nP=randint(1, 6), nF=randint(1, 6)) -> None:
        self.n = n
        self.nC = nC # Number of convolutional layers (nC)
        self.nP = nP # Number of pooling layers (nP)
        self.nF = nF # Number of fully connected layers (nF)
        self.v_nC = randint(1 - self.nc, 6 - self.nc)
        self.v_nP = randint(1 - self.nc, 6 - self.nc)
        self.v_nF = randint(1 - self.nc, 6 - self.nc)
        self.nC_best = nC
        self.nP_best = nP
        self.nF_best = nF
        self.best_particle = None
    
    def getFitness(self):
        if self.best_particle:
            return self.best_particle.fitness
        return float('-inf')
    
    def updatePosition(self):
        self.nC += self.v_nC
        self.nP += self.v_nP
        self.nF += self.v_nF
    
    def updateVelocty(self, w, best_particle, c1=2, c2=2):
        r1 = random()
        r2 = random()
        # updating v_nC
        self.v_nC = w * self.v_nC + c1 * r1 * (self.nC_best - self.nC) + c2 * r2 * (best_particle.nC_best - self.nC)
        v_nC_max = 5 - self.nC
        if self.v_nC > v_nC_max:
            self.v_nC = v_nC_max
        v_nC_min = 1 - self.nC
        if self.v_nC < v_nC_min:
            self.v_nC = v_nC_min
        # updating v_nP
        self.v_nP = w * self.v_nP + c1 * r1 * (self.nP_best - self.nP) + c2 * r2 * (best_particle.nP_best - self.nP)
        v_nP_max = 5 - self.nP
        if self.v_nP > v_nP_max:
            self.v_nP = v_nP_max
        v_nP_min = 1 - self.nP
        if self.v_nP < v_nP_min:
            self.v_nP = v_nP_min
        # updating v_nF
        self.v_nF = w * self.v_nF + c1 * r1 * (self.nF_best - self.nF) + c2 * r2 * (best_particle.nF_best - self.nF)
        v_nF_max = 5 - self.nF
        if self.v_nF > v_nF_max:
            self.v_nF = v_nF_max
        v_nF_min = 1 - self.nF
        if self.v_nF < v_nF_min:
            self.v_nF = v_nF_min

    def calculate_pbest(self, t_max): # algorithm 3
        swarm_sl2 = [Particle2() * self.n]
        best_particle = None
        for t in range(t_max):
            for particle in swarm_sl2:
                c_nf = particle.c_nf
                c_fs = particle.c_fs if particle.c_fs % 2 == 1 else particle.c_fs - 1
                c_pp = particle.c_pp
                c_ss = particle.c_ss
                p_fs = particle.p_fs
                p_ss = particle.p_ss
                p_pp = particle.p_pp
                op   = particle.op
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
                w = self.calculate_omega(t, t_max)
                particle.updatePosition()
                particle.updateVelocity(w, best_particle)
        return best_particle

class Particle2:
    def __init__(self, c_nf=randint(1, 65), c_fs=randint(1, 14), c_pp=randint(0, 2), c_ss=randint(1, 6),
                                            p_fs=randint(1, 14), p_ss=randint(1, 6), p_pp=randint(0, 2),
                                            op=randint(1, 1025)) -> None:
        self.c_nf = c_nf # Number of ﬁlters (c_nf)
        self.c_fs = c_fs # Filter Size (c_fs) (odd)
        self.c_pp = c_pp # Padding pixels (c_pp)
        self.c_ss = c_ss # Stride Size (c_ss)(<c_fs)
        self.p_fs = p_fs # Filter Size (p_fs)(odd)
        self.p_ss = p_ss # Stride Size (p_ss)
        self.p_pp = p_pp # Padding pixels (p_pp) (<p_fs)
        self.op = op     # Number of neurons (op)
        self.v_c_nf = randint(1 - c_nf, 65 - c_nf) 
        self.v_c_fs = randint(1 - c_fs, 14 - c_fs)
        self.v_c_pp = randint(0 - c_pp, 2 - c_pp)
        self.v_c_ss = randint(1 - c_ss, 6 - c_ss)
        self.v_p_fs = randint(1 - p_fs, 14 - p_fs)
        self.v_p_ss = randint(1 - p_ss, 6 - p_ss)
        self.v_p_pp = randint(0 - p_pp, 2 - p_pp)
        self.v_op = randint(1 - op, 1025 - op)
        self.c_nf_best = c_nf
        self.c_fs_best = c_fs
        self.c_pp_best = c_pp
        self.c_ss_best = c_ss
        self.p_fs_best = p_fs
        self.p_ss_best = p_ss
        self.p_pp_best = p_pp
        self.op_best = op
        self.fitness = float('-inf')
    
    def updatePosition(self):
        self.c_nf += self.v_c_nf
        self.c_fs += self.v_c_fs
        self.c_pp += self.v_c_pp
        self.c_ss += self.v_c_ss
        self.p_fs += self.v_p_fs
        self.p_ss += self.v_p_ss
        self.p_pp += self.v_p_pp
        self.op += self.v_op
    
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
        v_c_ss_max = 5 - self.c_ss
        if self.v_c_ss > v_c_ss_max:
            self.v_c_ss = v_c_ss_max
        v_c_ss_min = 1 - self.c_ss
        if self.v_c_ss < v_c_ss_min:
            self.v_c_ss = v_c_ss_min
        # updating v_c_fs
        self.v_c_fs = w * self.v_c_fs + c1 * r1 * (self.c_fs_best - self.c_fs) + c2 * r2 * (best_particle.c_fs_best - self.c_fs)
        v_c_fs_max = 13 - self.c_fs
        if self.v_c_fs > v_c_fs_max:
            self.v_c_fs = v_c_fs_max
        v_c_fs_min = 1 - self.c_fs
        if self.v_c_fs < v_c_fs_min:
            self.v_c_fs = v_c_fs_min
        # updating v_p_ss
        self.v_p_ss = w * self.v_p_ss + c1 * r1 * (self.p_ss_best - self.p_ss) + c2 * r2 * (best_particle.p_ss_best - self.p_ss)
        v_p_ss_max = 5 - self.p_ss
        if self.v_p_ss > v_p_ss_max:
            self.v_p_ss = v_p_ss_max
        v_p_ss_min = 0 - self.p_ss
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
        # updating v_p_pp
        self.v_op = w * self.v_op + c1 * r1 * (self.op_best - self.op) + c2 * r2 * (best_particle.op_best - self.p_op)
        v_op_max = 1024 - self.op
        if self.v_op > v_op_max:
            self.v_op = v_op_max
        v_op_min = 1 - self.op
        if self.v_op < v_op_min:
            self.v_op = v_op_min