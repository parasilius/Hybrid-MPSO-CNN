from particle import Particle1

def calculate_omega(t, t_max, a=0.2):
    if t < a * t_max:
        return 0.9
    return 1 / (1 + e ** ((10 * t - t_max) / t_max))

class HybridMPSOCNN:
    def __init__(self, number_of_classes, x_train, y_train, x_test, y_test, m=5, n=8):
        self.m = m
        self.n = n
        self.number_of_classes = number_of_classes
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.number_of_trained_cnns = 0
        self.cnns_trained_time = 0
        self.gbest = None
    
    def summary(self):
        print(f'Number of trained cnn\'s: {self.number_of_trained_cnns}')
        print(f'Average cnn train time: {self.cnns_trained_time / self.number_of_trained_cnns}')
        print('Best Hyperparameters:')
        print(f'Number of convolutional layers (nC): {self.gbest.nC_best}')
        print(f'Number of pooling layers (nP): {self.gbest.nP_best}')
        print(f'Number of fully connected layers (nF): {self.gbest.nF_best}')
        print(f'Number of ﬁlters (c_nf): {self.gbest.best_particle.c_nf_best}')
        print(f'Filter Size (c_fs) (odd): {self.gbest.best_particle.c_fs_best}')
        print(f'Padding pixels (c_pp): {self.gbest.best_particle.c_pp_best}')
        print(f'Stride Size (c_ss)(< c_fs): {self.gbest.best_particle.c_ss_best}')
        print(f'Filter Size (p_fs)(odd): {self.gbest.best_particle.p_fs_best}')
        print(f'Stride Size (p_ss): {self.gbest.best_particle.p_ss_best}')
        print(f'Padding pixels (p_pp) (< p_fs): {self.gbest.best_particle.p_pp_best}')
        print(f'Number of neurons (op): {self.gbest.best_particle.op_best}')
        print(f'Fitness (aka accuracy of best model): {self.gbest.getFitness()}')
    
    def run(self): # based on algorithm 2
        swarm1 = [Particle1(self.n)] * self.m
        # t_max = randint(5, 8)
        t_max = 3
        for t in range(t_max):
            for particle in swarm1:
                nC = particle.nC
                nP = particle.nP
                nF = particle.nF
                best_particle, cnn_counter, cnns_trained_time = particle.calculate_pbest(self.number_of_classes, self.x_train, self.y_train, self.x_test, self.y_test)
                self.number_of_trained_cnns += cnn_counter
                self.cnns_trained_time += cnns_trained_time
                print(f'number of trained cnn\'s so far: {self.number_of_trained_cnns}')
                if best_particle.fitness > particle.getFitness():
                    particle.nC_best = nC
                    particle.nP_best = nP
                    particle.nF_best = nF
                    particle.best_particle = best_particle
                if not self.gbest or best_particle.fitness > self.gbest.getFitness():
                    self.gbest = particle
                w = calculate_omega(t, t_max)
                particle.updateVelocity(w)
                particle.updatePosition()
        print('Done!')