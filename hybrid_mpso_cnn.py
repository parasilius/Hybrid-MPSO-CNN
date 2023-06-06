from particle import Particle1

class HybridMPSOCNN:
    def __init__(self, m, n):
        self.m = m
        self.n = n
    
    def run(self, t_max_1, t_max_2): # algorithm 2
        swarm1 = [Particle1(self.n)] * self.m
        gbest = None
        for t in range(t_max_1):
            for particle in swarm1:
                nC = particle.nC
                nP = particle.nP
                nF = particle.nF
                best_particle = particle.calculate_pbest()
                if best_particle.fitness > particle.getFitness():
                    particle.nC_best = nC
                    particle.nP_best = nP
                    particle.nF_best = nF
                    particle.best_particle = best_particle
                if not gbest or best_particle.fitness > gbest.getFitness():
                    gbest = particle
                particle.updatePosition()
                particle.updateVelocty()