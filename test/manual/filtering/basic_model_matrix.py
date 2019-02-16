""" Particle filtering for a trivial model
    Also illustrates that the """

import numpy
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps + 1,2))
    y = numpy.zeros((steps,2))
    x[0] = numpy.random.normal(0.0, R[0], (2,)).reshape((1, 2))
    for k in range(1, steps + 1):
        x[k] = x[k - 1] +  numpy.random.normal(1.0, Q, (2,)).reshape((1, 2))
        y[k - 1] = x[k] +  5*numpy.random.normal(0.0, R[0], (2,)).reshape((1, 2))
    return (x, y)

class Model(interfaces.ParticleFiltering):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        self.P0 = numpy.copy(P0)
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)

    def create_initial_estimate(self, N):
        return numpy.random.normal(0.0, self.P0, (2*N,)).reshape(N, 2)

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return numpy.random.normal(0.0, self.Q, (2*N,)).reshape((N, 2))

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        particles += noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        logyprob = numpy.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            logyprob[k] = kalman.lognormpdf(numpy.linalg.norm(particles[k] - y), self.R)
        return logyprob

    # def logp_xnext_full(self, part, past_trajs, pind,
    #                     future_trajs, find, ut, yt, tt, cur_ind):
    #
    #     diff = future_trajs[0].pa.part[find] - part
    #
    #     logpxnext = numpy.empty(len(diff), dtype=float)
    #     for k in range(len(logpxnext)):
    #         logpxnext[k] = kalman.lognormpdf(diff[k].reshape(-1, 1), numpy.asarray(self.Q).reshape(1, 1))
    #     return logpxnext



if __name__ == '__main__':
    steps = 50
    num = 200
    P0 = 1.0
    Q = 1.0
    R = numpy.asarray(((1.0,),))

    # Make realization deterministic
    numpy.random.seed(100)
    (x, y) = generate_dataset(steps, P0, Q, R)

    model = Model(P0, Q, R)
    sim = simulator.Simulator(model, u=None, y=y)
    numOfResampling = sim.simulate(num, num, smoother='ancestor')
    print("Number of resampling that occured:",numOfResampling)

    plt.plot(x[:,0], x[:,1], 'r-') # Real data
    # plt.plot(y[:,0], y[:,1], 'x--') # Measured data
    plt.plot(y[:,0], y[:,1], 'x') # Measured data

    fvals = sim.get_filtered_mean()
    plt.plot(fvals[:,0], fvals[:,1], 'g-') # Filtered data

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
