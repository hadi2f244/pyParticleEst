""" Particle filtering for a trivial model
    Also illustrates that the """

import numpy
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator
import pyparticleest.filter as filter
from draw import Maze
import math
import random


def create_world(maze_data):
    world = Maze(maze_data)
    world.draw()
    return world

def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]

def add_little_noise(*coords):
    return add_noise(0.02, *coords)


class RealRobot(object):

    def __init__(self, world, speed):
        self.world = world
        self.x, self.y = world.random_free_place()
        self.h = 90
        self.chose_random_direction()
        self.step_count = 0
        self.speed = speed

    @property
    def xy(self):
        return self.x, self.y

    def chose_random_direction(self):
        heading = random.uniform(0, 360)
        self.h = heading

    def read_odometry(self,Q):
        h = numpy.random.normal(self.h, Q[0])
        speed = numpy.random.normal(self.speed, Q[1])
        return numpy.array([h, speed])

    def read_sensor(self):
        return add_little_noise(self.world.distance_to_nearest_beacon(*self.xy))[0]

    def move(self):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            if self.advance_by(self.speed, noisy=True,
                checker=lambda r, dx, dy: self.world.is_free(r.x+dx, r.y+dy)):
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction()

    def advance_by(self, speed, checker=None, noisy=False):
        h = self.h
        if noisy:
            speed, h = add_little_noise(speed, h)
            h += random.uniform(-3, 3) # needs more noise to disperse better
        r = math.radians(h)
        dx = math.sin(r) * speed
        dy = math.cos(r) * speed
        if checker is None or checker(self, dx, dy):
            self.move_by(dx, dy)
            return True
        return False

    def move_by(self, x, y):
        self.x += x
        self.y += y

class Model(interfaces.ParticleFiltering):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, Q, R, world, USE_BEACON):
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)
        self.world = world
        self.USE_BEACON = USE_BEACON

    def create_initial_estimate(self, N):
        # x,y,h

        xy = list(self.world.random_free_place())
        xy.append(random.uniform(0, 360))
        particles = numpy.array([xy],dtype=float)
        for _ in range(1,N):
            xy = list(self.world.random_free_place())
            xy.append(random.uniform(0, 360))
            particles = numpy.concatenate((particles,[xy]))
        return particles


    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        h = u[0]
        speed = u[1]

        heading_noise = numpy.random.normal(0, self.Q[0], (N,)).reshape((N, 1))
        speed_noise= numpy.random.normal(0, self.Q[1], (N,)).reshape((N, 1))

        # headingWithNoiseRadian = numpy.radians(headingWithNoise)
        # xWithNoise = numpy.sin(headingWithNoiseRadian) * speedWithNoise
        # yWithNoise = numpy.cos(headingWithNoiseRadian) * speedWithNoise
        # noise = numpy.concatenate((xWithNoise,yWithNoise,headingWithNoise),axis=1)
        noise = numpy.concatenate((heading_noise,speed_noise),axis=1)
        # noise = numpy.zeros((N, 2))
        # noise[:,0] = heading_noise
        # noise[:,1] = speed_noise
        
        return noise

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """

        N = len(particles)
        noisy_u = noise + u
        # print(noisy_u)

        h = noisy_u[:,0]
        speed = noisy_u[:,1]
        # print(h)
        # print(speed)
        # print(h)
        # print(speed)
        # h = u[0] + noise[:,0]
        # speed = u[1] + noise[:,1]

        h_rad = numpy.radians(h)
        xy = numpy.zeros((N, 2))
        xy[:,0] = numpy.array(numpy.sin(h_rad) * speed)
        xy[:,1] = numpy.array(numpy.cos(h_rad) * speed)

        # print(x)

        # print(np.vstack((x,y),axis=1))
        # print(xy[:,1])
        particles[:,:2] += xy
        particles[:,2] = h


        # for p in particles:
        #     h = u[0]  # in case robot changed heading, swirl particle heading too
        #     speed = u[1]
        #
        #     # while True:
        #     #     noisy = True
        #     #     h = p[2]
        #     #     # if noisy:
        #     #     print(speed,",",d_h)
        #     # speed, d_h = add_little_noise(speed, d_h)
        #     #     print(speed,",",d_h)
        #     #     print("#########")
        #     #         # d_h += random.uniform(-3, 3)  # needs more noise to disperse better
        #     #     r = math.radians(h)
        #     #     # print(speed)
        #     #     dx = math.sin(r) * speed
        #     #     dy = math.cos(r) * speed
        #     #     if self.world.is_free(p[0] + dx, p[1] + dy):
        #     #         p[0] += dx
        #     #         p[1] += dy
        #     #         break
        #     #     # Bumped into something or too long in same direction,
        #     #     # chose random new direction
        #     #     p[2] = random.uniform(0, 360)
        #
        #     r = math.radians(d_h)
        #     p[0] += math.sin(r) * speed
        #     p[1] += math.cos(r) * speed
        #     p[2] += d_h
        #
        #     # print("#################")
        #     # print(p)


    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        r_d = y

        yprob = numpy.empty(len(particles), dtype=float)

        logyprob = numpy.empty(len(particles), dtype=float)

        for k in range(len(particles)):
            p = particles[k]
            if self.world.is_free(p[0],p[1]):
                p_d_val = self.world.distance_to_nearest_beacon(p[0],p[1])
                p_d = numpy.array([p_d_val])
                if self.USE_BEACON :
                    logyprob[k] = kalman.lognormpdf(r_d-p_d, self.R)
                else:
                    logyprob[k] = kalman.lognormpdf(numpy.array([0]), self.R)
            else:
                logyprob[k] = numpy.array([-100])


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


def angle_between(p1, p2):
    x = numpy.array([p2[0]-p1[0]])
    y = numpy.array([p2[1]-p1[1]])
    ang = numpy.arctan2(y,x)
    print(ang)
    return numpy.rad2deg(ang % (2 * numpy.pi))

def guess_robot_heading(result_history):
    sum_angle = 0
    for i in range(1,len(result_history)):
        sum_angle += angle_between(result_history[i],result_history[i-1])
    return sum_angle / (len(result_history)-1)

if __name__ == '__main__':

    steps = 5000000
    num = 1000
    P0 = 1.0
    robot_speed = 0.2
    robot_heading_odometry_noise = 3
    robot_heading_measurement_noise =20
    Q = numpy.asarray((robot_heading_odometry_noise, robot_speed * 0.5)) #heading, speed variances
    R = numpy.asarray(((0.9 ** 2,),))
    result_history_length = 6
    USE_BEACON = True

    # maze_data = ((1, 1, 0, 0, 2, 0, 0, 0, 0, 1),
    #              (1, 2, 0, 0, 1, 1, 0, 0, 0, 0),
    #              (0, 1, 1, 0, 0, 0, 0, 1, 0, 1),
    #              (0, 0, 0, 0, 1, 0, 0, 1, 1, 2),
    #              (1, 1, 0, 1, 1, 2, 0, 0, 1, 0),
    #              (1, 1, 1, 0, 1, 1, 1, 0, 2, 0),
    #              (2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    #              (1, 2, 0, 1, 2, 1, 1, 0, 0, 0),
    #              (0, 0, 0, 0, 1, 0, 0, 0, 2, 0),
    #              (0, 0, 1, 0, 0, 2, 1, 1, 1, 0))
    maze_data = ((1, 1, 0, 0, 2, 0, 0, 0, 0, 1),
                 (1, 2, 0, 0, 1, 0, 0, 0, 0, 0),
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
                 (0, 0, 0, 0, 1, 2, 0, 0, 0, 0),
                 (0, 0, 0, 0, 1, 1, 0, 0, 2, 0),
                 (2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                 (1, 2, 0, 0, 2, 1, 0, 0, 0, 0),
                 (0, 0, 0, 0, 1, 0, 0, 0, 2, 0),
                 (0, 0, 0, 0, 0, 2, 0, 0, 0, 0))

    world = create_world(maze_data)
    # Make realization deterministic
    numpy.random.seed(1)

    model = Model(Q, R, world, USE_BEACON)
    sim = simulator.Simulator(model, u=None, y= numpy.array([0]))

    resamplings = 0
    robot = RealRobot(world, robot_speed)

    sim.pt = filter.ParticleTrajectory(sim.model, num,resample=0.99)

    # result_history = []
    for i in range(steps):
        # Run PF using noise corrupted input signal

        # Odometry
        robot.move()
        u = robot.read_odometry(Q)
        # sensor measurement

        # if len(result_history) == result_history_length :
        #     print(guess_robot_heading(result_history))

        y = numpy.array([robot.read_sensor()])

        # forward filter
        if (sim.pt.forward(u, y)):
            resamplings = resamplings + 1

        meanXY= sim.get_filtered_mean()[-1]
        model.x = meanXY[0]
        model.y = meanXY[1]

        # if len(result_history) < result_history_length:
        #     result_history.append(meanXY)
        # else :
        #     result_history = result_history[1:]
        #     result_history.append(meanXY)

        # ---------- Show current state ----------
        model.world.show_mean(meanXY[0],meanXY[1])
        model.world.show_robot(robot)
        model.world.show_particles_2(sim.pt.traj[-1].pa.part)

