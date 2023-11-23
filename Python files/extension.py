#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt

#define global variables

G = 6.6726e-11
pi = np.pi

class body:
    def __init__(self, name, mass, position, period, radius):
        self.name = name
        self.position = position
        self.mass = mass
        self.period = period
        self.radius = radius

bodies = [
    body("sun", 1.988e30, [0, 0], 0, 0), 
    body("earth", 7.35e22, [1.49598e11, 0], 31558118, 1.49598e11),
    body("mars", 6.4169e23, [2.27956e11, 0], 59355072, 2.27956e11),
    body("jupiter", 1.898e27, [7,78479e11, 0], 374335689, 7.78479e11),
    body("asteroid", 1000, [0.8e12, 0], 0, 0)
]

def planet_position(index, t):

    x = bodies[index].radius * math.cos(2 * np.pi * t / bodies[index].period)
    y = bodies[index].radius * math.sin(2 * np.pi * t / bodies[index].period)

    return x, y

def compute_displacement(index):

    r_body = [bodies[index].position[0] - bodies[4].position[0], bodies[index].position[1] - bodies[4].position[1]]

    return r_body

def new_compute_force(index):

    displacement = compute_displacement(index)

    force = [G * bodies[index].mass * bodies[4].mass * (displacement[0]/((np.linalg.norm(displacement)**3))), G * bodies[index].mass * bodies[4].mass * (displacement[1]/((np.linalg.norm(displacement)**3)))]

    return force

def compute_acceleration(index):

    force = new_compute_force(index)

    acceleration = [force[0] / bodies[4].mass, force[1] / bodies[4].mass]

    return acceleration

def leapfrog_position(position, velocity, acceleration, step):

    new_velocity_x = velocity[0] + 0.5 * (step * acceleration[0])
    new_velocity_y = velocity[1] + 0.5 * (step * acceleration[1])

    new_position_x = position[0] + (step * new_velocity_x)
    new_position_y = position[1] + (step * new_velocity_y)

    return new_position_x, new_position_y

def leapfrog_velocity(velocity, acceleration, step):

    new_velocity_x = velocity[0] + (step * acceleration[0])
    new_velocity_y = velocity[1] + (step * acceleration[1])

    return new_velocity_x, new_velocity_y

#create empty coordinate lists

earth_list = []
moon_list = []
mars_list = []
jupiter_list = []
asteroid_list = []
time_list = []

def time_step(time, step_size):

    initial_velocity = [0, 7000]

    velocity = initial_velocity

    position = bodies[4].position

    num_steps = int(time / step_size)

    for i in range(0, num_steps):

        bodies[1].position = planet_position(1, i * step_size)
        bodies[2].position = planet_position(2, i * step_size)
        bodies[3].position = planet_position(3, i * step_size)

        sun_acceleration = compute_acceleration(0)
        earth_acceleration = compute_acceleration(1)
        mars_acceleration = compute_acceleration(2)
        jupiter_acceleration = compute_acceleration(3)

        acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + jupiter_acceleration[0], sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + jupiter_acceleration[1]]

        velocity = leapfrog_velocity(velocity, acceleration, step_size)

        position = leapfrog_position(position, velocity, acceleration, step_size)

        bodies[4].position = position

        for k in range(1, 100):

            if i == int(k * num_steps/100):

                print(k, np.linalg.norm(velocity))

        earth_list.append([bodies[1].position[0], bodies[1].position[1]])
        mars_list.append([bodies[2].position[0], bodies[2].position[1]])
        jupiter_list.append([bodies[3].position[0], bodies[3].position[1]])
        asteroid_list.append([bodies[4].position[0], bodies[4].position[1]])
        time_list.append(i * step_size)

time_step(2e8, 1000)

x_earth = [coord[0] for coord in earth_list]
y_earth = [coord[1] for coord in earth_list]

x_mars = [coord[0] for coord in mars_list]
y_mars = [coord[1] for coord in mars_list]

x_jupiter = [coord[0] for coord in jupiter_list]
y_jupiter = [coord[1] for coord in jupiter_list]

x_asteroid = [coord[0] for coord in asteroid_list]
y_asteroid = [coord[1] for coord in asteroid_list]

days = [t / 86400 for t in time_list]

plt.figure(figsize=(6,6))
plt.plot(x_earth, y_earth, linestyle = '-', color = 'g', label = 'Earth', marker = '')
plt.plot(x_mars, y_mars, linestyle = '-', color = 'g', label = 'Mars', marker = '')
plt.plot(x_jupiter, y_jupiter, linestyle = '-', color = 'g', label = 'Jupiter', marker = '')
plt.plot(x_asteroid, y_asteroid, linestyle = '-', color = 'r', label = 'Asteroid', marker = '')
plt.xlabel("x position (m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("y position (m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

plt.legend(prop = legend_font, loc = 'upper right')
plt.axis('equal')
plt.xlim(-1e12, 1e12)
plt.ylim(-1e12, 1e12)
plt.show()