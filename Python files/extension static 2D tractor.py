#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#define global variables

G = 6.6726e-11
pi = np.pi

class body:
    def __init__(self, name, mass, position, period, radius, initial_angle=0):
        self.name = name
        self.position = position
        self.mass = mass
        self.period = period
        self.radius = radius
        self.initial_angle = initial_angle
        self.past_positions = []
        self.clearance = []

bodies = [
    body("sun", 1.988e30, [0, 0], 0, 0), 
    body("earth", 7.35e22, [1.49598e11, 0], 31558118, 1.49598e11, initial_angle=0.566332),
    body("mars", 6.4169e23, [2.27956e11, 0], 59355072, 2.27956e11, initial_angle=3.7001),
    body("venus", 4.86732e24, [1.08210e11, 0], 19414166, 1.08210e11, initial_angle=5.7857665),
    body("asteroid1", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid2", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid3", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid4", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid5", 26.99e9, [1.64e11, 0], 0, 0),
    body("tractor1", 1e6, [1.64e11, 0], 0, 0),
    body("tractor2", 1e6, [1.64e11, 0], 0, 0),
    body("tractor3", 1e6, [1.64e11, 0], 0, 0),
    body("tractor4", 1e6, [1.64e11, 0], 0, 0),
    body("tractor5", 1e6, [1.64e11, 0], 0, 0),
]

def planet_position(index, t):

    initial_angle = bodies[index].initial_angle

    x = bodies[index].radius * math.cos(initial_angle + 2 * np.pi * t / bodies[index].period)
    y = bodies[index].radius * math.sin(initial_angle + 2 * np.pi * t / bodies[index].period)

    return x, y

def compute_displacement(index, asteroid_index):

    r_body = [bodies[index].position[0] - bodies[asteroid_index].position[0],
            bodies[index].position[1] - bodies[asteroid_index].position[1]]

    return r_body

def new_compute_force(index, asteroid_index):

    displacement = compute_displacement(index, asteroid_index)

    force = [G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[0]/((np.linalg.norm(displacement)**3))),
            G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[1]/((np.linalg.norm(displacement)**3)))]

    return force

def compute_acceleration(index, asteroid_index):

    force = new_compute_force(index, asteroid_index)

    acceleration = [force[0] / bodies[asteroid_index].mass, force[1] / bodies[asteroid_index].mass]

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
venus_list = []
time_list = []

plt.figure(figsize=(6,6))

def time_step(time, step_size, asteroid_index, mass, tractor_radius):

    bodies[asteroid_index+5].mass = mass

    clearance_list = []

    initial_velocity = [0, 25547]

    velocity = initial_velocity

    position = bodies[asteroid_index].position

    num_steps = int(time / step_size)

    asteroid_past_positions = [[] for _ in range(len(bodies) - 1)]

    bodies[1].position = planet_position(1, 0)
    bodies[2].position = planet_position(2, 0)
    bodies[3].position = planet_position(3, 0)

    for i in range(0, num_steps):

        bodies[1].position = planet_position(1, i * step_size)
        bodies[2].position = planet_position(2, i * step_size)
        bodies[3].position = planet_position(3, i * step_size)

        sun_distance = np.linalg.norm(compute_displacement(0, asteroid_index))
        reduction_factor = 1 - (tractor_radius/sun_distance)

        bodies[asteroid_index+5].position = [bodies[asteroid_index].position[0] * reduction_factor, 
                                            bodies[asteroid_index].position[1] * reduction_factor]

        sun_acceleration = compute_acceleration(0, asteroid_index)
        earth_acceleration = compute_acceleration(1, asteroid_index)
        mars_acceleration = compute_acceleration(2, asteroid_index)
        venus_acceleration = compute_acceleration(3, asteroid_index)
        tractor_acceleration = compute_acceleration(asteroid_index+5, asteroid_index)

        acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + venus_acceleration[0] + tractor_acceleration[0],
                        sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + venus_acceleration[1] + tractor_acceleration[1]]

        velocity = leapfrog_velocity(velocity, acceleration, step_size)

        position = leapfrog_position(position, velocity, acceleration, step_size)

        bodies[asteroid_index].position = position

        for k in range(1, 100):

            if i == int(k * num_steps/100):

                print("Asteroid number",asteroid_index-3, k, "percent done")

        earth_displacement = compute_displacement(1, asteroid_index)

        earth_list.append(bodies[1].position)
        mars_list.append(bodies[2].position)
        venus_list.append(bodies[3].position)
        clearance_list.append(np.linalg.norm(earth_displacement))

        asteroid_past_positions[asteroid_index - 1].append(bodies[asteroid_index].position)

        if asteroid_index == 4:

            time_list.append(i * step_size)
    
    bodies[asteroid_index].past_positions = asteroid_past_positions[asteroid_index - 1]
    bodies[asteroid_index].clearance = clearance_list

sim_time = 3.33e7
step = 10000

time_step(sim_time, step, 4, 0, 1000)
time_step(sim_time, step, 5, 2e7, 1000)
time_step(sim_time, step, 6, 3e7, 1000)
time_step(sim_time, step, 7, 4e7, 1000)
time_step(sim_time, step, 8, 5e7, 1000)

x_earth = [coord[0] for coord in earth_list]
y_earth = [coord[1] for coord in earth_list]

x_mars = [coord[0] for coord in mars_list]
y_mars = [coord[1] for coord in mars_list]

x_venus = [coord[0] for coord in venus_list]
y_venus = [coord[1] for coord in venus_list]

x_asteroid1 = [coord[0] for coord in bodies[4].past_positions]
y_asteroid1 = [coord[1] for coord in bodies[4].past_positions]

x_asteroid2 = [coord[0] for coord in bodies[5].past_positions]
y_asteroid2 = [coord[1] for coord in bodies[5].past_positions]

x_asteroid3 = [coord[0] for coord in bodies[6].past_positions]
y_asteroid3 = [coord[1] for coord in bodies[6].past_positions]

x_asteroid4 = [coord[0] for coord in bodies[7].past_positions]
y_asteroid4 = [coord[1] for coord in bodies[7].past_positions]

x_asteroid5 = [coord[0] for coord in bodies[8].past_positions]
y_asteroid5 = [coord[1] for coord in bodies[8].past_positions]

days = [t / 86400 for t in time_list]

plt.scatter(0, 0, color='yellow', marker='o', s=500, edgecolor='black', zorder=2)

circle = plt.Circle((0,0), bodies[0].radius, edgecolor='orange', facecolor='none', linestyle='dashed', linewidth=0.8)
plt.gca().add_patch(circle)
circle = plt.Circle((0,0), bodies[1].radius, edgecolor='green', facecolor='none', linestyle='dashed', linewidth=0.8)
plt.gca().add_patch(circle)
circle = plt.Circle((0,0), bodies[2].radius, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=0.8)
plt.gca().add_patch(circle)
circle = plt.Circle((0,0), bodies[3].radius, edgecolor='orange', facecolor='none', linestyle='dashed', linewidth=0.8)
plt.gca().add_patch(circle)

plt.scatter(bodies[1].position[0], bodies[1].position[1], marker='o', color='green', s=50)
plt.scatter(bodies[2].position[0], bodies[2].position[1], marker='o', color='red', s=50)
plt.scatter(bodies[3].position[0], bodies[3].position[1], marker='o', color='orange', s=50)

plt.scatter(bodies[4].position[0], bodies[4].position[1], marker='o', color='navy', s=20)
plt.scatter(bodies[5].position[0], bodies[5].position[1], marker='o', color='blue', s=20)
plt.scatter(bodies[6].position[0], bodies[6].position[1], marker='o', color='darkviolet', s=20)
plt.scatter(bodies[7].position[0], bodies[7].position[1], marker='o', color='fuchsia', s=20)
plt.scatter(bodies[8].position[0], bodies[8].position[1], marker='o', color='crimson', s=20)

plt.plot(x_asteroid1, y_asteroid1, linestyle='-', color='navy', label='Asteroid 1', marker='')
plt.plot(x_asteroid2, y_asteroid2, linestyle='-', color='blue', label='Asteroid 2', marker='')
plt.plot(x_asteroid3, y_asteroid3, linestyle='-', color='darkviolet', label='Asteroid 3', marker='')
plt.plot(x_asteroid4, y_asteroid4, linestyle='-', color='fuchsia', label='Asteroid 4', marker='')
plt.plot(x_asteroid5, y_asteroid5, linestyle='-', color='crimson', label='Asteroid 5', marker='')

plt.title("", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.xlabel("x position (m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("y position (m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

plt.legend(prop = legend_font, loc = 'upper right')
plt.axis('equal')
#plt.xlim(-2.5e11, 2.5e11)
#plt.ylim(-2.5e11, 2.5e11)
plt.show()

plt.plot(days, bodies[4].clearance, linestyle='-', color='navy', label='Asteroid 1 clearance', marker='')
plt.plot(days, bodies[5].clearance, linestyle='-', color='blue', label='Asteroid 2 clearance', marker='')
plt.plot(days, bodies[6].clearance, linestyle='-', color='darkviolet', label='Asteroid 3 clearance', marker='')
plt.plot(days, bodies[7].clearance, linestyle='-', color='fuchsia', label='Asteroid 4 clearance', marker='')
plt.plot(days, bodies[8].clearance, linestyle='-', color='crimson', label='Asteroid 5 clearance', marker='')

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("Earth clearance (m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.legend(prop = legend_font, loc = 'upper center')
plt.show()