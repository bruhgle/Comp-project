#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    body("earth", 1.988e30, [1.49598e11, 0], 0, 0), 
    body("moon", 7.35e22, [1.49598e11, 0], 0, 1.49598e11),
    body("rocket", 6.4169e23, [2.27956e11, 0], 59355072, 2.27956e11)
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

earth_list = []
moon_list = []
mars_list = []
jupiter_list = []
asteroid_list = []
time_list = []

fig, ax = plt.subplots()
ax.set_xlim(-1e12, 1e12)
ax.set_ylim(-1e12, 1e12)
ax.set_aspect('equal')

planet_size = 4
asteroid_size = 3

# Plot initial positions of celestial bodies
earth_plot, = ax.plot([], [], linestyle='-', color='g', label='Earth', marker='o', markersize = planet_size)
mars_plot, = ax.plot([], [], linestyle='-', color='g', label='Mars', marker='o', markersize = planet_size)
jupiter_plot, = ax.plot([], [], linestyle='-', color='g', label='Jupiter', marker='o', markersize = planet_size)
asteroid_plot, = ax.plot([], [], linestyle='-', color='r', label='Asteroid', marker='o', markersize = asteroid_size)

sun_circle = plt.Circle((0, 0), 1e10, color='yellow', label='Sun')
ax.add_patch(sun_circle)

# Set up the legend
legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}
ax.legend(prop=legend_font, loc='upper right')

velocity = [-3000, 7000]

position = bodies[4].position

def update(frame):

    global velocity, position

    bodies[1].position = planet_position(1, frame * step_size)
    bodies[2].position = planet_position(2, frame * step_size)
    bodies[3].position = planet_position(3, frame * step_size)
 
    sun_acceleration = compute_acceleration(0)
    earth_acceleration = compute_acceleration(1)
    mars_acceleration = compute_acceleration(2)
    jupiter_acceleration = compute_acceleration(3)

    acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + jupiter_acceleration[0], sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + jupiter_acceleration[1]]

    velocity = leapfrog_velocity(velocity, acceleration, step_size)

    position = leapfrog_position(position, velocity, acceleration, step_size)

    bodies[4].position = position

    x_earth, y_earth = bodies[1].position
    x_mars, y_mars = bodies[2].position
    x_jupiter, y_jupiter = bodies[3].position
    x_asteroid, y_asteroid = bodies[4].position

    earth_plot.set_data(x_earth, y_earth)
    mars_plot.set_data(x_mars, y_mars)
    jupiter_plot.set_data(x_jupiter, y_jupiter)
    asteroid_plot.set_data(x_asteroid, y_asteroid)

    sun_circle.set_center((0, 0))

    return earth_plot, mars_plot, jupiter_plot, asteroid_plot, sun_circle

num_frames = int(2e8 / 1000)

step_size = 100000

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)

# Display the animation
plt.show()