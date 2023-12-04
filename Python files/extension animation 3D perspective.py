#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    body("sun", 1.988e30, [0, 0, 0], 0, 0), 
    body("earth", 7.35e22, [1.49598e11, 0, 0], 31558118, 1.49598e11),
    body("mars", 6.4169e23, [2.27956e11, 0, 0], 59355072, 2.27956e11),
    body("jupiter", 1.898e27, [7,78479e11, 0, 0], 374335689, 7.78479e11),
    body("asteroid", 1000, [1.49598e11, 0, 1e12], 0, 0)
]

def planet_position(index, t):

    x = bodies[index].radius * math.cos(2 * np.pi * t / bodies[index].period)
    y = bodies[index].radius * math.sin(2 * np.pi * t / bodies[index].period)
    z = 0

    return x, y, z

def compute_displacement(index):

    r_body = [bodies[index].position[0] - bodies[4].position[0], bodies[index].position[1] - bodies[4].position[1], bodies[index].position[0] - bodies[4].position[2]]

    return r_body

def new_compute_force(index):

    displacement = compute_displacement(index)

    force = [G * bodies[index].mass * bodies[4].mass * (displacement[0]/((np.linalg.norm(displacement)**3))), G * bodies[index].mass * bodies[4].mass * (displacement[1]/((np.linalg.norm(displacement)**3))), G * bodies[index].mass * bodies[4].mass * (displacement[2]/((np.linalg.norm(displacement)**3)))]

    return force

def compute_acceleration(index):

    force = new_compute_force(index)

    acceleration = [force[0] / bodies[4].mass, force[1] / bodies[4].mass, force[2] / bodies[4].mass]

    return acceleration

def leapfrog_position(position, velocity, acceleration, step):

    new_velocity_x = velocity[0] + 0.5 * (step * acceleration[0])
    new_velocity_y = velocity[1] + 0.5 * (step * acceleration[1])
    new_velocity_z = velocity[2] + 0.5 * (step * acceleration[2])

    new_position_x = position[0] + (step * new_velocity_x)
    new_position_y = position[1] + (step * new_velocity_y)
    new_position_z = position[2] + (step * new_velocity_z)

    return new_position_x, new_position_y, new_position_z

def leapfrog_velocity(velocity, acceleration, step):

    new_velocity_x = velocity[0] + (step * acceleration[0])
    new_velocity_y = velocity[1] + (step * acceleration[1])
    new_velocity_z = velocity[2] + (step * acceleration[2])

    return new_velocity_x, new_velocity_y, new_velocity_z

fig, (ax_above, ax_side) = plt.subplots(1, 2, figsize=(12, 6))

# Set up the initial view limits
ax_above.set_xlim([-1.5e12, 1.5e12])
ax_above.set_ylim([-1.5e12, 1.5e12])

ax_side.set_xlim([-1.5e12, 1.5e12])
ax_side.set_ylim([-1.5e12, 1.5e12])

earth_size = 5
mars_size = 5
jupiter_size = 7
asteroid_size = 3
sun_radius = 5e10

earth_above, = ax_above.plot([], [], linestyle='-', color='green', label='Earth', marker = 'o',markersize=earth_size)
mars_above, = ax_above.plot([], [], linestyle='-', color='green', label='Mars', marker = 'o',markersize=mars_size)
jupiter_above, = ax_above.plot([], [], linestyle='-', color='green', label='Jupiter', marker = 'o',markersize=jupiter_size)
asteroid_above, = ax_above.plot([], [], linestyle='-', color='red', label='Asteroid', marker = 'o',markersize=asteroid_size)

earth_side, = ax_side.plot([], [], linestyle='-', color='green', label='Earth', marker = 'o',markersize=earth_size)
mars_side, = ax_side.plot([], [], linestyle='-', color='green', label='Mars', marker = 'o',markersize=mars_size)
jupiter_side, = ax_side.plot([], [], linestyle='-', color='green', label='Jupiter', marker = 'o',markersize=jupiter_size)
asteroid_side, = ax_side.plot([], [], linestyle='-', color='red', label='Asteroid', marker = 'o',markersize=asteroid_size)

sun_circle_above = plt.Circle((0, 0), sun_radius, color='orange', label='Sun')
ax_above.add_patch(sun_circle_above)

sun_circle_side = plt.Circle((0, 0), sun_radius, color='orange', label='Sun')
ax_side.add_patch(sun_circle_side)

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}
ax_above.legend(prop=legend_font, loc='upper right')
ax_side.legend(prop=legend_font, loc='upper right')

velocity = [-3000, 7000, 0]

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

    acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + jupiter_acceleration[0],
                    sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + jupiter_acceleration[1],
                    sun_acceleration[2] + earth_acceleration[2] + mars_acceleration[2] + jupiter_acceleration[2]]

    velocity = leapfrog_velocity(velocity, acceleration, step_size)

    position = leapfrog_position(position, velocity, acceleration, step_size)

    bodies[4].position = position

    x_earth, y_earth, z_earth = bodies[1].position
    x_mars, y_mars, z_mars = bodies[2].position
    x_jupiter, y_jupiter, z_jupiter = bodies[3].position
    x_asteroid, y_asteroid, z_asteroid = bodies[4].position

    earth_above.set_data(x_earth, y_earth)
    mars_above.set_data(x_mars, y_mars)
    jupiter_above.set_data(x_jupiter, y_jupiter)
    asteroid_above.set_data(x_asteroid, y_asteroid)

    earth_side.set_data(x_earth, z_earth)
    mars_side.set_data(x_mars, z_mars)
    jupiter_side.set_data(x_jupiter, z_jupiter)
    asteroid_side.set_data(x_asteroid, z_asteroid)

    sun_circle_above.set_center((0, 0))
    sun_circle_side.set_center((0, 0))

    return earth_above, mars_above, jupiter_above, asteroid_above, earth_side, mars_side, jupiter_side, asteroid_side, sun_circle_above, sun_circle_side

step_size = 100000

num_frames = int(2e8 / step_size)

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)

animation_var = animation

plt.show()