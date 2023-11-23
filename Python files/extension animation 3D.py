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
    body("asteroid", 1000, [0.8e12, 0, 1e12], 0, 0)
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-1.5e12, 1.5e12])
ax.set_ylim([-1.5e12, 1.5e12])
ax.set_zlim([-1.5e12, 1.5e12])

earth_radius = 1e10
mars_radius = 2e10
jupiter_radius = 3e10
asteroid_radius = 1e9
sun_radius = 1e11

earth = ax.scatter([bodies[1].position[0]], [bodies[1].position[1]], [0], color='green', label='Earth')
mars = ax.scatter([bodies[2].position[0]], [bodies[2].position[1]], [0], color='red', label='Mars')
jupiter = ax.scatter([bodies[3].position[0]], [bodies[3].position[1]], [0], color='orange', label='Jupiter')
asteroid = ax.scatter([bodies[4].position[0]], [bodies[4].position[1]], [bodies[4].position[2]], color='blue', label='Asteroid')

sun_radius = 1e11
sun = ax.scatter([0], [0], [0], color='yellow', label='Sun')

# Set up the legend
legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}
ax.legend(prop=legend_font, loc='upper right')

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

    earth._offsets3d = ([bodies[1].position[0]], [bodies[1].position[1]], [0])
    mars._offsets3d = ([bodies[2].position[0]], [bodies[2].position[1]], [0])
    jupiter._offsets3d = ([bodies[3].position[0]], [bodies[3].position[1]], [0])
    asteroid._offsets3d = ([bodies[4].position[0]], [bodies[4].position[1]], [bodies[4].position[2]])

    return earth, mars, jupiter, asteroid

num_frames = int(2e8 / 1000)

step_size = 100000

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=1, blit=False)

# Display the animation
plt.show()