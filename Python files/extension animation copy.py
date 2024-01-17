import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# Define global variables
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
        self.separation = []

bodies = [
    body("sun", 1.988e30, [0, 0], 0, 0), 
    body("earth", 7.35e22, [1.49598e11, 0], 31558118, 1.49598e11, initial_angle=4.55900034), 
    body("mars", 6.4169e23, [2.27956e11, 0], 59355072, 2.27956e11, initial_angle=2.181659), 
    body("venus", 4.86732e24, [1.08210e11, 0], 19414166, 1.08210e11, initial_angle=0.968658),
    body("asteroid1", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid2", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid3", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid4", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid5", 26.99e9, [1.64e11, 0], 0, 0),
]

def planet_position(index, t):
    x = bodies[index].radius * math.cos(bodies[index].initial_angle + 2 * np.pi * t / bodies[index].period)
    y = bodies[index].radius * math.sin(bodies[index].initial_angle + 2 * np.pi * t / bodies[index].period)
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

earth_list = []
mars_list = []
venus_list = []
time_list = []

fig, ax = plt.subplots()
ax.set_xlim(-1.5e11, 1.5e11)
ax.set_ylim(-1.5e11, 1.5e11)
ax.set_aspect('equal')

planet_size = 4
asteroid_size = 3

# Plot initial positions of celestial bodies
earth_plot, = ax.plot([], [], linestyle='-', color='g', label='Earth', marker='o', markersize=planet_size)
mars_plot, = ax.plot([], [], linestyle='-', color='g', label='Mars', marker='o', markersize=planet_size)
venus_plot, = ax.plot([], [], linestyle='-', color='g', label='Venus', marker='o', markersize=planet_size)
asteroid_plot, = ax.plot([], [], linestyle='-', color='r', label='Asteroid', marker='o', markersize=asteroid_size)

sun_circle = plt.Circle((0, 0), 1e9, color='yellow', label='Sun')
ax.add_patch(sun_circle)

# Set up the legend
legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}
ax.legend(prop=legend_font, loc='upper right')

velocity = [0, 25547]
position = bodies[4].position

def update(frame):
    plt.clf()  # Clear the previous frame

    for i in range(1, 4):  # Update positions for Earth, Mars, and Venus
        bodies[i].position = planet_position(i, frame * step_size)

    sun_acceleration = compute_acceleration(0, 4)
    earth_acceleration = compute_acceleration(1, 4)
    mars_acceleration = compute_acceleration(2, 4)
    venus_acceleration = compute_acceleration(3, 4)

    acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + venus_acceleration[0],
                    sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + venus_acceleration[1]]

    velocity = leapfrog_velocity(velocity, acceleration, step_size)
    position = leapfrog_position(position, velocity, acceleration, step_size)
    bodies[4].position = position

    # Update position for Earth, Mars, Venus, and the asteroid
    x_earth, y_earth = bodies[1].position
    x_mars, y_mars = bodies[2].position
    x_venus, y_venus = bodies[3].position
    x_asteroid, y_asteroid = bodies[4].position

    earth_plot, = plt.plot(x_earth, y_earth, linestyle='-', color='g', label='Earth', marker='o', markersize=planet_size)
    mars_plot, = plt.plot(x_mars, y_mars, linestyle='-', color='g', label='Mars', marker='o', markersize=planet_size)
    venus_plot, = plt.plot(x_venus, y_venus, linestyle='-', color='g', label='Venus', marker='o', markersize=planet_size)
    asteroid_plot, = plt.plot(x_asteroid, y_asteroid, linestyle='-', color='r', label='Asteroid', marker='o', markersize=asteroid_size)

    # Plot Sun
    plt.scatter(0, 0, color='yellow', marker='o', s=10, edgecolor='black', zorder=2)
    sun_circle = plt.Circle((0, 0), 1e9, color='yellow', label='Sun')
    plt.gca().add_patch(sun_circle)

    plt.title("Celestial Bodies Animation", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    plt.xlabel("x position (m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    plt.ylabel("y position (m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    plt.axis('equal')

    return earth_plot, mars_plot, venus_plot, asteroid_plot, sun_circle

num_frames = int(2e8 / 1000)
step_size = 100000

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=1, blit=False)

# Display the animation
plt.show()