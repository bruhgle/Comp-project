#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

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
        self.trail = {'x': [], 'y': []}  

bodies = [
    body("sun", 1.988e30, [0, 0], 0, 0), 
    body("earth", 7.35e22, [1.49598e11, 0], 31558118, 1.49598e11, initial_angle=4.55900034), #1 orbit before impact position: 0.566332, sep 30 orbit before impact position: 4.560036
    body("mars", 6.4169e23, [2.27956e11, 0], 59355072, 2.27956e11, initial_angle=2.181659), #1 orbit before impact position: 3.7001, sep 30 orbit before impact position: 2.181659
    body("venus", 4.86732e24, [1.08210e11, 0], 19414166, 1.08210e11, initial_angle=0.968658), #1 orbit before impact position: 5.7857665, sep 30 orbit before impact position: 0.968658
    body("asteroid1", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid2", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid3", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid4", 26.99e9, [1.64e11, 0], 0, 0),
    body("asteroid5", 26.99e9, [1.64e11, 0], 0, 0),
]

for i in range(4, 9):
    bodies[i].trail = {'x': [], 'y': []}

def planet_position(index, t):

    initial_angle = bodies[index].initial_angle

    x = bodies[index].radius * math.cos(initial_angle + 2 * np.pi * t / bodies[index].period)
    y = bodies[index].radius * math.sin(initial_angle + 2 * np.pi * t / bodies[index].period)

    return x, y

def compute_displacement(index, asteroid_index):

    r_body = [bodies[index].position[0] - bodies[asteroid_index].position[0], bodies[index].position[1] - bodies[asteroid_index].position[1]]

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
moon_list = []
mars_list = []
venus_list = []
asteroid_list = []
time_list = []

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.3e12, 0.3e12)
ax.set_ylim(-0.3e12, 0.3e12)
ax.set_aspect('equal')

# Convert axis limits to astronomical units for display
au_xlim = np.array(ax.get_xlim()) / 1.496e11
au_ylim = np.array(ax.get_ylim()) / 1.496e11

# Set custom tick labels for astronomical units with smaller ticks
au_xticks = np.arange(au_xlim[0], au_xlim[1], 0.5)
au_yticks = np.arange(au_ylim[0], au_ylim[1], 0.5)

# Convert custom tick labels back to simulation units
sim_xticks = au_xticks * 1.496e11
sim_yticks = au_yticks * 1.496e11

# Set the custom tick labels
ax.set_xticks(sim_xticks)
ax.set_yticks(sim_yticks)
ax.set_xticklabels([f"{tick:.1f}" for tick in au_xticks])
ax.set_yticklabels([f"{tick:.1f}" for tick in au_yticks])

planet_size = 4
asteroid_size = 3

# Plot initial positions of celestial bodies
earth_plot, = ax.plot([], [], linestyle='-', color='blue', label='Earth', marker='o', markersize = planet_size)
mars_plot, = ax.plot([], [], linestyle='-', color='grey', label='Mars', marker='o', markersize = planet_size)
venus_plot, = ax.plot([], [], linestyle='-', color='grey', label='Venus', marker='o', markersize = planet_size)
asteroid1_plot, = ax.plot([], [], linestyle='-', color='maroon', label='Undiverted asteroid', marker='o', markersize = asteroid_size)
asteroid2_plot, = ax.plot([], [], linestyle='-', color='red', label='8km/s diversion', marker='o', markersize = asteroid_size)
asteroid3_plot, = ax.plot([], [], linestyle='-', color='darkorange', label='16km/s diversion', marker='o', markersize = asteroid_size)
asteroid4_plot, = ax.plot([], [], linestyle='-', color='greenyellow', label='24km/s diversion', marker='o', markersize = asteroid_size)
asteroid5_plot, = ax.plot([], [], linestyle='-', color='green', label='32km/s diversion', marker='o', markersize = asteroid_size)

#ax.scatter(96697280301.42848, 114145510583.23306, color='red', marker='x')

earth_orbit_circle = plt.Circle((0, 0), bodies[1].radius, color='blue', fill=False, linestyle='dashed', linewidth=1)
mars_orbit_circle = plt.Circle((0, 0), bodies[2].radius, color='grey', fill=False, linestyle='dashed', linewidth=1)
venus_orbit_circle = plt.Circle((0, 0), bodies[3].radius, color='grey', fill=False, linestyle='dashed', linewidth=1)

# Add the circles to the plot
ax.add_patch(earth_orbit_circle)
ax.add_patch(mars_orbit_circle)
ax.add_patch(venus_orbit_circle)

trail_length = 2000  # Adjust the trail length as needed

asteroid1_trail, = ax.plot([], [], linewidth=1, linestyle='-', color='maroon')
asteroid2_trail, = ax.plot([], [], linewidth=1, linestyle='-', color='red')
asteroid3_trail, = ax.plot([], [], linewidth=1, linestyle='-', color='darkorange')
asteroid4_trail, = ax.plot([], [], linewidth=1, linestyle='-', color='greenyellow')
asteroid5_trail, = ax.plot([], [], linewidth=1, linestyle='-', color='green')

sun_circle = plt.Circle((0, 0), 1e10, facecolor='yellow', edgecolor = 'black', label='Sun')
ax.add_patch(sun_circle)

# Set up the legend
legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}
label_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 13}

legend_elements = [
        Line2D([0], [0], linewidth=2, linestyle='-', color='maroon', markersize=10, label='Undiverted asteroid'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='red', markersize=10, label='8km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='darkorange', markersize=10, label='16km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='greenyellow', markersize=10, label='24km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='green', markersize=10, label='32km/s diversion'),
    ]

ax.legend(handles = legend_elements, prop=legend_font, loc='upper left', bbox_to_anchor=(1,1))

ax.set_xlabel(fontdict=label_font, xlabel = "x position (AU)")
ax.set_ylabel(fontdict=label_font, ylabel = "y position (AU)")

velocity1 = [0, 25547]
velocity2 = [0, 25547]
velocity3 = [0, 25547]
velocity4 = [0, 25547]
velocity5 = [0, 25547]

position1 = bodies[4].position
position2 = bodies[5].position
position3 = bodies[6].position
position4 = bodies[7].position
position5 = bodies[8].position

def update(frame):

    global velocity1, velocity2, velocity3, velocity4, velocity5, position1, position2, position3, position4, position5

    bodies[1].position = planet_position(1, frame * step)
    bodies[2].position = planet_position(2, frame * step)
    bodies[3].position = planet_position(3, frame * step)
 
    sun_acceleration1 = compute_acceleration(0, 4)
    earth_acceleration1 = compute_acceleration(1, 4)
    mars_acceleration1 = compute_acceleration(2, 4)
    venus_acceleration1 = compute_acceleration(3, 4)

    sun_acceleration2 = compute_acceleration(0, 5)
    earth_acceleration2 = compute_acceleration(1, 5)
    mars_acceleration2 = compute_acceleration(2, 5)
    venus_acceleration2 = compute_acceleration(3, 5)

    sun_acceleration3 = compute_acceleration(0, 6)
    earth_acceleration3 = compute_acceleration(1, 6)
    mars_acceleration3 = compute_acceleration(2, 6)
    venus_acceleration3 = compute_acceleration(3, 6)

    sun_acceleration4 = compute_acceleration(0, 7)
    earth_acceleration4 = compute_acceleration(1, 7)
    mars_acceleration4 = compute_acceleration(2, 7)
    venus_acceleration4 = compute_acceleration(3, 7)

    sun_acceleration5 = compute_acceleration(0, 8)
    earth_acceleration5 = compute_acceleration(1, 8)
    mars_acceleration5 = compute_acceleration(2, 8)
    venus_acceleration5 = compute_acceleration(3, 8)

    acceleration1 = [sun_acceleration1[0] + earth_acceleration1[0] + mars_acceleration1[0] + venus_acceleration1[0],
                    sun_acceleration1[1] + earth_acceleration1[1] + mars_acceleration1[1] + venus_acceleration1[1]]

    acceleration2 = [sun_acceleration2[0] + earth_acceleration2[0] + mars_acceleration2[0] + venus_acceleration2[0],
                    sun_acceleration2[1] + earth_acceleration2[1] + mars_acceleration2[1] + venus_acceleration2[1]]

    acceleration3 = [sun_acceleration3[0] + earth_acceleration3[0] + mars_acceleration3[0] + venus_acceleration3[0],
                    sun_acceleration3[1] + earth_acceleration3[1] + mars_acceleration3[1] + venus_acceleration3[1]]
    
    acceleration4 = [sun_acceleration4[0] + earth_acceleration4[0] + mars_acceleration4[0] + venus_acceleration4[0],
                    sun_acceleration4[1] + earth_acceleration4[1] + mars_acceleration4[1] + venus_acceleration4[1]]

    acceleration5 = [sun_acceleration5[0] + earth_acceleration5[0] + mars_acceleration5[0] + venus_acceleration5[0],
                    sun_acceleration5[1] + earth_acceleration5[1] + mars_acceleration5[1] + venus_acceleration5[1]]

    velocity1 = leapfrog_velocity(velocity1, acceleration1, step)
    velocity2 = leapfrog_velocity(velocity2, acceleration2, step)
    velocity3 = leapfrog_velocity(velocity3, acceleration3, step)
    velocity4 = leapfrog_velocity(velocity4, acceleration4, step)
    velocity5 = leapfrog_velocity(velocity5, acceleration5, step)

    if frame == impulse_frame:
        # Apply the velocity change to asteroid 1
        velocity1 = [velocity1[0] + impulse1[0], velocity1[1] + impulse1[1]]
        velocity2 = [velocity2[0] + impulse2[0], velocity2[1] + impulse2[1]]
        velocity3 = [velocity3[0] + impulse3[0], velocity3[1] + impulse3[1]]
        velocity4 = [velocity4[0] + impulse4[0], velocity4[1] + impulse4[1]]
        velocity5 = [velocity5[0] + impulse5[0], velocity5[1] + impulse5[1]]
        
    
    position1 = leapfrog_position(position1, velocity1, acceleration1, step)
    position2 = leapfrog_position(position2, velocity2, acceleration2, step)
    position3 = leapfrog_position(position3, velocity3, acceleration3, step)
    position4 = leapfrog_position(position4, velocity4, acceleration4, step)
    position5 = leapfrog_position(position5, velocity5, acceleration5, step)

    for i, pos in enumerate([position1, position2, position3, position4, position5]):
        bodies[i + 4].trail['x'].append(pos[0])
        bodies[i + 4].trail['y'].append(pos[1])

    bodies[4].position = position1
    bodies[5].position = position2
    bodies[6].position = position3
    bodies[7].position = position4
    bodies[8].position = position5

    asteroid1_trail.set_data(bodies[4].trail['x'][-trail_length:], bodies[4].trail['y'][-trail_length:])
    asteroid2_trail.set_data(bodies[5].trail['x'][-trail_length:], bodies[5].trail['y'][-trail_length:])
    asteroid3_trail.set_data(bodies[6].trail['x'][-trail_length:], bodies[6].trail['y'][-trail_length:])
    asteroid4_trail.set_data(bodies[7].trail['x'][-trail_length:], bodies[7].trail['y'][-trail_length:])
    asteroid5_trail.set_data(bodies[8].trail['x'][-trail_length:], bodies[8].trail['y'][-trail_length:])

    x_earth, y_earth = bodies[1].position
    x_mars, y_mars = bodies[2].position
    x_jupiter, y_jupiter = bodies[3].position
    x_asteroid1, y_asteroid1 = bodies[4].position
    x_asteroid2, y_asteroid2 = bodies[5].position
    x_asteroid3, y_asteroid3 = bodies[6].position
    x_asteroid4, y_asteroid4 = bodies[7].position
    x_asteroid5, y_asteroid5 = bodies[8].position

    earth_plot.set_data(x_earth, y_earth)
    mars_plot.set_data(x_mars, y_mars)
    venus_plot.set_data(x_jupiter, y_jupiter)
    asteroid1_plot.set_data(x_asteroid1, y_asteroid1)
    asteroid2_plot.set_data(x_asteroid2, y_asteroid2)
    asteroid3_plot.set_data(x_asteroid3, y_asteroid3)
    asteroid4_plot.set_data(x_asteroid4, y_asteroid4)
    asteroid5_plot.set_data(x_asteroid5, y_asteroid5)

    sun_circle.set_center((0, 0))

    current_date = start_date + timedelta(seconds=frame * step)
    date_text.set_text(f"Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")

    return earth_plot, mars_plot, venus_plot, asteroid1_plot, asteroid2_plot, asteroid3_plot, asteroid4_plot, asteroid5_plot, sun_circle, asteroid1_trail, asteroid2_trail, asteroid3_trail, asteroid4_trail, asteroid5_trail, date_text

sim_time = 1.708281e8 + 0.8e8 #3.3e7 for one orbit, 1.708281e8 = time between sep30 and impact
step = 1000
num_frames = int(sim_time / step)

impulse1 = [0, 0]
impulse2 = [0, -100]
impulse3 = [0, -200]
impulse4 = [0, -300]
impulse5 = [0, -400]

impulse_time = 1.4e7
impulse_frame = int(impulse_time / step)

start_date = datetime(2023, 11, 15, 0, 0, 0)

# Add date counter text in the top-left corner
date_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black', ha='left')
date_text.set_text(f"Date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")

# ... (previous code)

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=1, blit=False)

# Display the animation (optional)
plt.show()