#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
import ephem
from matplotlib.patches import Ellipse

#define global variables

G = 6.6726e-11
pi = np.pi

start_date_str = "2024-09-24 00:00:00"
start_pos = [-1.590269546178139E+011,   3.858859839484540E+10,  -5.805981106895613E+09]
start_vel = [-4.677768707144813E+03,  -2.537175688165507E+04,   1.242563246429153E+03]
start_pos_sigma = [3.85854528E+02,          8.20628908E+02,          3.73558771E+02]
start_vel_sigma = [1.32453254E-04,          4.33736823E-05,          8.80633419E-05]
sim_time = 1.435968e+7
step = 100000
start_time = 0
num_asteroids = 3

start_date_formatted = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")

class body:
    def __init__(self, name, mass, acceleration, velocity, position, radius):
        self.name = name
        self.position = position
        self.acceleration = acceleration
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
        self.past_positions = []
        self.clearance = []
        self.separation = []
        self.trail = {'x': [], 'y': []}  

bodies = [
    body("sun", 1.9884e30, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0),                #0   
    body("Mercury", 3.3010e23, [0, 0, 0], [0, 0, 0], [0, 0, 0], 5.7909e10),    #1
    body("Venus", 4.8673e24, [0, 0, 0], [0, 0, 0], [0, 0, 0], 1.08210e11),     #2
    body("Earth", 5.9722e24, [0, 0, 0], [0, 0, 0], [0, 0, 0], 1.49598e11),     #3           #mass errors found at [https://web.archive.org/web/20161224174302/http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf]
    body("Mars", 6.4169e23, [0, 0, 0], [0, 0, 0], [0, 0, 0], 2.27956e11),      #4
    body("Jupiter", 1.8985e27, [0, 0, 0], [0, 0, 0], [0, 0, 0], 7.78479e11),   #5
    body("Saturn", 5.6846e26, [0, 0, 0], [0, 0, 0], [0, 0, 0], 1.432041e12),   #6
    body("Uranus", 8.6813e25, [0, 0, 0], [0, 0, 0], [0, 0, 0], 2.867043e12),   #7
    body("Neptune", 1.0243e26, [0, 0, 0], [0, 0, 0], [0, 0, 0], 4.513953e12),  #8
]

def random_value(center, sigma):
    
    random_value = np.random.normal(loc=center, scale=sigma)
    return random_value

def compute_solar_distance(index, date):

    if index == 3:

        sun = ephem.Sun()

        # Set the date for the calculation
        sun.compute(date)

        # Get the distance from Earth to the Sun in astronomical units (AU)
        distance_au = sun.earth_distance

        # Convert AU to kilometers (1 AU = 149,597,870.7 km)
        distance_km = distance_au * 149597870.7

        return distance_km

    else:

        planet_name = bodies[index].name

        observer = ephem.Observer()
        observer.lat = '0'  
        observer.lon = '0'
    
        planet = getattr(ephem, planet_name)()

        # Set the date for the calculation
        observer.date = date
        planet.compute(observer)

        # Get the distance from the planet to the Sun in astronomical units (AU)
        distance_au = planet.sun_distance

        # Convert AU to kilometers (1 AU = 149,597,870.7 km)
        distance_km = distance_au * 149597870.7

    return distance_km

def planet_angle(index, startdate: datetime, t:int):

    global datetime_date

    datetime_date = startdate + timedelta(seconds = t)

    date_string = datetime_date.strftime('%Y-%m-%d %H:%M:%S')

    date = ephem.Date(date_string)

    if index == 1:

        mercury = ephem.Mercury(date)

        longitude = mercury.hlon
        latitude = mercury.hlat

    if index == 2:

        venus = ephem.Venus(date)

        longitude = venus.hlong
        latitude = venus.hlat

    if index == 3:

        sun = ephem.Sun(date)

        longitude = sun.hlong
        latitude = sun.hlat

    if index == 4:

        mars = ephem.Mars(date)

        longitude = mars.hlong
        latitude = mars.hlat

    if index == 5:

        jupiter = ephem.Jupiter(date)

        longitude = jupiter.hlong
        latitude = jupiter.hlat

    if index == 6:

        saturn = ephem.Saturn(date)

        longitude = saturn.hlong
        latitude = saturn.hlat

    if index == 7:

        uranus = ephem.Uranus(date)

        longitude = uranus.hlong
        latitude = uranus.hlat

    if index == 8:

        neptune = ephem.Neptune(date)

        longitude = neptune.hlong
        latitude = neptune.hlat

    longitude_deg = math.degrees(longitude)
    longitude_rad = math.radians(longitude_deg)
    
    latitude_deg = math.degrees(latitude)
    latitude_rad = math.radians(latitude_deg)

    return longitude_rad, latitude_rad

def planet_position(index, startdate, t):

    datetime_date = startdate + timedelta(seconds = t)

    date_string = datetime_date.strftime('%Y-%m-%d %H:%M:%S')

    date = ephem.Date(date_string)

    angles = planet_angle(index, startdate, t)

    long = angles[0]
    lat = angles[1]

    solar_distance = compute_solar_distance(index, date) * 1000
    solar_radius = solar_distance * math.cos(lat)

    x = solar_radius * math.cos(long)
    y = solar_radius * math.sin(long)
    z = solar_distance * math.sin(lat)

    return x, y, z

def compute_displacement(index, asteroid_index):

    r_body = [0,0,0]

    r_body[0] = bodies[index].position[0] - bodies[asteroid_index].position[0]
    r_body[1] = bodies[index].position[1] - bodies[asteroid_index].position[1]
    r_body[2] = bodies[index].position[2] - bodies[asteroid_index].position[2]

    return r_body

def new_compute_force(index, asteroid_index):
    displacement = compute_displacement(index, asteroid_index)
    displacement_norm = np.linalg.norm(displacement)
    
    # Check if displacement_norm is zero
    if displacement_norm == 0:
        # Return zero force if displacement_norm is zero
        return [0, 0, 0]
    else:
        force = [G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[0]/(displacement_norm**3)),
                 G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[1]/(displacement_norm**3)),
                 G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[2]/(displacement_norm**3))]
        return force

def compute_acceleration(index, asteroid_index):

    force = new_compute_force(index, asteroid_index)

    acceleration = [force[0] / bodies[asteroid_index].mass,
                    force[1] / bodies[asteroid_index].mass,
                    force[2] / bodies[asteroid_index].mass]

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

def compute_moid(index):

    moid = min(bodies[index].clearance)

    return moid

for i in range(1, 1 + num_asteroids):
    asteroid_name = f"asteroid{i}"
    randposx = random_value(start_pos[0],start_pos_sigma[0])
    randposy = random_value(start_pos[1],start_pos_sigma[1])
    randposz = random_value(start_pos[2],start_pos_sigma[2])
    randvelx = random_value(start_vel[0],start_vel_sigma[0])
    randvely = random_value(start_vel[1],start_vel_sigma[1])
    randvelz = random_value(start_vel[2],start_vel_sigma[2])
    randmass = 4.0e+10
    asteroid = body(asteroid_name, randmass, [0,0,0], [randvelx, randvely, randvelz], [randposx, randposy, randposz], 0)
    bodies.append(asteroid)

for i in range(9, 9+num_asteroids):
    bodies[i].trail = {'x': [], 'y': []}

def calculate_average(entries):
    averages = []
    for i in range(0, len(entries), num_asteroids):
        sublist = entries[i:i+num_asteroids]
        avg = sum(sublist) / len(sublist)
        averages.append(avg)
    return averages

def sum_vectors(vectors_list):
    # Initialize variables to store the sum of components
    sum_x = 0
    sum_y = 0
    sum_z = 0
    
    # Loop through each vector in the list
    for vector in vectors_list:
        # Add the x, y, and z components of each vector to the sum variables
        sum_x += vector[0]
        sum_y += vector[1]
        sum_z += vector[2]
    
    # Return the sum of components as a new vector
    return [sum_x, sum_y, sum_z]

earth_list = []
moon_list = []
mars_list = []
venus_list = []
mercury_list = []
jupiter_list = []
saturn_list = []
neptune_list = []
uranus_list = []

time_list = []
impulse_coords = []

acceleration_list = []

earth_final_pos = [0,0,0]
venus_final_pos = [0,0,0]
mars_final_pos = [0,0,0]

plot_size = 0.3e12

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-plot_size, plot_size)
ax.set_ylim(-plot_size, plot_size)
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
planet_plots = []

for _ in range(7):

    plot, = ax.plot([], [], linestyle='-', color='blue', label='planet', marker='o', markersize = planet_size)
    planet_plots.append(plot)

asteroid_plots = []

for _ in range(num_asteroids):
    plot, = ax.plot([], [], linestyle='-', color="red", label="asteroid", marker='o', markersize=asteroid_size)
    asteroid_plots.append(plot)
    
#ax.scatter(96697280301.42848, 114145510583.23306, color='red', marker='x')

orbit_circles = []

for i in range(1,9):

    circle = plt.Circle((0, 0), bodies[i].radius, color='grey', fill=False, linestyle='dashed', linewidth=1)
    orbit_circles.append(circle)
    ax.add_patch(circle)

trail_length = 2000  # Adjust the trail length as needed

trails = []

for _ in range(num_asteroids):

    plot, = ax.plot([], [], linewidth=1, linestyle='-', color='red')
    trails.append(plot)

sun_circle = plt.Circle((0, 0), 1e10, facecolor='yellow', edgecolor = 'black', label='Sun')
ax.add_patch(sun_circle)

# Set up the legend
legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}
label_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 13}

#legend_elements = [
        #Line2D([0], [0], linewidth=2, linestyle='-', color='maroon', markersize=10, label='Undiverted asteroid'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='red', markersize=10, label='8km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='darkorange', markersize=10, label='16km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='greenyellow', markersize=10, label='24km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='green', markersize=10, label='32km/s diversion'),
    #]

#ax.legend(handles = legend_elements, prop=legend_font, loc='upper left', bbox_to_anchor=(1,1))

ax.set_xlabel(fontdict=label_font, xlabel = "x position (AU)")
ax.set_ylabel(fontdict=label_font, ylabel = "y position (AU)")

def update(frame):

    global start_pos, start_vel, start_pos_sigma, start_vel_sigma

    for i in range(1,9):

        bodies[i].position = planet_position(i, start_date_formatted, frame * step)

    for i in range(9,9+num_asteroids):

        sun_acc = compute_acceleration(0, i) 
        mer_acc = compute_acceleration(1, i)
        ven_acc = compute_acceleration(2, i)
        ear_acc = compute_acceleration(3, i)
        mar_acc = compute_acceleration(4, i)
        jup_acc = compute_acceleration(5, i)
        sat_acc = compute_acceleration(6, i)
        ura_acc = compute_acceleration(7, i)
        nep_acc = compute_acceleration(8, i)

        acc = [0,0,0]

        acc[0] = sun_acc[0]+mer_acc[0]+ven_acc[0]+ear_acc[0]+mar_acc[0]+jup_acc[0]+sat_acc[0]+ura_acc[0]+nep_acc[0]
        acc[1] = sun_acc[1]+mer_acc[1]+ven_acc[1]+ear_acc[1]+mar_acc[1]+jup_acc[1]+sat_acc[1]+ura_acc[1]+nep_acc[1]
        acc[2] = sun_acc[2]+mer_acc[2]+ven_acc[2]+ear_acc[2]+mar_acc[2]+jup_acc[2]+sat_acc[0]+ura_acc[2]+nep_acc[2]

        bodies[i].acceleration = acc

    for i in range(9,9+num_asteroids):

        bodies[i].velocity = leapfrog_velocity(bodies[i].velocity, bodies[i].acceleration, step)

    if frame == impulse_frame:
        # Apply the velocity change to asteroid 1
        bodies[i].velocity = [bodies[i].velocity[0] + impulse1[0], bodies[i].velocity[1] + impulse1[1], bodies[i].velocity[2] + impulse1[2]]
    
    for i in range(9,9+num_asteroids):

        bodies[i].position = leapfrog_position(bodies[i].position, bodies[i].velocity, bodies[i].acceleration, step)

    positions = []

    for i in range(num_asteroids):

        positions.append(bodies[i].position)

    for i, pos in enumerate(positions):
        bodies[i + 9].trail['x'].append(pos[0])
        bodies[i + 9].trail['y'].append(pos[1])

    #trails = []
    
    #for i in range(9, 9+num_asteroids):
        #trail = bodies[i].trail
        #x_data = trail['x'][-trail_length:]
        #asteroid_trail = trails[i-9].set_data(x_data, y_data)
        #y_data = trail['y'][-trail_length:]
        #trails.append(asteroid_trail)

    planet_plotsi = []

    for i in range(1, 9):  # Assuming Mercury is bodies[1] and Earth is bodies[2]
        planet_plot = bodies[i].position
        x_data = planet_plot[0]
        y_data = planet_plot[1]
        plot, = ax.plot(x_data, y_data, linestyle='-', color='blue', label='planet', marker = 'o', markersize=planet_size)  # corrected line
        planet_plotsi.append(plot)

    ast_plots = []

    for i in range(9, 9+num_asteroids):  # Assuming Mercury is bodies[1] and Earth is bodies[2]
        ast_plot = bodies[i].position
        x_data = ast_plot[0]
        y_data = ast_plot[1]
        plot, = ax.plot(x_data, y_data, linestyle='-', color='red', label='planet', marker = 'o', markersize=asteroid_size)  # corrected line
        ast_plots.append(plot)

    sun_circle.set_center((0, 0))

    current_date = start_date + timedelta(seconds=frame * step)
    date_text.set_text(f"Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")

    return planet_plotsi, ast_plots, date_text

num_frames = int(sim_time / step)

impulse1 = [0, 0, 0]

impulse_time = 1.4e7
impulse_frame = int(impulse_time / step)

start_date = datetime(2023, 11, 15, 0, 0, 0)

# Add date counter text in the top-left corner
date_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black', ha='left')
date_text.set_text(f"Date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=1, blit=False)

# Display the animation (optional)
plt.show()