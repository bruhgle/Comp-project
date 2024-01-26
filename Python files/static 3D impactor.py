#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import ephem
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from astroquery.jplhorizons import Horizons

#define global variables

G = 6.6726e-11
pi = np.pi

obj_id = '101955'

# Query Horizons for orbital elements
horizons = Horizons(obj_id, id_type='smallbody')
elements = horizons.elements()

# Extract orbital elements
a = elements['a'][0]  # semi-major axis in AU
ecc = elements['e'][0]  # eccentricity
inc = elements['incl'][0]  # inclination in degrees
raan = elements['Omega'][0]  # right ascension of ascending node in degrees
argp = elements['w'][0]  # argument of periapsis in degrees
nu = elements['nu'][0]  # true anomaly in degrees

# Convert inclination and argument of periapsis to radians
inc_rad = inc * (180 / 3.14159265359)
argp_rad = argp * (180 / 3.14159265359)

# Create Orbit object
asteroid_orbit = Orbit.from_classical(Sun, a, ecc, inc_rad, raan, argp_rad, nu)

# Get heliocentric ecliptic coordinates
x_ec, y_ec, z_ec = asteroid_orbit.r.eci()

class body:
    def __init__(self, name, mass, position, radius):
        self.name = name
        self.position = position
        self.mass = mass
        self.radius = radius
        self.past_positions = [] 
        self.clearance = []
        self.separation = []

bodies = [
    body("sun", 1.9884e30, [0, 0, 0], 0), 
    body("Earth", 5.9722e24, [0, 0, 0], 1.49598e11), #mass errors found at [https://web.archive.org/web/20161224174302/http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf]
    body("Mars", 6.4169e23, [0, 0, 0], 2.27956e11), 
    body("Venus", 4.8673e24, [0, 0, 0], 1.08210e11),
    body("asteroid1", 26.99e9, [x_ec, y_ec, z_ec], 0),
    body("asteroid2", 26.99e9, [x_ec, y_ec, z_ec], 0),
    body("asteroid3", 26.99e9, [x_ec, y_ec, z_ec], 0), #ephemis details found at https://doi.org/10.1016/j.icarus.2021.114594
    body("asteroid4", 26.99e9, [x_ec, y_ec, z_ec], 0),
    body("asteroid5", 26.99e9, [x_ec, y_ec, z_ec], 0),
    body("Mercury", 3.3010e23, [0, 0, 0], 5.7909e10), #radius values not accurate
    body("Jupiter", 1.8985e27, [0, 0, 0], 7.78479e11),
    body("Saturn", 5.6846e26, [0, 0, 0], 1.432041e12),
    body("Uranus", 8.6813e25, [0, 0, 0], 2.867043e12),
    body("Neptune", 1.0243e26, [0, 0, 0], 4.513953e12),
]

def compute_solar_distance(index, date):

    if index == 1:

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

    datetime_date = startdate + timedelta(seconds = t)

    date_string = datetime_date.strftime('%Y-%m-%d %H:%M:%S')

    date = ephem.Date(date_string)

    if index == 1:

        sun = ephem.Sun(date)

        longitude = sun.hlon
        latitude = sun.hlat

    if index == 2:

        mars = ephem.Mars(date)

        longitude = mars.hlong
        latitude = mars.hlat

    if index == 3:

        venus = ephem.Venus(date)

        longitude = venus.hlong
        latitude = venus.hlat

    if index == 9:

        mercury = ephem.Mercury(date)

        longitude = mercury.hlong
        latitude = mercury.hlat

    if index == 10:

        jupiter = ephem.Jupiter(date)

        longitude = jupiter.hlong
        latitude = jupiter.hlat

    if index == 11:

        saturn = ephem.Saturn(date)

        longitude = saturn.hlong
        latitude = saturn.hlat

    if index == 12:

        uranus = ephem.Uranus(date)

        longitude = uranus.hlong
        latitude = uranus.hlat

    if index == 13:

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

    r_body = [bodies[index].position[0] - bodies[asteroid_index].position[0],
            bodies[index].position[1] - bodies[asteroid_index].position[1],
            bodies[index].position[2] - bodies[asteroid_index].position[2]]

    return r_body

def new_compute_force(index, asteroid_index):

    displacement = compute_displacement(index, asteroid_index)

    force = [G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[0]/((np.linalg.norm(displacement)**3))),
            G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[1]/((np.linalg.norm(displacement)**3))),
            G * bodies[index].mass * bodies[asteroid_index].mass * (displacement[2]/((np.linalg.norm(displacement)**3)))]

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

#create empty coordinate lists

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

earth_final_pos = [0,0,0]
venus_final_pos = [0,0,0]
mars_final_pos = [0,0,0]

def time_step(time, step_size, asteroid_index, impulse, impulse_time):

    global earth_final_pos, venus_final_pos, mars_final_pos, mercury_final_pos, jupiter_final_pos, saturn_final_pos, uranus_final_pos, neptune_final_pos

    start_date_formatted = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")

    clearance_list = []
    separation_list = []

    initial_velocity = [0, 2547, 0]

    velocity = initial_velocity

    position = bodies[asteroid_index].position

    num_steps = int(time / step_size)

    asteroid_past_positions = [[] for _ in range(len(bodies) - 1)]

    bodies[1].position = planet_position(1, start_date_formatted, 0)
    bodies[2].position = planet_position(2, start_date_formatted,0)
    bodies[3].position = planet_position(3, start_date_formatted,0)
    bodies[9].position = planet_position(9, start_date_formatted, 0)
    bodies[10].position = planet_position(10, start_date_formatted, 0)
    bodies[11].position = planet_position(11, start_date_formatted, 0)
    bodies[12].position = planet_position(12, start_date_formatted, 0)
    bodies[13].position = planet_position(13, start_date_formatted, 0)

    for i in range(0, num_steps):

        bodies[1].position = planet_position(1, start_date_formatted, i * step_size)
        bodies[2].position = planet_position(2, start_date_formatted, i * step_size)
        bodies[3].position = planet_position(3, start_date_formatted, i * step_size)
        bodies[9].position = planet_position(9, start_date_formatted, i * step_size)
        bodies[10].position = planet_position(10, start_date_formatted, i * step_size)
        bodies[11].position = planet_position(11, start_date_formatted, i * step_size)
        bodies[12].position = planet_position(12, start_date_formatted, i * step_size)
        bodies[13].position = planet_position(13, start_date_formatted, i * step_size)

        sun_acceleration = compute_acceleration(0, asteroid_index)
        earth_acceleration = compute_acceleration(1, asteroid_index)
        mars_acceleration = compute_acceleration(2, asteroid_index)
        venus_acceleration = compute_acceleration(3, asteroid_index)
        mercury_acceleration = compute_acceleration(9, asteroid_index)
        jupiter_acceleration = compute_acceleration(10, asteroid_index)
        saturn_acceleration = compute_acceleration(11, asteroid_index)
        uranus_acceleration = compute_acceleration(12, asteroid_index)
        neptune_acceleration = compute_acceleration(13, asteroid_index)

        acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + venus_acceleration[0]
                        + mercury_acceleration[0] + jupiter_acceleration[0] + saturn_acceleration[0] + neptune_acceleration[0] + uranus_acceleration[0],
                        sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + venus_acceleration[1]
                        + mercury_acceleration[1] + jupiter_acceleration[1] + saturn_acceleration[1] + neptune_acceleration[1] + uranus_acceleration[1],
                        sun_acceleration[2] + earth_acceleration[2] + mars_acceleration[2] + venus_acceleration[2]
                        + mercury_acceleration[2] + jupiter_acceleration[2] + saturn_acceleration[2] + neptune_acceleration[2] + uranus_acceleration[2]]

        velocity = leapfrog_velocity(velocity, acceleration, step_size)

        position = leapfrog_position(position, velocity, acceleration, step_size)

        if i == int(impulse_time / step_size):

            velocity = [velocity[0] + impulse[0], velocity[1] + impulse[1], velocity[2] + impulse[2]]

            impulse_coords.append(bodies[asteroid_index].position)

        bodies[asteroid_index].position = position

        for k in range(1, 100):

            if i == int(k * num_steps/100):

                print("Asteroid number",asteroid_index-3, k, "percent done")

        earth_displacement = compute_displacement(1, asteroid_index)

        earth_list.append(bodies[1].position)
        mars_list.append(bodies[2].position)
        venus_list.append(bodies[3].position)
        mercury_list.append(bodies[9].position)
        jupiter_list.append(bodies[10].position)
        saturn_list.append(bodies[11].position)
        uranus_list.append(bodies[12].position)
        neptune_list.append(bodies[13].position)
        clearance_list.append(np.linalg.norm(earth_displacement))

        separation = compute_displacement(4, asteroid_index)
        
        asteroid_past_positions[asteroid_index - 1].append(bodies[asteroid_index].position)

        if asteroid_index == 4:

            time_list.append(i * step_size)
            earth_final_pos = bodies[1].position
            mars_final_pos = bodies[2].position
            venus_final_pos = bodies[3].position
            mercury_final_pos = bodies[9].position
            jupiter_final_pos = bodies[10].position
            saturn_final_pos = bodies[11].position
            neptune_final_pos = bodies[12].position
            uranus_final_pos = bodies[13].position
    
    bodies[asteroid_index].past_positions = asteroid_past_positions[asteroid_index - 1]
    bodies[asteroid_index].clearance = clearance_list
    bodies[asteroid_index].separation = separation_list

start_date_str = "2024-01-23 12:00:00"

sim_time = 1.708281e9 #3.3e7 for one orbit, 1.708008e8 = time between sep30 and impact
step = 10000

time_step(sim_time, step, 4, [0, 0, 0], 0) #time step 1000 for final sim
#time_step(sim_time, step, 5, [0, -3, 0], 1.4e7) #0.192e7 extra for 100ms change
#time_step(sim_time, step, 6, [0, -6, 0], 1.4e7) #0.3885e7 extra for 200ms change
#time_step(sim_time, step, 7, [0, -9, 0], 1.4e7) #0.59e7 extra for 300ms change
#time_step(sim_time, step, 8, [0, -12, 0], 1.4e7) #0.795e7 extra for 400ms change

start_time = 1.6e8
start_step = int(start_time / step)

x_earth = [coord[0] for coord in earth_list]
y_earth = [coord[1] for coord in earth_list]
z_earth = [coord[2] for coord in earth_list]

x_mars = [coord[0] for coord in mars_list]
y_mars = [coord[1] for coord in mars_list]
z_mars = [coord[2] for coord in mars_list]

x_venus = [coord[0] for coord in venus_list]
y_venus = [coord[1] for coord in venus_list]
z_venus = [coord[2] for coord in venus_list]

x_mercury = [coord[0] for coord in mercury_list]
y_mercury = [coord[1] for coord in mercury_list]
z_mercury = [coord[2] for coord in mercury_list]

x_jupiter = [coord[0] for coord in jupiter_list]
y_jupiter = [coord[1] for coord in jupiter_list]
z_jupiter = [coord[2] for coord in jupiter_list]

x_saturn = [coord[0] for coord in saturn_list]
y_saturn = [coord[1] for coord in saturn_list]
z_saturn = [coord[2] for coord in saturn_list]

x_uranus = [coord[0] for coord in uranus_list]
y_uranus = [coord[1] for coord in uranus_list]
z_uranus = [coord[2] for coord in uranus_list]

x_neptune = [coord[0] for coord in neptune_list]
y_neptune = [coord[1] for coord in neptune_list]
z_neptune = [coord[2] for coord in neptune_list]

x_asteroid1 = [coord[0] for coord in bodies[4].past_positions]
y_asteroid1 = [coord[1] for coord in bodies[4].past_positions]
z_asteroid1 = [coord[2] for coord in bodies[4].past_positions]

x_asteroid1 = x_asteroid1[start_step:]
y_asteroid1 = y_asteroid1[start_step:]
z_asteroid1 = z_asteroid1[start_step:]

x_asteroid2 = [coord[0] for coord in bodies[5].past_positions]
y_asteroid2 = [coord[1] for coord in bodies[5].past_positions]
z_asteroid2 = [coord[2] for coord in bodies[5].past_positions]

x_asteroid2 = x_asteroid2[start_step:]
y_asteroid2 = y_asteroid2[start_step:]
z_asteroid2 = z_asteroid2[start_step:]

x_asteroid3 = [coord[0] for coord in bodies[6].past_positions]
y_asteroid3 = [coord[1] for coord in bodies[6].past_positions]
z_asteroid3 = [coord[2] for coord in bodies[6].past_positions]

x_asteroid3 = x_asteroid3[start_step:]
y_asteroid3 = y_asteroid3[start_step:]
z_asteroid3 = z_asteroid3[start_step:]

x_asteroid4 = [coord[0] for coord in bodies[7].past_positions]
y_asteroid4 = [coord[1] for coord in bodies[7].past_positions]
z_asteroid4 = [coord[2] for coord in bodies[7].past_positions]

x_asteroid4 = x_asteroid4[start_step:]
y_asteroid4 = y_asteroid4[start_step:]
z_asteroid4 = z_asteroid4[start_step:]

x_asteroid5 = [coord[0] for coord in bodies[8].past_positions]
y_asteroid5 = [coord[1] for coord in bodies[8].past_positions]
z_asteroid5 = [coord[2] for coord in bodies[8].past_positions]

x_asteroid5 = x_asteroid5[start_step:]
y_asteroid5 = y_asteroid5[start_step:]
z_asteroid5 = z_asteroid5[start_step:]

clearance_diff4 = [bi - ai for ai, bi in zip(bodies[4].clearance, bodies[4].clearance)]
clearance_diff5 = [bi - ai for ai, bi in zip(bodies[4].clearance, bodies[5].clearance)]
clearance_diff6 = [bi - ai for ai, bi in zip(bodies[4].clearance, bodies[6].clearance)]
clearance_diff7 = [bi - ai for ai, bi in zip(bodies[4].clearance, bodies[7].clearance)]
clearance_diff8 = [bi - ai for ai, bi in zip(bodies[4].clearance, bodies[8].clearance)]

days = [t / 86400 for t in time_list]

def plot_positions():

    plt.figure(figsize=(6,6))

    plt.scatter(0, 0, color='yellow', marker='o', s=50, edgecolor='black', zorder=2)

    circle = plt.Circle((0,0), bodies[0].radius, edgecolor='orange', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[1].radius, edgecolor='black', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[2].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[3].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[9].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[10].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[11].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[12].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0,0), bodies[13].radius, edgecolor='slategray', facecolor='none', linestyle='dashed', linewidth=0.8)
    plt.gca().add_patch(circle)
    circle = plt.Circle((earth_final_pos[0], earth_final_pos[1]), 3.84e8, edgecolor='grey', facecolor='none', linewidth=2)
    plt.gca().add_patch(circle)
    circle = plt.Circle((earth_final_pos[0], earth_final_pos[1]), 6.378e6, edgecolor='none', facecolor='blue', linewidth=2)
    plt.gca().add_patch(circle)

    plt.scatter(mars_final_pos[0], mars_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(venus_final_pos[0], venus_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(earth_final_pos[0], earth_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(mercury_final_pos[0], mercury_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(jupiter_final_pos[0], jupiter_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(saturn_final_pos[0], saturn_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(uranus_final_pos[0], uranus_final_pos[1], marker='o', color='slategray', s=50)
    plt.scatter(neptune_final_pos[0], neptune_final_pos[1], marker='o', color='slategray', s=50)

    plt.scatter(bodies[5].position[0], bodies[5].position[1], marker='o', color='red', s=20)
    plt.scatter(bodies[6].position[0], bodies[6].position[1], marker='o', color='darkorange', s=20)
    plt.scatter(bodies[7].position[0], bodies[7].position[1], marker='o', color='greenyellow', s=20)
    plt.scatter(bodies[8].position[0], bodies[8].position[1], marker='o', color='green', s=20)
    plt.scatter(bodies[4].position[0], bodies[4].position[1], marker='o', color='maroon', s=2)

    plt.plot(x_asteroid1, y_asteroid1, linestyle='-', color='maroon', label='Undiverted asteroid', marker='')
    plt.plot(x_asteroid2, y_asteroid2, linestyle='-', color='red', label='8km/s diversion', marker='')
    plt.plot(x_asteroid3, y_asteroid3, linestyle='-', color='darkorange', label='16km/s diversion', marker='')
    plt.plot(x_asteroid4, y_asteroid4, linestyle='-', color='greenyellow', label='24km/s diversion', marker='')
    plt.plot(x_asteroid5, y_asteroid5, linestyle='-', color='green', label='32km/s diversion', marker='')

    plt.title("", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    plt.xlabel("x position (10  m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    plt.ylabel("y position (10  m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

    legend_elements = [
        Line2D([0], [0], linewidth=2, linestyle='-', color='maroon', markersize=10, label='Undiverted asteroid'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='red', markersize=10, label='8km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='darkorange', markersize=10, label='16km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='greenyellow', markersize=10, label='24km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='green', markersize=10, label='32km/s diversion'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='grey', markersize=10, label='Lunar orbit'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='black', markersize=10, label='Earth orbit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Earth'),
    ]

    plt.legend(handles=legend_elements, prop = legend_font, loc = 'upper left')
    plt.axis('equal')
    plt.show()

def plot_clearance():

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(days, bodies[8].clearance, linestyle='-', color='green', label='32km/s diversion', marker='')
    ax1.plot(days, bodies[7].clearance, linestyle='-', color='greenyellow', label='24km/s diversion', marker='')
    ax1.plot(days, bodies[6].clearance, linestyle='-', color='darkorange', label='16km/s diversion', marker='')
    ax1.plot(days, bodies[5].clearance, linestyle='-', color='red', label='8km/s diversion', marker='')
    ax1.plot(days, bodies[4].clearance, linestyle='-', color='maroon', label='Undiverted asteroid', marker='')

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

    ax1.set_xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    ax1.set_ylabel("Earth clearance (10  m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    ax1.legend(prop = legend_font, loc = 'lower left')

    ax2.plot(days, clearance_diff8, linestyle='-', color='green', label='32km/s diversion', marker='')
    ax2.plot(days, clearance_diff7, linestyle='-', color='greenyellow', label='24km/s diversion', marker='')
    ax2.plot(days, clearance_diff6, linestyle='-', color='darkorange', label='16km/s diversion', marker='')
    ax2.plot(days, clearance_diff5, linestyle='-', color='red', label='8km/s diversion', marker='')
    ax2.plot(days, clearance_diff4, linestyle='-', color='maroon', label='Undiverted baseline', marker='')

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

    ax2.set_xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    ax2.set_ylabel("Undiverted asteroid deviation (10  m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    ax2.legend(prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

plot_positions()

#plot_clearance()

print(earth_final_pos)