#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import ephem
from matplotlib.patches import Ellipse

#define global variables

G = 6.6726e-11
pi = np.pi

start_date_str = "2028-09-24 00:00:00"
start_pos = [1.140858785274808E+11,   2.202717727038012E+10,   1.505460279676625E+09]
start_vel = [-3.072333498677471E+03,   3.622336749588278E+04,  -2.006119059140651E+03]
start_pos_sigma = [3.81924509E+02,          1.28002435E+03,          3.90649455E+02]
start_vel_sigma = [3.70165958E-04,          6.37531330E-05,          1.05375217E-04]
sim_time = 17566400
step = 1000
start_time = 0
num_asteroids = 1

class body:
    def __init__(self, name, mass, position, asteroid_acceleration, radius):
        self.name = name
        self.position = position
        self.asteroid_acceleration = asteroid_acceleration
        self.mass = mass
        self.radius = radius
        self.past_positions = []
        self.clearance = []
        self.separation = []

bodies = [
    body("sun", 1.9884e30, [0, 0, 0], [0, 0, 0], 0),                #0   
    body("Mercury", 3.3010e23, [0, 0, 0], [0, 0, 0], 5.7909e10),    #1
    body("Venus", 4.8673e24, [0, 0, 0], [0, 0, 0], 1.08210e11),     #2
    body("Earth", 5.9722e24, [0, 0, 0], [0, 0, 0], 1.49598e11),     #3           #mass errors found at [https://web.archive.org/web/20161224174302/http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf]
    body("Mars", 6.4169e23, [0, 0, 0], [0, 0, 0], 2.27956e11),      #4
    body("Jupiter", 1.8985e27, [0, 0, 0], [0, 0, 0], 7.78479e11),   #5
    body("Saturn", 5.6846e26, [0, 0, 0], [0, 0, 0], 1.432041e12),   #6
    body("Uranus", 8.6813e25, [0, 0, 0], [0, 0, 0], 2.867043e12),   #7
    body("Neptune", 1.0243e26, [0, 0, 0], [0, 0, 0], 4.513953e12),  #8
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

def compute_moid(index):

    moid = min(bodies[index].clearance)

    return moid

for i in range(1, 1 + num_asteroids):
    asteroid_name = f"asteroid{i}"
    randposx = random_value(start_pos[0],start_pos_sigma[0])
    randposy = random_value(start_pos[1],start_pos_sigma[1])
    randposz = random_value(start_pos[2],start_pos_sigma[2])
    randmass = 4.0e+10
    asteroid = body(asteroid_name, randmass, [randposx, randposy, randposz], [0,0,0], 0)
    bodies.append(asteroid)

def calculate_average(entries):
    averages = []
    for i in range(0, len(entries), num_asteroids):
        sublist = entries[i:i+num_asteroids]
        avg = sum(sublist) / len(sublist)
        averages.append(avg)
    return averages

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

acceleration_list = []

earth_final_pos = [0,0,0]
venus_final_pos = [0,0,0]
mars_final_pos = [0,0,0]

def time_step(time, step_size, asteroid_index, impulse, impulse_time):

    global earth_final_pos, venus_final_pos, mars_final_pos, mercury_final_pos, jupiter_final_pos, saturn_final_pos, uranus_final_pos, neptune_final_pos

    start_date_formatted = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")

    clearance_list = []
    separation_list = []

    initial_velocity = [random_value(start_vel[0], start_vel_sigma[0]),
                        random_value(start_vel[1], start_vel_sigma[1]),
                        random_value(start_vel[2], start_vel_sigma[2])]

    velocity = initial_velocity

    position = bodies[asteroid_index].position

    num_steps = int(time / step_size)

    asteroid_past_positions = [[] for _ in range(len(bodies) - 1)]

    for i in range(1,9):
        
        bodies[i].position = planet_position(i, start_date_formatted, 0)

    for i in range(0, num_steps):

        for j in range(1,9):

            bodies[j].position = planet_position(j, start_date_formatted, i * step_size)

        sun_acceleration = compute_acceleration(0, asteroid_index)
        mercury_acceleration = compute_acceleration(1, asteroid_index)
        venus_acceleration = compute_acceleration(2, asteroid_index)
        earth_acceleration = compute_acceleration(3, asteroid_index)
        mars_acceleration = compute_acceleration(4, asteroid_index)
        jupiter_acceleration = compute_acceleration(5, asteroid_index)
        saturn_acceleration = compute_acceleration(6, asteroid_index)
        uranus_acceleration = compute_acceleration(7, asteroid_index)
        neptune_acceleration = compute_acceleration(8, asteroid_index)

        acceleration = [sun_acceleration[0] + earth_acceleration[0] + mars_acceleration[0] + venus_acceleration[0]
                        + mercury_acceleration[0] + jupiter_acceleration[0] + saturn_acceleration[0] + neptune_acceleration[0] + uranus_acceleration[0],
                        sun_acceleration[1] + earth_acceleration[1] + mars_acceleration[1] + venus_acceleration[1]
                        + mercury_acceleration[1] + jupiter_acceleration[1] + saturn_acceleration[1] + neptune_acceleration[1] + uranus_acceleration[1],
                        sun_acceleration[2] + earth_acceleration[2] + mars_acceleration[2] + venus_acceleration[2]
                        + mercury_acceleration[2] + jupiter_acceleration[2] + saturn_acceleration[2] + neptune_acceleration[2] + uranus_acceleration[2]]

        velocity = leapfrog_velocity(velocity, acceleration, step_size)

        position = leapfrog_position(position, velocity, acceleration, step_size)

        if i == int(impulse_time / step_size):

            velocity = [velocity[0] + impulse[0],
                        velocity[1] + impulse[1],
                        velocity[2] + impulse[2]]

            impulse_coords.append(bodies[asteroid_index].position)

        bodies[asteroid_index].position = position

        for k in range(1, 100):

            if i == int(k * num_steps/100):

                print("Asteroid",asteroid_index-8, k, "%, asteroid position =", bodies[9].position)

        earth_displacement = compute_displacement(1, asteroid_index)

        mercury_list.append(bodies[1].position)
        venus_list.append(bodies[2].position)
        earth_list.append(bodies[3].position)
        mars_list.append(bodies[4].position)
        jupiter_list.append(bodies[5].position)
        saturn_list.append(bodies[6].position)
        uranus_list.append(bodies[7].position)
        neptune_list.append(bodies[8].position)
        clearance_list.append(np.linalg.norm(earth_displacement))

        separation = compute_displacement(4, asteroid_index)
        
        asteroid_past_positions[asteroid_index - 1].append(bodies[asteroid_index].position)

        if asteroid_index == 9:

            time_list.append(i * step_size)
            mercury_final_pos = bodies[1].position
            venus_final_pos = bodies[2].position
            earth_final_pos = bodies[3].position
            mars_final_pos = bodies[4].position
            jupiter_final_pos = bodies[5].position
            saturn_final_pos = bodies[6].position
            neptune_final_pos = bodies[7].position
            uranus_final_pos = bodies[8].position
    
    bodies[asteroid_index].past_positions = asteroid_past_positions[asteroid_index - 1]
    bodies[asteroid_index].clearance = clearance_list
    bodies[asteroid_index].separation = separation_list

for i in range(num_asteroids):

    time_step(sim_time, step, 9 + i, [0, 0, 0], 0)

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

x_asteroids = []
y_asteroids = []
z_asteroids = []

for i in range(9, 9 + num_asteroids):
    x_asteroid = [coord[0] for coord in bodies[i].past_positions]
    y_asteroid = [coord[1] for coord in bodies[i].past_positions]
    z_asteroid = [coord[2] for coord in bodies[i].past_positions]

    x_asteroid = x_asteroid[start_step:]
    y_asteroid = y_asteroid[start_step:]
    z_asteroid = z_asteroid[start_step:]

    x_asteroids.append(x_asteroid)
    y_asteroids.append(y_asteroid)
    z_asteroids.append(z_asteroid)

clearance_diffs = []

for i in range(9, 9 + (num_asteroids)):

    clearance_diffs.append(bodies[i].clearance)

days = [t / 86400 for t in time_list]

def plot_positions():

    plt.figure(figsize=(9,9))

    plt.xlim(-3e11,3e11)
    plt.ylim(-3e11,3e11)

    plt.scatter(0, 0, color='yellow', marker='o', s=50, edgecolor='black', zorder=2)

    for i in range(1, 9):

        circle = plt.Circle((0,0), bodies[i].radius, edgecolor='black', facecolor='none', linestyle='dashed', linewidth=0.8)
        plt.gca().add_patch(circle)

    circle = plt.Circle((earth_final_pos[0], earth_final_pos[1]), 3.84e8, edgecolor='grey', facecolor='none', linewidth=2)
    plt.gca().add_patch(circle)
    circle = plt.Circle((earth_final_pos[0], earth_final_pos[1]), 6.378e6, edgecolor='none', facecolor='blue', linewidth=2)
    plt.gca().add_patch(circle)

    plt.scatter(mars_final_pos[0], mars_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(venus_final_pos[0], venus_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(earth_final_pos[0], earth_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(mercury_final_pos[0], mercury_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(jupiter_final_pos[0], jupiter_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(saturn_final_pos[0], saturn_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(uranus_final_pos[0], uranus_final_pos[1], marker='o', color='black', s=50)
    plt.scatter(neptune_final_pos[0], neptune_final_pos[1], marker='o', color='black', s=50)

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'cyan', 'brown', 'gray', 'black']

    for i in range(9,(num_asteroids)+9):
        
        plt.scatter(bodies[i].position[0], bodies[i].position[1], marker='o', color=colors[i % len(colors)], s=20)

    for i in range(num_asteroids):

        plt.plot(x_asteroids[i], y_asteroids[i], linestyle='-', color=colors[i % len(colors)], marker='')

    plt.title("", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    plt.xlabel("x position (10  m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    plt.ylabel("y position (10  m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

    #legend_elements = [
        #Line2D([0], [0], linewidth=2, linestyle='-', color='maroon', markersize=10, label='Undiverted asteroid'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='red', markersize=10, label='8km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='darkorange', markersize=10, label='16km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='green', markersize=10, label='32km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='greenyellow', markersize=10, label='24km/s diversion'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='grey', markersize=10, label='Lunar orbit'),
        #Line2D([0], [0], linewidth=2, linestyle='-', color='black', markersize=10, label='Earth orbit'),
        #Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Earth'),
    #]

    #plt.legend(handles=legend_elements, prop = legend_font, loc = 'upper left')
    plt.show()

def plot_clearance():

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i in range(9, 9 + num_asteroids):
        
        ax1.plot(days, bodies[i].clearance, linestyle='-', color='green', marker='')

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

    ax1.set_xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    ax1.set_ylabel("Earth clearance (10  m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11})
    ax1.legend(prop = legend_font, loc = 'lower left')

    for i in range(num_asteroids):

        ax2.plot(days, clearance_diffs[i], linestyle='-', color='green', label='32km/s diversion', marker='')
    
    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

    ax2.set_xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    ax2.set_ylabel("Undiverted asteroid deviation (10  m)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
    ax2.legend(prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

print(datetime_date)

moids = []

for i in range(9,9+num_asteroids):

    moids.append(compute_moid(i))

average_moids = calculate_average(moids)

print(average_moids)

print("Asteroid final position:", bodies[9].position)

plot_positions()

plot_clearance()