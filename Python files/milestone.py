#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#define initial varibles

m_earth = 5.9742e24
m_moon = 7.35e22
G = 6.6726e-11
d = 3.84e8
pi = np.pi
RK_step = 10
taylor_modifier = 1.006787964156102059
runge_modifier = 1.0078
runge_modifier_vel = 1.0078

#define COM for earth and moon

COM = (m_moon * d) / (m_earth + m_moon)

#define general body class

class body:
    def __init__(self, name, mass, position):
        self.name = name
        self.position = position
        self.mass = mass

#calculate L2 point

r_L2 = d * (1 + (( m_moon / (3 * m_earth)) ** (1/3))) - COM

#define object for each body using body class

bodies = [
    body("Earth", 5.9742e24, [-COM, 0]), 
    body("Moon", 7.35e22, [d - COM, 0]), 
    body("Rocket", 30000, [0, 0])
]

#define function to compute period

def compute_period(m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((d ** 3 ) / (G * m_total)) ** 0.5

    return(p)

period = compute_period(bodies[0].mass, bodies[1].mass)

#define function for planet position

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t * RK_step / period)
    y = distance * math.sin(2 * pi * t * RK_step / period)

    return x, y

#define function to find displacement of rocket from planet {index} (0 = earth, 1 = moon)

def compute_displacement(index):

    r_body = [bodies[index].position[0] - bodies[2].position[0], bodies[index].position[1] - bodies[2].position[1]]

    return r_body

#define function to find force on rocket from planet {index}

def compute_force(displacement, index):

    disp = displacement[index]

    distance = np.linalg.norm(disp)

    angle = math.atan2(disp[1], disp[0])

    force = (G * bodies[2].mass * bodies[index].mass) / (distance ** 2)

    x_force = force * math.cos(angle)
    y_force = force * math.sin(angle)

    return x_force, y_force

def acceleration_equation(rocket_pos, t):

    earth_pos = planet_position(-COM, period, t)
    moon_pos = planet_position(d-COM, period, t)

    earth_displacement = [rocket_pos[0] - earth_pos[0], rocket_pos[1] - earth_pos[1]]
    earth_acceleration = [- G * bodies[0].mass * (earth_displacement[0]/np.linalg.norm(earth_displacement)), - G * bodies[0].mass * (earth_displacement[1]/np.linalg.norm(earth_displacement))]

    moon_displacement = [rocket_pos[0] - moon_pos[0], rocket_pos[1] - moon_pos[1]]
    moon_acceleration = [- G * bodies[1].mass * (moon_displacement[0]/np.linalg.norm(moon_displacement)), - G * bodies[1].mass * (moon_displacement[1]/np.linalg.norm(moon_displacement))]

    acceleration = earth_acceleration + moon_acceleration

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

def taylor_position(position, velocity, acceleration, step):

    new_position_x = position[0] + (step * velocity[0]) + ((step ** 2) * acceleration[0] / 2)
    new_position_y = position[1] + (step * velocity[1]) + ((step ** 2) * acceleration[1] / 2)

    return new_position_x, new_position_y

def taylor_velocity(velocity, acceleration, step):

    new_velocity_x = velocity[0] + (step * acceleration[0])
    new_velocity_y = velocity[1] + (step * acceleration[1])

    return new_velocity_x, new_velocity_y

#create empty coordinate lists

earth_list_taylor = []
moon_list_taylor = []
rocket_list_taylor = []
deviation_list_taylor = []
time_list_taylor = []

earth_list_runge = []
moon_list_runge = []
rocket_list_runge = []
deviation_list_runge = []
time_list_runge = []

#loop over orbits and populate coordinate lists

def time_step_runge(steps):
    
    initial_vel = [0, (2 * pi * (r_L2 * runge_modifier_vel)/period)]
    
    velocity = initial_vel

    bodies[2].position = [runge_modifier * r_L2, 0]

    position = bodies[2].position
    
    for i in range(0, steps):
        bodies[0].position = planet_position(-COM, period, i)
        bodies[1].position = planet_position(d - COM, period, i)
        
        displacement = [compute_displacement(0), compute_displacement(1)]
        
        earth_force = compute_force(displacement, 0)
        moon_force = compute_force(displacement, 1)

        force = [earth_force[0] + moon_force[0], earth_force[1] + moon_force[1]]

        acceleration = [force[0] / bodies[2].mass, force[1] / bodies[2].mass]

        velocity = leapfrog_velocity(velocity, acceleration, RK_step)

        position = leapfrog_position(position, velocity, acceleration, RK_step)

        bodies[2].position = position

        deviation = np.linalg.norm(bodies[2].position) - (r_L2 * runge_modifier)

        for k in range(1, 100):

            if i == int(k * steps/100):

                print(k, np.linalg.norm(velocity))

        if i == steps - 1:
            
            i_r = runge_modifier * r_L2 / 1000
            f_r = bodies[2].position[0] / 1000
            miss_cm = (bodies[2].position[0] - (runge_modifier * r_L2)) * 100
            accuracy = bodies[2].position[0] / (runge_modifier * r_L2)

            formatted_ir = "{:.2f}".format(i_r)
            formatted_fr = "{:.2f}".format(f_r)
            formatted_miss = "{:.2f}".format(miss_cm)
            formatted_accuracy = "{:.15f}".format(accuracy)

            print("Initial radius:", formatted_ir, "km")
            print("Final radius:", formatted_fr, "km")
            print("Missed by", formatted_miss, "cm")
            print("Accuracy:", formatted_accuracy)

        earth_list_runge.append([bodies[0].position[0], bodies[0].position[1]])
        moon_list_runge.append([bodies[1].position[0], bodies[1].position[1]])
        rocket_list_runge.append([bodies[2].position[0], bodies[2].position[1]])
        deviation_list_runge.append(deviation/1000)

def time_step_taylor(steps):
    
    initial_vel = [0, (2 * pi * (r_L2 * taylor_modifier)/period)]
    
    velocity = initial_vel

    bodies[2].position = [taylor_modifier * r_L2, 0]

    print(initial_vel)

    position = bodies[2].position
    
    for i in range(0, steps):
        bodies[0].position = planet_position(-COM, period, i)
        bodies[1].position = planet_position(d - COM, period, i)
        
        displacement = [compute_displacement(0), compute_displacement(1)]
        
        earth_force = compute_force(displacement, 0)
        moon_force = compute_force(displacement, 1)

        force = [earth_force[0] + moon_force[0], earth_force[1] + moon_force[1]]

        acceleration = [force[0] / bodies[2].mass, force[1] / bodies[2].mass]

        velocity = taylor_velocity(velocity, acceleration, RK_step)

        position = taylor_position(position, velocity, acceleration, RK_step)

        bodies[2].position = position

        deviation = np.linalg.norm(bodies[2].position) - (r_L2 * taylor_modifier)

        for k in range(1, 100):

            if i == int(k * steps/100):

                print(k,np.linalg.norm(velocity))

        if i == steps - 1:
            
            i_r = taylor_modifier * r_L2 / 1000
            f_r = bodies[2].position[0] / 1000
            miss_cm = (bodies[2].position[0] - (taylor_modifier * r_L2)) * 100
            accuracy = bodies[2].position[0] / (taylor_modifier * r_L2)

            formatted_ir = "{:.2f}".format(i_r)
            formatted_fr = "{:.2f}".format(f_r)
            formatted_miss = "{:.2f}".format(miss_cm)
            formatted_accuracy = "{:.15f}".format(accuracy)

            print("Initial radius:", formatted_ir, "km")
            print("Final radius:", formatted_fr, "km")
            print("Missed by", formatted_miss, "cm")
            print("Accuracy:", formatted_accuracy)

        earth_list_taylor.append([bodies[0].position[0], bodies[0].position[1]])
        moon_list_taylor.append([bodies[1].position[0], bodies[1].position[1]])
        rocket_list_taylor.append([bodies[2].position[0], bodies[2].position[1]])
        deviation_list_taylor.append(deviation/1000)

num_coordinates = int(1 * (1/RK_step) * period)

time_step_taylor(num_coordinates)

time_step_runge(num_coordinates)

x_earth_taylor = [coord[0] for coord in earth_list_taylor]
y_earth_taylor = [coord[1] for coord in earth_list_taylor]

x_moon_taylor = [coord[0] for coord in moon_list_taylor]
y_moon_taylor = [coord[1] for coord in moon_list_taylor]

x_rocket_taylor = [coord[0] for coord in rocket_list_taylor]
y_rocket_taylor = [coord[1] for coord in rocket_list_taylor]

x_earth_runge = [coord[0] for coord in earth_list_runge]
y_earth_runge = [coord[1] for coord in earth_list_runge]

x_moon_runge = [coord[0] for coord in moon_list_runge]
y_moon_runge = [coord[1] for coord in moon_list_runge]

x_rocket_runge = [coord[0] for coord in rocket_list_runge]
y_rocket_runge = [coord[1] for coord in rocket_list_runge]

step_size = period / num_coordinates

for i in range(0, num_coordinates):

    time_list_taylor.append(i * step_size)
    time_list_runge.append(i * step_size)

days = [t / 86400 for t in time_list_taylor]
days = [t / 86400 for t in time_list_runge]

#plot results

plt.figure(figsize=(6,6))
plt.plot(x_earth_taylor, y_earth_taylor, linestyle = '-', color = 'b', label = 'Earth')
plt.plot(x_moon_taylor, y_moon_taylor, linestyle = '-', color = 'grey', label = 'Moon', marker = '')
plt.plot(x_rocket_taylor, y_rocket_taylor, linestyle = '-', color = 'r', label = 'Rocket', marker = '')
plt.title("", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.xlabel("x position (10  km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("y position (10  km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

plt.legend(prop = legend_font, loc = 'upper right')
plt.axis('equal')
plt.xlim(-15e8, 15e8)
plt.ylim(-15e8, 15e8)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(x_earth_runge, y_earth_runge, linestyle = '-', color = 'b', label = 'Earth')
plt.plot(x_moon_runge, y_moon_runge, linestyle = '-', color = 'grey', label = 'Moon', marker = '')
plt.plot(x_rocket_runge, y_rocket_runge, linestyle = '-', color = 'r', label = 'Rocket', marker = '')
plt.title("", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.xlabel("x position (10  km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("y position (10  km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

plt.legend(prop = legend_font, loc = 'upper right')
plt.axis('equal')
plt.xlim(-1e9, 1e9)
plt.ylim(-1e9, 1e9)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(days, deviation_list_taylor, linestyle = '-', color = 'b')
plt.title("")
plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("Deviation (km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.show()

plt.figure(figsize=(8,4))
plt.plot(days, deviation_list_runge, linestyle = '-', color = 'b')
plt.title("")
plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("Deviation (km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.show()