#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt

#define initial varibles

m_earth = 5.9742e24
m_moon = 7.35e22
G = 6.6726e-11
d = 3.84e8
pi = np.pi
RK_step = 10
modifier = 1.006787964156102055

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
    body("Rocket", 30000, [modifier * r_L2, 0])
]

#define function to compute period

def compute_period(m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((d ** 3 ) / (G * m_total)) ** 0.5

    return(p)

period = compute_period(bodies[0].mass, bodies[1].mass)

#define function for planet position

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t * RK_step/ period)
    y = distance * math.sin(2 * pi * t * RK_step/ period)

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

def runge_kutta(new_variable, old_variable, step):

    z1x = step * old_variable[0]
    z1y = step * old_variable[1]
    
    z2x = step * (old_variable[0] + (z1x / 2))
    z2y = step * (old_variable[1] + (z1y / 2))
    
    z3x = step * (old_variable[0] + (z2x / 2))
    z3y = step * (old_variable[1] + (z2y / 2))

    z4x = step * (old_variable[0] + z3x)
    z4y = step * (old_variable[1] + z3y)
    
    new_variable_x = new_variable[0] + (z1x + (2 * z2x) + (2 * z3x) + z4x) / 6
    new_variable_y = new_variable[1] + (z1y + (2 * z2y) + (2 * z3y) + z4y) / 6
    
    return new_variable_x, new_variable_y

def taylor_position(position, velocity, acceleration, step):

    new_position_x = position[0] + (step * velocity[0]) + ((step ** 2) * acceleration[0] / 2)
    new_position_y = position[1] + (step * velocity[1]) + ((step ** 2) * acceleration[1] / 2)

    return new_position_x, new_position_y

def taylor_velocity(velocity, acceleration, step):

    new_velocity_x = velocity[0] + (step * acceleration[0])
    new_velocity_y = velocity[1] + (step * acceleration[1])

    return new_velocity_x, new_velocity_y

#create empty coordinate lists

num_coordinates = int((1/RK_step) * period)

earth_list = []
moon_list = []
rocket_list = []
deviation_list = []
time_list = []

#loop over orbits and populate coordinate lists

def time_step(steps):
    
    initial_vel = [0, (2 * pi * (r_L2 * modifier)/period)]
    
    velocity = initial_vel

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

        deviation = np.linalg.norm(bodies[2].position) - (r_L2 * modifier)

        for k in range(1, 100):

            if i == int(k * steps/100):

                print(k,"percent done")

        if i == steps - 1:
            
            i_r = modifier * r_L2 / 1000
            f_r = bodies[2].position[0] / 1000
            miss_cm = (bodies[2].position[0] - (modifier * r_L2)) * 100
            accuracy = bodies[2].position[0] / (modifier * r_L2)

            formatted_ir = "{:.2f}".format(i_r)
            formatted_fr = "{:.2f}".format(f_r)
            formatted_miss = "{:.2f}".format(miss_cm)
            formatted_accuracy = "{:.15f}".format(accuracy)

            print("Initial radius:", formatted_ir, "km")
            print("Final radius:", formatted_fr, "km")
            print("Missed by", formatted_miss, "cm")
            print("Accuracy:", formatted_accuracy)

        earth_list.append([bodies[0].position[0], bodies[0].position[1]])
        moon_list.append([bodies[1].position[0], bodies[1].position[1]])
        rocket_list.append([bodies[2].position[0], bodies[2].position[1]])
        deviation_list.append(deviation/1000)

time_step(num_coordinates)

x_earth = [coord[0] for coord in earth_list]
y_earth = [coord[1] for coord in earth_list]

x_moon = [coord[0] for coord in moon_list]
y_moon = [coord[1] for coord in moon_list]

x_rocket = [coord[0] for coord in rocket_list]
y_rocket = [coord[1] for coord in rocket_list]

step_size = period / num_coordinates

for i in range(0, num_coordinates):

    time_list.append(i * step_size)

days = [t / 86400 for t in time_list]

#plot results

plt.figure(figsize=(6,6))
plt.plot(x_earth, y_earth, linestyle = '-', color = 'b', label = 'Earth')
plt.plot(x_moon, y_moon, linestyle = '-', color = 'grey', label = 'Moon', marker = '')
plt.plot(x_rocket, y_rocket, linestyle = '-', color = 'r', label = 'Rocket', marker = '')
plt.title("Orbital positions", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.xlabel("x position (10  km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})
plt.ylabel("y position (10  km)", fontdict={'family': 'DejaVu Serif', 'color':  'black', 'weight': 'normal', 'size': 11})

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

plt.legend(prop = legend_font, loc = 'upper right')
plt.axis('equal')
plt.xlim(-5e8, 5e8)
plt.ylim(-5e8, 5e8)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(days, deviation_list, linestyle = '-', color = 'b')
plt.title("Deviation of rocket from perfect circle")
plt.xlabel("Time (days)")
plt.ylabel("Deviation (km)")
plt.ylim(-6, 0.5)
plt.show()