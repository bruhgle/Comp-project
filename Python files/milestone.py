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
    body("Rocket", 30000, [r_L2, 0])
]

#define function to compute period

def compute_period(m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((d ** 3 ) / (G * m_total)) ** 0.5

    return(p)

period = compute_period(bodies[0].mass, bodies[1].mass)

#define function for planet position

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t / period)
    y = distance * math.sin(2 * pi * t / period)

    return x, y

#define function to find displacement of rocket from planet {index} (0 = earth, 1 = moon)

def compute_displacement(index):

    r_body = [bodies[2].position[0] - bodies[index].position[0], bodies[2].position[1] - bodies[index].position[1]]

    return r_body

#define function to find force on rocket from planet {index}

def compute_force(displacement, index):

    disp = displacement[index]

    distance = np.linalg.norm(disp)

    angle = math.atan2(disp[1], disp[0])

    force = (G * bodies[2].mass * bodies[index].mass) / (distance ** 2)

    x_force = - force * math.cos(angle)
    y_force = - force * math.sin(angle)

    return x_force, y_force

def runge_kutta(initial_new, old_variable, step):

    z1x = step * old_variable[0]
    z1y = step * old_variable[1]
    
    z2x = step * (old_variable[0] + (z1x / 2))
    z2y = step * (old_variable[1] + (z1y / 2))
    
    z3x = step * (old_variable[0] + (z2x / 2))
    z3y = step * (old_variable[1] + (z2y / 2))

    z4x = step * (old_variable[0] + z3x)
    z4y = step * (old_variable[1] + z3y)
    
    new_variable_x = initial_new[0] + (z1x + (2 * z2x) + (2 * z3x) + z4x) / 6
    new_variable_y = initial_new[1] + (z1y + (2 * z2y) + (2 * z3y) + z4y) / 6
    
    return new_variable_x, new_variable_y

RK_step = 500

def compute_velocity(v_initial, acceleration, step):

    velocity = runge_kutta(v_initial, acceleration, step)

    return velocity

def compute_position(pos_initial, velocity, step):
    
    position = runge_kutta(pos_initial, velocity, step)

    return position

#create empty coordinate lists

num_coordinates = 5

earth_list = []
moon_list = []
rocket_list = []

#loop over orbits and populate coordinate lists

def time_step(steps):
    
    for i in range(0, steps):
        bodies[0].position = planet_position(-COM, period, i)
        bodies[1].position = planet_position(d - COM, period, i)
        
        # Calculate displacement vector for each iteration
        displacement = [compute_displacement(0), compute_displacement(1)]
        
        earth_force = compute_force(displacement, 0)
        moon_force = compute_force(displacement, 1)

        force = [earth_force[0] + moon_force[0], earth_force[1] + moon_force[1]]

        acceleration = [force[0] / bodies[2].mass, force[1] / bodies[2].mass]

        initial_vel = [0, (2 * pi * r_L2) / period]

        velocity = compute_velocity(initial_vel, acceleration, RK_step)

        position = compute_position([r_L2, 0], velocity, RK_step)

        bodies[2].position = position

        earth_list.append(bodies[0].position)
        moon_list.append(bodies[1].position)
        rocket_list.append(bodies[2].position)

time_step(num_coordinates)

x_earth = [coord[0] for coord in earth_list]
y_earth = [coord[1] for coord in earth_list]

x_moon = [coord[0] for coord in moon_list]
y_moon = [coord[1] for coord in moon_list]

x_rocket = [coord[0] for coord in rocket_list]
y_rocket = [coord[1] for coord in rocket_list]

#plot results

plt.figure(figsize=(8,8))
plt.plot(x_earth, y_earth, linestyle = '-', color = 'b', label = 'Earth')
plt.plot(x_moon, y_moon, linestyle = '-', color = 'grey', label = 'Moon')
plt.plot(x_rocket, y_rocket, linestyle = '-', color = 'r', label = 'Rocket')

plt.title("Orbital positions")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()