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
    body("Rocket", 30000, [r_L2, 0.1])
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

def compute_force(index):

    displacement = compute_displacement(index)

    distance = np.linalg.norm(displacement)

    angle = math.atan2(displacement[1], displacement[0])

    force = (G * bodies[2].mass * bodies[index].mass) / (distance ** 2)

    x_force = - force * math.cos(angle)
    y_force = - force * math.sin(angle)

    return x_force, y_force

force = compute_force(0) + compute_force(1)

acceleration = [force[0] / bodies[2].mass, [force[1] / bodies[1].mass]]

initial_vel = [0,(2 * pi * r_L2) / period]

#create empty coordinate lists

num_coordinates = int(period/8)

earth_list = []
moon_list = []
rocket_list = []

#loop over orbits and populate coordinate lists

def time_step(steps):

    for i in range(0, steps):

        bodies[0].position = planet_position(-COM, period, i)
        bodies[1].position = planet_position(d - COM, period, i)

        print(compute_force(0))

        earth_list.append(bodies[0].position)
        moon_list.append(bodies[1].position)

time_step(num_coordinates)

x_earth = [coord[0] for coord in earth_list]
y_earth = [coord[1] for coord in earth_list]

x_moon = [coord[0] for coord in moon_list]
y_moon = [coord[1] for coord in moon_list]

#plot results

plt.figure(figsize=(8,8))
plt.plot(x_earth, y_earth, linestyle = '-', color = 'b', label = 'Earth')
plt.plot(x_moon, y_moon, linestyle = '-', color = 'r', label = 'Moon')

plt.title("Orbital positions")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()