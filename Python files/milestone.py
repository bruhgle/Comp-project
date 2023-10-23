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

#define period for the earth and moon

period = compute_period(bodies[0].mass, bodies[1].mass)

#define function for planet position

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t / period)
    y = distance * math.sin(2 * pi * t / period)

    return x, y

#create empty coordinate lists

num_coordinates = 2

earth_list = []
moon_list = []
rocket_list = []

#loop over orbits and populate coordinate lists

for i in range(0, num_coordinates):

    bodies[0].position = planet_position(-COM, period, i)
    bodies[1].position = planet_position(d - COM, period, i)

    earth_list.append(bodies[0].position)
    moon_list.append(bodies[1].position)

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

r_earth = [bodies[2].position[0] - bodies[0].position[0], bodies[2].position[1] - bodies[0].position[1]]
r_moon = [bodies[2].position[0] - bodies[1].position[0], bodies[2].position[1] - bodies[1].position[1]]

d_earth = np.linalg.norm(r_earth)
d_moon = np.linalg.norm(r_moon)

print(d_earth)

F_earth = [( - G * bodies[2].mass * bodies[0].mass) / (r_earth[0] ** 2), ( - G * bodies[2].mass * bodies[0].mass) / (r_earth[1] ** 2)]
F_moon = [( - G * bodies[2].mass * bodies[0].mass) / (r_moon[0] ** 2), ( - G * bodies[2].mass * bodies[0].mass) / (r_moon[1] ** 2)]

acceleration = [(F_earth[0] / bodies[2].mass) + (F_moon[0] / bodies[2].mass), ((F_earth[1] / bodies[2].mass) + (F_moon[1] / bodies[2].mass))]

initial_vel = [0,(2 * pi * r_L2) / period]

