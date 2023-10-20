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

#define object for each body using above class

bodies = [
    body("Earth", 5.9742e24, [-COM, 0]), 
    body("Moon", 7.35e22, [d - COM, 0]), 
    body("Rocket", 30000, [-COM, 4000000])
]

rocket_to_earth = (((bodies[2].position[0] - bodies[0].position[0]) ** 2)+((bodies[2].position[1] - bodies[0].position[1]) ** 2)) ** 0.5
rocket_to_moon = (((bodies[2].position[0] - bodies[1].position[0]) ** 2)+((bodies[2].position[1] - bodies[1].position[1]) ** 2)) ** 0.5

#define function to compute period

def compute_period(m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((d ** 3 ) / (G * m_total)) ** 0.5

    return(p)

period = compute_period(bodies[0].mass, bodies[1].mass)

print(period/86400)

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t / period)
    y = distance * math.sin(2 * pi * t / period)

    return x, y

num_coordinates = 5

earth_list = []
moon_list = []

for i in range(0, 5):

    bodies[0].position = planet_position(-COM, period, i)
    bodies[1].position = planet_position(d - COM, period, i)

    earth_list.append(bodies[0].position)
    moon_list.append(bodies[1].position)

x_coordinates_earth = [coord[0] for coord in earth_list]
y_coordinates_earth = [coord[1] for coord in earth_list]

x_coordinates_moon = [coord[0] for coord in moon_list]
y_coordinates_moon = [coord[1] for coord in moon_list]

plt.figure(figsize=(8,8))
plt.plot(x_coordinates_earth, y_coordinates_earth, linestyle = '-', color = 'b')
plt.plot(x_coordinates_moon, y_coordinates_moon, linestyle = '-', color = 'r')

plt.title("Orbit position position")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
