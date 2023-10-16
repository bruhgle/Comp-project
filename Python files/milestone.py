#import libraries

import numpy as np
import math

#define initial varibles

m_earth = 5.9742 * 10**24
m_moon = 7.35 * 10**22
G = 6.6726 * 10**-11
d = 3.84*10**8
pi = np.pi

#define COM for earth and moon

COM = (m_moon * d) / (m_earth + m_moon)

#define general body class

class body:
    def __init__(self, mass, x, y, d_e, d_m, F_e, F_m):
        self.x = x
        self.y = y
        self.mass = mass
        self.d_e = d_e
        self.d_m = d_m
        self.F_e = F_e
        self.F_m = F_m

#define object for each body using above class

earth = body(m_earth, - COM, 0, 0, d, 0, 0)
moon = body(m_moon, d - COM, 0, d, 0, 0, 0)
rocket = body(30000, earth.x, 4000000, 0, 0, 0, 0)

rocket_to_earth = (((rocket.x - earth.x) ** 2)+((rocket.y - earth.y) ** 2)) ** 0.5
rocket_to_moon = (((rocket.x - moon.x) ** 2)+((rocket.y - moon.y) ** 2)) ** 0.5
F_earth = G * ((rocket.mass * earth.mass) / (rocket_to_earth ** 2))
F_moon = G * ((rocket.mass * moon.mass) / (rocket_to_moon ** 2))

#reinitialise rocket attributes to correct values

rocket = body(30000, earth.x, 4000000, rocket_to_earth, rocket_to_moon, F_earth, F_moon)

#define function to compute period

def compute_period(x_1, x_2, m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((abs(x_2 - x_1) **3 ) / (G * m_total)) ** 0.5

    return(p)

period = compute_period(earth.x, moon.x, earth.mass, moon.mass)

#define function to compute planet position with SHM

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t / period)
    y = distance * math.sin(2 * pi * t / period)

    return x, y

