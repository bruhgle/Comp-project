#import libraries

import numpy as np
import math

#define global variables

m_earth = 5.9742 * 10**24
m_moon = 7.35 * 10**22
G = 6.6726 * 10**-11
d = 3.84*10**8
pi = np.pi

COM = (m_moon * d) / (m_earth + m_moon)
x_earth, y_earth = ( - COM, 0)
x_moon, y_moon = (d - COM, 0)

#compute period

def compute_period(x_1, x_2, m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((abs(x_2 - x_1) **3 ) / (G * m_total)) ** 0.5

    return(p)
    
orbit_period = compute_period(x_earth, x_moon, m_earth, m_moon)

#define x and y circular motion functions

def planet_position(distance, period, t):

    x = distance * math.cos(2 * pi * t / period)
    y = distance * math.sin(2 * pi * t / period)

    return x, y

a = 0.0001

class object:
    def __init__(self, mass, x, y, d_e, d_m, F_earth, F_moon):
        self.mass = mass
        self.x = x
        self.y = y
        self.d_e = d_e
        self.d_m = d_m
        self.F_earth = F_earth
        self.F_moon = F_moon

flyer = rocket(20000)



self.d_e = (((x - x_earth) ** 2)+((y - y_earth) ** 2)) ** 0.5
self.d_m = (((x - x_moon) ** 2)+((y - y_moon) ** 2)) ** 0.5
self.F_earth = G * ((mass * m_earth) / (d_e ** 2))
self.F_moon = G * ((mass * m_moon) / (d_e ** 2))