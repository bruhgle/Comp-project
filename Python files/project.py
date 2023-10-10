#import libraries

import numpy as np
import math

#Define global variables

m_earth = 5.9742 * 10**24
m_moon = 7.35 * 10**22
G = 6.6726 * 10**-11
d = 3.84*10**8
pi = np.pi
m_total = m_earth + m_moon

COM = (m_moon * d) / m_total
xi_earth = - COM
xi_moon = d - COM

def compute_period(x_1, x_2, m_1, m_2):

    m_total = m_1 + m_2

    p = 2 * pi * ((abs(x_2 - x_1) **3 ) / (G * m_total)) ** 0.5

    return(p)
    
period = compute_period(xi_earth, xi_moon, m_earth, m_moon)

print(period/86400)

def x_movement(t):
   
    for i in range(0, t):
        
        x = np.cos((pi * t)/period)

    return(x)

print(x_movement(5000000))