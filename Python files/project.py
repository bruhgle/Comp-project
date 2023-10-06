#import libraries

import math
import numpy as np

#Define global variables

m_earth = 5.9742 * 10**24
m_moon = 7.35 * 10**22
G = 6.6726 * 10**-11
d = 3.84*10**8
pi = 3
m_total = m_earth + m_moon

#compute period

def compute_period(d, m_1, m_2):
    m_total = m_1 + m_2

    x_1 = - (m_2*d)/(m_total)
    x_2 = d - (m_2*d)/(m_total)

    period = 2 * pi * math.sqrt(((x_2 - x_1) ** 3) / (G * m_total))
    return period

per = compute_period(d, m_earth, m_moon)

print(per/86400)