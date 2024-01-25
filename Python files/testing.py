#import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import ephem

#define global variables

G = 6.6726e-11
pi = np.pi

class body:
    def __init__(self, name, mass, position, radius):
        self.name = name
        self.position = position
        self.mass = mass
        self.radius = radius
        self.past_positions = []
        self.clearance = []
        self.separation = []

bodies = [
    body("sun", 1.9884e30, [0, 0], 0), 
    body("Earth", 5.9722e24, [0, 0], 1.49598e11), #mass errors found at [https://web.archive.org/web/20161224174302/http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf]
    body("Mars", 6.4169e23, [0, 0], 2.27956e11), 
    body("Venus", 4.8673e24, [0, 0], 1.08210e11),
    body("asteroid1", 26.99e9, [1.64e11, 0], 0),
    body("asteroid2", 26.99e9, [1.64e11, 0], 0),
    body("asteroid3", 26.99e9, [1.64e11, 0], 0),
    body("asteroid4", 26.99e9, [1.64e11, 0], 0),
    body("asteroid5", 26.99e9, [1.64e11, 0], 0),
    body("Mercury", 3.3010e23, [0, 0], 5.7909e10), #radius values not accurate
    body("Jupiter", 1.8985e27, [0, 0], 7.78479e11),
    body("Saturn", 5.6846e26, [0, 0], 1.432041e12),
    body("Uranus", 8.6813e25, [0, 0], 2.867043e12),
    body("Neptune", 1.0243e26, [0, 0], 4.513953e12),
]

def compute_solar_distance(index, date):

    if index == 1:

        sun = ephem.Sun()

        # Set the date for the calculation
        sun.compute(date)

        # Get the distance from Earth to the Sun in astronomical units (AU)
        distance_au = sun.earth_distance

        # Convert AU to kilometers (1 AU = 149,597,870.7 km)
        distance_km = distance_au * 149597870.7

        return distance_km

    else:

        planet_name = bodies[index].name

        observer = ephem.Observer()
        observer.lat = '0'  
        observer.lon = '0'
    
        planet = getattr(ephem, planet_name)()

        # Set the date for the calculation
        observer.date = date
        planet.compute(observer)

        # Get the distance from the planet to the Sun in astronomical units (AU)
        distance_au = planet.sun_distance

        # Convert AU to kilometers (1 AU = 149,597,870.7 km)
        distance_km = distance_au * 149597870.7

    return distance_km

print(compute_solar_distance(1,"2024/01/25"))