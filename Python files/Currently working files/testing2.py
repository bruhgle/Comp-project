import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import ephem
from matplotlib.patches import Ellipse
from astroquery.jplhorizons import Horizons

start_date_str = "2024-02-16 00:00:00"
sim_time = 1.627312e8

def true_position(begin_date, t):
    # Convert input date string to datetime object
    start_date = datetime.strptime(begin_date.split()[0], '%Y-%m-%d').date() + timedelta(seconds = t)

    # Calculate end date as start date plus one day
    end_date = start_date + timedelta(days=1)

    # Convert end date back to string in the required format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Create a Horizons query object for Apophis to get vector table
    obj = Horizons(id='99942', location='500@0', epochs={'start': start_date_str, 'stop': end_date_str, 'step': '1d'})

    # Perform the query to get the vector table
    vec = obj.vectors()

    # Extracting heliocentric Cartesian coordinates (X, Y, Z) from the vector table
    x_helio = vec['x'][0] * 149597870700
    y_helio = vec['y'][0] * 149597870700
    z_helio = vec['z'][0] * 149597870700

    # Return the heliocentric coordinates
    return [x_helio, y_helio, z_helio]

real_positions = []

for i in range(100):

    real_pos = true_position(start_date_str, i * int(sim_time/100))
    real_positions.append(real_pos)

print(real_positions)