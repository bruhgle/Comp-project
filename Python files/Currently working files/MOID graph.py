import matplotlib.pyplot as plt
import numpy as np

down = np.load('moids1000down.npy').tolist()

# Sample data for 5 lines
x = [0, 15, 30, 45, 60]  # Kilometers per second
y1 = [10, 20, 30, 40, 50]
y2 = [15, 25, 35, 45, 55]
y3 = [20, 30, 40, 50, 60]
y4 = [25, 35, 45, 55, 65]
y5 = [30, 40, 50, 60, 70]

# Plotting the lines
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.plot(x, y3, label='Line 3')
plt.plot(x, y4, label='Line 4')
plt.plot(x, y5, label='Line 5')
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.plot(x, y3, label='Line 3')
plt.plot(x, y4, label='Line 4')
plt.plot(x, y5, label='Line 5')

# Adding labels and title
plt.xlabel('Impactor relative velocity (km/s)')
plt.ylabel('MOID (10 km)')
plt.xticks(x)  # Set x-axis ticks

# Adding legend
plt.legend()

# Display the plot
plt.show()
