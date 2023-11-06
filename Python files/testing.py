
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a plot
plt.plot(x, y, label='Data')

# Customize legend font properties
legend_font = {'family': 'sans-serif', 'size': 12, 'weight': 'bold', 'color': 'purple'}

# Add legend with custom font properties
plt.legend(prop=legend_font)

# Show the plot
plt.show()