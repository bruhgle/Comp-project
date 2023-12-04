import matplotlib.pyplot as plt

# Create a figure
plt.figure()

# Define circle parameters
center = (2, 2)  # Center coordinates (x, y)
radius = 1.5      # Radius of the circle

# Plot the circle
circle = plt.Circle(center, radius, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=2)
plt.gca().add_patch(circle)

# Set plot properties
plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is equal
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Circle on Matplotlib Plot')

# Show the plot
plt.show()