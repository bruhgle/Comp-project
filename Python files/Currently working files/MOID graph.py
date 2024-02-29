import matplotlib.pyplot as plt
import numpy as np

down = np.load('moids1000down.npy').tolist()
up = np.load('moids1000up.npy').tolist()
upleft = np.load('moids1000upleft.npy').tolist()
upright = np.load('moids1000upright.npy').tolist()
downleft = np.load('moids1000downleft.npy').tolist()
downright = np.load('moids1000downright.npy').tolist()
zup = np.load('moids1000zup.npy').tolist()
zdown = np.load('moids1000zdown.npy').tolist()

temp = np.load('moids1000RightWithLeftAtEnd.npy').tolist()

right = [temp[1], temp[51], temp[101], temp[151], temp[201]]
left = [temp[226], temp[227], temp[228], temp[229], temp[230]]

print(len(temp))

print(right)
print(left)

# Sample data for 5 lines
x = [0, 15, 30, 45, 60]

font = fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': 11}

legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': 9}

# Plotting the lines
plt.plot(x, down, marker='o', label='(0,-1,0)')
plt.plot(x, up, marker='o', label='(0,+1,0)')
plt.plot(x, left, marker='o', label='(-1,0,0)')
plt.plot(x, right, marker='o', label='(+1,0,0)')
plt.plot(x, upleft, marker='o', label='(-√2,√2, 0)')
plt.plot(x, upright, marker='o', label='(√2,√2,0)')
plt.plot(x, downleft, marker='o', label='(-√2,-√2,0)')
plt.plot(x, downright, marker='o', label='(√2,-√2,0)')
plt.plot(x, zup, marker='o', label='(0,0,1)')
plt.plot(x, zdown, marker='o', label='(0,0,-1)')

# Adding labels and title
plt.xlabel('Impactor relative velocity (km/s)', fontdict=font)
plt.ylabel('MOID (10 km)', fontdict=font)
plt.xticks(x)  # Set x-axis ticks

# Adding legend
plt.legend()

# Display the plot
plt.show()
