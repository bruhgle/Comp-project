def runge_kutta_position(x0, y0, v_x, v_y, h):
    k1x = h * v_x
    k1y = h * v_y
    
    k2x = h * (v_x + k1x / 2)
    k2y = h * (v_y + k1y / 2)
    
    k3x = h * (v_x + k2x / 2)
    k3y = h * (v_y + k2y / 2)
    
    k4x = h * (v_x + k3x)
    k4y = h * (v_y + k3y)
    
    x = x0 + (k1x + 2*k2x + 2*k3x + k4x) / 6
    y = y0 + (k1y + 2*k2y + 2*k3y + k4y) / 6
    
    return x, y

# Example usage
x0 = 0   # Initial x position
y0 = 0   # Initial y position
v_x = 10 # Example x component of velocity
v_y = 5  # Example y component of velocity
h = 0.1  # Step size (time interval)

x, y = runge_kutta_position(x0, y0, v_x, v_y, h)
print(f"Computed position: x = {x}, y = {y}")