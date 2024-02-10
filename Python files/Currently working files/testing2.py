n=5

for i in range(9,9+5*n):

    if i >= 9 and i < 9+n:

        print(i, "impulse 1")
    
    if i >= 9+n and i < 9+2*n:

        print(i, "impulse 2")

    if i >= 9+2*n and i < 9+3*n:

        print(i, "impulse 3")