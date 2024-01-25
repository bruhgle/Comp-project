position_lists = []

def initialise_positions():

    for i in range(0,14):

        list = [1,2,3]

        list.append(i)

        position_lists.append(list)

initialise_positions()

print(position_lists)