import numpy as np

goal_state = np.array( [ 
                        [0, 1, 2],
                        [3, 4, 5], 
                        [6, 7, 8],
                ])

def manhattan_distance(x,y) : 
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def adjacents(x, y) : 
    for m, n in {(0, 1), (0, -1), (1, 0), (-1, 0)} : 
        x_, y_ = x + m , y + n 
        if 0 <= x_ <= 2 and 0 <= y_ <= 2 : 
            yield x_, y_

def estimated_cost(a) : 
    sum_ = 0 
    for i in range(8) : 
        x, y = np.where(a.value == i) 
        x_, y_ = np.where(goal_state == i)
        sum_ += manhattan_distance((x,y) , (x_,y_))
    return sum_

