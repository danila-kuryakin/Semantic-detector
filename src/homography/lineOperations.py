# Searches for linear equation variables
# y = kx + b
def find_line_equation(line):
    x1, y1, x2, y2 = line
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return {'k': k, 'b': b}


# Looking for the point where the lines intersect
def find_intersection_point(line_a, line_b):
    x = int((line_b['b'] - line_a['b'])/(line_a['k'] - line_b['k']))
    y = int(line_b['k'] * x + line_b['b'])
    return x, y


def move_line(line, x, y):
    x1, y1, x2, y2 = line
    return [x1 + x, y1 + y, x2 + x, y2 + y]


# perspective transform from point
def warp_point(coord, warp_matrix):
    x = coord[0]
    y = coord[1]
    d = warp_matrix[2, 0] * x + warp_matrix[2, 1] * y + warp_matrix[2, 2]

    return (
        int((warp_matrix[0, 0] * x + warp_matrix[0, 1] * y + warp_matrix[0, 2]) / d), # x
        int((warp_matrix[1, 0] * x + warp_matrix[1, 1] * y + warp_matrix[1, 2]) / d), # y
    )