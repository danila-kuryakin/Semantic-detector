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