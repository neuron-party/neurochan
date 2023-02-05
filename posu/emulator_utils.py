import numpy as np
import math


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def calc_fade_in(time, t1, t2, limit=1):
    # goes from 0 to limit
    return limit * (1 / (t2 - t1)) * (time - t1)

def calc_fade_out(time, t1, t2, limit=1):
    # goes from limit to 0
    return limit * (-1 / (t2 - t1)) * (time - t2)

def draw_point(surface, point, a):
    old_color = surface.get_at(point)
    new_r = int(old_color.r * (1 - a) + 255 * a)
    new_g = int(old_color.g * (1 - a) + 255 * a)
    new_b = int(old_color.b * (1 - a) + 255 * a)
    surface.set_at(point, (new_r, new_g, new_b))

def draw_filled_circle(surface, center, radius, a):
    tl_x, tl_y = max(0, center[0] - radius), max(0, center[1] - radius)
    br_x, br_y = min(tl_x + 2 * radius + 1, surface.get_width()), min(tl_y + 2 * radius + 1, surface.get_height())
    for i in range(tl_x, br_x):
        for j in range(tl_y, br_y):
            if dist(center, (i, j)) < radius:
                draw_point(surface, (i, j), a)

def draw_empty_circle(surface, center, radius, a, width=2):
    tl_x, tl_y = max(0, center[0] - radius - width // 2), max(0, center[1] - radius - width // 2)
    br_x, br_y = min(center[0] + radius + width // 2 + 1, surface.get_width()), min(center[1] + radius + width // 2 + 1, surface.get_height())
    for i in range(tl_x, br_x):
        for j in range(tl_y, br_y):
            if radius - width // 2 <= dist(center, (i, j)) < radius + width // 2:
                draw_point(surface, (i, j), a)

def should_draw_point(point, centers, radius):
    np.random.shuffle(centers)
    for center in centers:
        if dist(center, point) <= radius:
            return True
    return False
                
def draw_slider_body(surface, points, radius, a):
    centers = set()
    for t in np.arange(0, 1, 0.05):
        px = int(sum([points[i][0] * (1 - t) ** (len(points) - i - 1) * t ** (i) * math.comb(len(points) - 1, i) for i in range(len(points))]))
        py = int(sum([points[i][1] * (1 - t) ** (len(points) - i - 1) * t ** (i) * math.comb(len(points) - 1, i) for i in range(len(points))]))
        centers.add((px, py))
    centers = list(centers)
        
    l = max(0, min(centers, key=lambda x: x[0])[0] - radius)
    r = min(max(centers, key=lambda x: x[0])[0] + radius + 1, surface.get_width())
    b = max(0, min(centers, key=lambda x: x[1])[1] - radius)
    t = min(max(centers, key=lambda x: x[1])[1] + radius + 1, surface.get_height())
    
    for i in range(l, r):
        for j in range(b, t):
            if should_draw_point((i, j), centers, radius):
                draw_point(surface, (i, j), a)

def draw_cursor(surface, center, radius):
    tl_x, tl_y = max(0, center[0] - radius), max(0, center[1] - radius)
    br_x, br_y = min(tl_x + 2 * radius + 1, surface.get_width()), min(tl_y + 2 * radius + 1, surface.get_height())
    for i in range(tl_x, br_x):
        for j in range(tl_y, br_y):
            if dist(center, (i, j)) < radius:
                surface.set_at((i, j), (255, 255, 0))