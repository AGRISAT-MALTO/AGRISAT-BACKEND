import math
from shapely.geometry import Polygon


def poly_area(coords):
    centroid = {"lat": sum(p[0] for p in coords)/len(coords), "lng": sum(p[1] for p in coords)/len(coords)}
    lat_factor = 111_320
    lng_factor = 111_320 * math.cos((centroid['lat'] * math.pi) / 180)
    points = [((p[1]-centroid['lng']) * lng_factor, (p[0]-centroid['lat']) * lat_factor) for p in coords]
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return abs(area) / 2


def perimeter(coords):
    centroid = {"lat": sum(p[0] for p in coords)/len(coords), "lng": sum(p[1] for p in coords)/len(coords)}
    lat_factor = 111_320
    lng_factor = 111_320 * math.cos((centroid['lat'] * math.pi) / 180)
    points = [((p[1]-centroid['lng']) * lng_factor, (p[0]-centroid['lat']) * lat_factor) for p in coords]
    peri = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        peri += math.hypot(points[j][0]-points[i][0], points[j][1]-points[i][1])
    return peri


def check(coords):
    area = poly_area(coords)
    per = perimeter(coords)
    compact = (4*math.pi*area)/(per*per)
    poly = Polygon([(p[1], p[0]) for p in coords])
    hull_ratio = poly.area / poly.convex_hull.area if poly.convex_hull.area else 0
    lat_min = min(p[0] for p in coords); lat_max=max(p[0] for p in coords); lng_min=min(p[1] for p in coords); lng_max=max(p[1] for p in coords)
    width = (lng_max-lng_min)*111320*max(abs(math.cos(((lat_min+lat_max)/2)*math.pi/180)),0.1)
    height = (lat_max-lat_min)*110574
    aspect = max(width,height)/min(width,height)
    print('area', area, 'compact', compact, 'hull_ratio', hull_ratio, 'aspect', aspect)
    return (compact >= 0.18) and (hull_ratio >= 0.55) and (aspect <= 12)


good = [(0,0),(0,1),(1,1),(1,0)]
bad = [(0,0),(0.2,0),(0.4,0.8),(0.8,0.1),(1,0),(0.9,1),(0.5,1.5),(0.2,1),(0,0.9)]
print('good', check(good))
print('bad', check(bad))
