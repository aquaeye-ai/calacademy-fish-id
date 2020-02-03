#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Created on Tue Sep 26 09:41:15 2017
@author: Harald Ruda, John Cast
"""

#import numpy as np

from math import pi, sqrt, atan2, degrees

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# boundary_utils.py
# 
# routines that help process boundaries, usually stored with 'start' and 'end' points
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define constants for the possible fish boundaries
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FISH_BOUNDARY = 'fish-boundary'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Some other useful constants, ie dict tags for lines and intersections
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

TYPE = 'label'

BGN = 'start'
END = 'end'

BGN_CLOSE_INT = 'start-closest-intersection'
END_CLOSE_INT = 'end-closest-intersection'
CLOSE_INT     = '-closest-intersection'

LEFT = 'left'
RGHT = 'right'
NAME = 'name'
THICK = 'thickness'
POINT = 'point'
LINES = 'lines'
CLOSE_LINES = 'closest-lines'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# JSON functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def json_boundaries(json_lines):
    """
    Copy the JSON lines and set the LEFT and RGHT tags.
    """
    lines = [json_line for json_line in json_lines]
    
    for line in lines:
        line[LEFT] = None
        line[RGHT] = None
    
    return lines

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# line functions
# - - - - - - - - 
# a line is a dict
# it should contain 'start' and 'end' (x,y) tuples
# other items 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_line(p1, p2, label=None, thickness=1, name=None):
    """
    Create a new line, with the default tags required by many functions.
    """
    newline = {}
    newline[BGN] = p1
    newline[END] = p2
    newline[TYPE] = label
    newline[NAME] = name
    newline[THICK] = thickness
    newline[RGHT] = None
    newline[LEFT] = None
    
    return newline


def line_name(line):
    """
    For printing, return the most useful name available.
    """
    basic = ' from '
    basic += (str(line[BGN_CLOSE_INT]) if BGN_CLOSE_INT in line else str(line[BGN]))
    basic += ' to '
    basic += (str(line[END_CLOSE_INT]) if END_CLOSE_INT in line else str(line[END]))
    
    if NAME in line and line[NAME] is not None:
        return str(line[NAME]) + basic
    
    if TYPE in line and line[TYPE] is not None:
        return str(line[TYPE]) + basic
    
    return 'line' + basic


def print_lines(lines):
    """
    Print out information about lines. 
    """
    print(len(lines), "lines:")
    
    for line in lines:
        print("  ", line[TYPE], "from", line[BGN_CLOSE_INT], "to", line[END_CLOSE_INT], line_name(line))


def check_lines(lines, intersections):
    """
    Chech the structure of lines to make sure all required tags are present.
    # line should have left/right also start/end and start-end/end-end (??)
    """
    required_tags = [ BGN_CLOSE_INT, END_CLOSE_INT ]
    
    for line in lines:
        for tag in required_tags:
            if tag not in line:
                print("no " + tag + " in the line")
            elif line[tag] not in intersections:
                print(tag + " not in intersections")


def other_end_of(line, intersection):
    """
    Given a line and one intersection, return the closest intersection on the opposite end.
    """
    other = (END if line[BGN_CLOSE_INT] == intersection else BGN)
    
    return line[other + CLOSE_INT]


def point_separation(p1, p2):
    """
    The distance between two points.
    """
    return sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 )


def closest_endpoint_separation(line1, line2):
    """
    Try all four combinations of end-points.
    """
    end_choices = [ BGN, END ]
    closest_separation = None
    
    for end1 in end_choices:
        for end2 in end_choices:
            separation = point_separation(line1[end1], line2[end2])
            
            if (closest_separation is None) or (separation < closest_separation):
                closest_separation = separation
                closest_end1 = end1
                closest_end2 = end2
    
    return closest_separation, closest_end1, closest_end2


def line_slope(line):
    """
    Return the slope of the line, or None if vertical.
    """
    dx = line[END][0] - line[BGN][0]
    dy = line[END][1] - line[BGN][1]
    
    return None if dx == 0 else dy / float(dx)


def line_orientation(line):
    """
    Return orientation [0,180) in DEGREES
    """
    dx = line[END][0] - line[BGN][0]
    dy = line[END][1] - line[BGN][1]
    
    orientation = atan2(dy, dx)
    while (orientation < 0.0):
        orientation += pi
    
    return degrees(orientation)


def nearly_parallel(line1, line2, parallel_limit=10):
    """
    Are the lines nearly parallel? Within the parallel_limit, in DEGREES?
    """
    return abs(line_orientation(line1) - line_orientation(line2)) < parallel_limit 


def non_parallel_line_nearby(lines, point, line1, line2):
    """
    Need to improve this.
    """
    found = False
    
    for line in lines:
        
        if line == line1:
            continue
        if line == line2:
            continue
        
        # is the point anywhere near the line??
        # find the closest point on the line 
        # i.e. projection of point onto line
        
        if point_separation(point, line[BGN]) < 10:
            found = True
        
        if point_separation(point, line[END]) < 10:
            found = True
    
    return found


def generate_joins_and_endcuts(lines):
    """
    In cases where parallel or nearly parallel lines do not come together, 
    but end close to each other. It may be useful (or necessary) to introduce 
    a fake (illusory!) line to join the two ends.
    
    No need to do anything if a crossing line is present
    
    Also may join two lines with an intersection.
    """
    add_lines = []
    intersections = {}
    
    for n, line1 in enumerate(lines):
        for line2 in lines[n+1:]:
            
            separation, end1, end2 = closest_endpoint_separation(line1, line2)
            x = int( (line1[end1][0] + line2[end2][0]) / 2 )
            y = int( (line1[end1][1] + line2[end2][1]) / 2 )
            
            if separation <= 1 and nearly_parallel(line1, line2): # just join
                key = (x,y)
                newinter = create_intersection(key, line1, line2)
                #print("Creating an intersection!", key, "joining", line_name(line1), "and", line_name(line2))
                intersections[key] = newinter
            
            elif separation < 10 and non_parallel_line_nearby(lines, (x,y), line1, line2):
                # do nothing
                pass
            
            elif separation < 10 and nearly_parallel(line1, line2):  # make end-cut
                newline = create_line(line1[end1], line2[end2], label=line1[TYPE], name='illusory')
                #print("Creating a new line!", "from", line1[end1], "of", line_name(line1), "and", line2[end2], "of", line_name(line2))
                add_lines.append(newline)
    
    for line in add_lines:
        lines.append(line)
    
    return intersections


def split_lines(intersections, lines):
    """
    """
    for line in lines:
        line['split-points'] = []
    
    for point in intersections:
        for line in intersections[point][LINES]:
            if line not in intersections[point][CLOSE_LINES]:
                line['split-points'].append(point)
    
    new_lines = []
    for line in lines:
        if not line['split-points']:
            new_lines.append(line)
        else:
            # process points in order, from start to end (distance from start)
            # first figure out the order
            #p0 = line['start-closest-intersection']
            #p1 = line['end-closest-intersection']
            #dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            #splits = { ((p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2) : p for p in line['split-points'] }
            splits = { t_value(p, line) : p for  p in line['split-points'] }
            t_params = list(splits.keys())
            t_params.sort()
            
            # now do one at a time
            original = line
            remove = []
            #print("t-values:", t_params, "for points", line['split-points'], "of line", line['label'], "from", line['start-closest-intersection'], "to", line['end-closest-intersection'])
            
            for t_param in t_params:
                if (t_param < 0) or (1 < t_param):
                    continue
                point = splits[t_param]
                
                remove.append(original)
                line1 = original.copy()
                line2 = original.copy()
                line1[END] = point
                line1[END_CLOSE_INT] = point
                line2[BGN] = point
                line2[BGN_CLOSE_INT] = point
                
                new_lines.append(line1)
                #new_lines.append(line2)
                
                intersections[point][LINES].append(line1)
                intersections[point][LINES].append(line2)
                intersections[point][CLOSE_LINES].append(line1)
                intersections[point][CLOSE_LINES].append(line2)
                
                for unused in remove:
                    if unused in intersections[point][LINES]:
                        intersections[point][LINES].remove(unused)
                
                other = original[BGN_CLOSE_INT]
                if original in intersections[other][LINES]:   # shouldn't it be?
                    intersections[other][LINES].remove(original)
                if original in intersections[other][CLOSE_LINES]:   # shouldn't it be?
                    intersections[other][CLOSE_LINES].remove(original)
                intersections[other][LINES].append(line1)
                intersections[other][CLOSE_LINES].append(line1)
                
                other = original[END_CLOSE_INT]
                if original in intersections[other][LINES]:   # shouldn't it be?
                    intersections[other][LINES].remove(original)
                if original in intersections[other][CLOSE_LINES]:   # shouldn't it be?
                    intersections[other][CLOSE_LINES].remove(original)
                intersections[other][LINES].append(line2)
                intersections[other][CLOSE_LINES].append(line2)
                
                original = line2
                
            new_lines.append(original)
        
    return new_lines


def t_value(point, line):
    """
    """
    p0 = line[BGN_CLOSE_INT]
    p1 = line[END_CLOSE_INT]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ddx, ddy = point[0] - p0[0], point[1] - p0[1]
    if dx == 0 and dy == 0:
        t = 0
    else:
        t = sqrt(ddx * ddx + ddy * ddy) / sqrt(dx * dx + dy * dy)
    if (dx * ddx < 0) or (dy * ddy < 0):
        t = - t
    return t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# intersection functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# an intersection is a dict
# the 'key' should be an (x,y) tuple
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_intersection(point, line1, line2):
    """
    Create a new intersection, with the tags required. 
    """
    intersection = {}
    intersection[POINT] = point
    intersection[LINES] = [ line1, line2 ]
    intersection[CLOSE_LINES] = []
    
    return intersection


def print_intersections(intersections):
    """
    """
    print(len(intersections), "intersections:")
    
    for inter in intersections:
        
        if not intersections[inter][CLOSE_LINES]:
            print("  ", inter, "no closest lines")
        else:
            print("  ", inter, "closest-lines:") #, intersections[inter])
        
        for line in intersections[inter][CLOSE_LINES]:
            #print("    ", line['label'])
            #print("    ", line['label'], "from", line['start'], "to", line['end'])
            print("    ", line[TYPE], "from", line[BGN_CLOSE_INT], "to", line[END_CLOSE_INT], line_name(line))
        
        others = False
        for line in intersections[inter][LINES]:
            if line not in intersections[inter][CLOSE_LINES]:
                others = True
        if not others:
            continue
        print("->", inter, "OTHER-lines:") #, intersections[inter])
        for line in intersections[inter][LINES]:
            if line not in intersections[inter][CLOSE_LINES]:
                print("    ", line[TYPE], "from", line[BGN_CLOSE_INT], "to", line[END_CLOSE_INT], line_name(line))


def print_intersections_full(intersections):
    """
    """
    print("intersections:")
    for inter in intersections:
        
        if not intersections[inter][CLOSE_LINES]:
            print("  ", inter, "no closest lines")
        else:
            print("  ", inter, "closest-lines:")
        for line in intersections[inter][CLOSE_LINES]:
            print("    ", line[TYPE], "from", line[BGN_CLOSE_INT], "to", line[END_CLOSE_INT])
        
        if not intersections[inter][LINES]:
            print("  ", inter, "no other lines")
        else:
            print("  ", inter, "other-lines:")
        for line in intersections[inter][LINES]:
            print("    ", line[TYPE], "from", line[BGN_CLOSE_INT], "to", line[END_CLOSE_INT])


def check_intersections(lines, intersections):
    """
    """
    remove = []
    for point in intersections:
        if len(intersections[point][CLOSE_LINES]) < 2:
            print("NOT ENOUGH LINES at intersection", point)
            
            closest = None
            closest_distance = None
            
            for other in intersections:
                if other == point:
                    continue
                
                dx = other[0] - point[0]
                dy = other[1] - point[1]
                d2 = dx * dx + dy * dy
                
                if closest == None or d2 < closest_distance:
                    closest = other
                    closest_distance = d2
            
            print("  The point is", point, "keys", intersections[point].keys())
            print("  closest", len(intersections[point][CLOSE_LINES]), [line_name(line) for line in intersections[point][CLOSE_LINES]])
            print("  + lines", len(intersections[point][LINES]), [line_name(line) for line in intersections[point][LINES] if line not in intersections[point][CLOSE_LINES] ])
            
            print("  The closest point is", closest, "keys", intersections[closest].keys())
            print("  closest", len(intersections[closest][CLOSE_LINES]), [line_name(line) for line in intersections[closest][CLOSE_LINES]])
            print("  + lines", len(intersections[closest][LINES]), [line_name(line) for line in intersections[closest][LINES] if line not in intersections[closest][CLOSE_LINES] ])
            
            for line in intersections[point][LINES]:
                if line not in intersections[closest][LINES]:
                    intersections[closest][LINES].append(line)
                    if (line[BGN_CLOSE_INT] == point):
                        line[BGN_CLOSE_INT] = closest
                    if (line[END_CLOSE_INT] == point):
                        line[END_CLOSE_INT] = closest
            
            for line in intersections[point][CLOSE_LINES]:
                if line not in intersections[closest][CLOSE_LINES]:
                    intersections[closest][CLOSE_LINES].append(line)
                    if (line[BGN_CLOSE_INT] == point):
                        line[BGN_CLOSE_INT] = closest
                    if (line[END_CLOSE_INT] == point):
                        line[END_CLOSE_INT] = closest
            
            #intersections.remove(point)
            remove.append(point)
    
    for point in remove:
        del intersections[point]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def generate_intersections(lines, intersections):
    """
    Generate all the intersections among the lines, 
    and keep track of which lines appear at which intersections
    also find the closest intersection to each endpoint of each line
    """
    
    for line in lines:
        #print()
        #print("slope =", line_slope(line), "perp =", -1/line_slope(line))
        #print("orientation =", int(line_orientation(line)))
        find_closest_intersection(lines, intersections, line, BGN, END)
        find_closest_intersection(lines, intersections, line, END, BGN)
    
    return intersections

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def find_closest_intersection(lines, intersections, line1, near, far):
    """
    """
    avoid = None
    closest = None
    newintersections = {}
    #newintersections = intersections
    
    if far + CLOSE_INT in line1:
        avoid = line1[far + CLOSE_INT]
    
    for line2 in lines:
        if line1 == line2:
            continue
        # check and see if directions are too close?
        intersection = compute_intersection(line1, line2, near, far)
        
        if intersection is None:
            continue
        
        # print("  ", tuple(line1[near]), intersection, line2['label'])
        key = tuple(intersection[0:2])
        
        if key == avoid:
            #print("AVOIDING", key, (line1['name'] if 'name' in line1 else ''), (line2['name'] if 'name' in line2 else ''))
            continue
        
        if key not in intersections:
            if key not in newintersections:
                newintersections[key] = {}
                newintersections[key][POINT] = key
                newintersections[key][LINES] = [ line1, line2 ]
                newintersections[key][CLOSE_LINES] = []
            
            if line1 not in newintersections[key][LINES]:
                newintersections[key][LINES].append(line1)
            
            if line2 not in newintersections[key][LINES]:
                newintersections[key][LINES].append(line2)
            
#            intersections[key] = {}
#            intersections[key]['point'] = key
#            intersections[key]['lines'] = [ line1, line2 ]
#            intersections[key]['closest-lines'] = []
        else:
            #print("key in intersections, should do something", key)
            if line1 not in intersections[key][LINES]:
                intersections[key][LINES].append(line1)
            if line2 not in intersections[key][LINES]:
                intersections[key][LINES].append(line2)
        
        if (closest is None) or (abs(closest[4]) > abs(intersection[4])):
            closest = intersection
    
    key = tuple(closest[0:2])
    line1[near + CLOSE_INT] = key
    #print("*", near + '-closest-intersection', "is", key, "for", (line1['name'] if 'name' in line1 else ''))
    
    if key not in intersections:
        intersections[key] = newintersections[key]
        #print("adding intersection", key, [line['name'] if 'name' in line else '' for line in intersections[key]['lines']])
    
    intersections[key][CLOSE_LINES].append(line1)
    #print("      closest lines", key, [line['name'] if 'name' in line else '' for line in intersections[key]['closest-lines']])
    #print("        other lines", key, [line['name'] if 'name' in line else '' for line in intersections[key]['lines']])
    #print("line", line1['label'], "point", line1[near], "is closest to intersection point", key)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_intersection(line1, line2, near, far):
    """
    """
    # use formula derived
    x1 = line1[near][0]
    dx1 = line1[far][0] - x1
    x2 = line2[near][0]
    dx2 = line2[far][0] - x2
    y1 = line1[near][1]
    dy1 = line1[far][1] - y1
    y2 = line2[near][1]
    dy2 = line2[far][1] - y2
    
    den = float(dx1 * dy2 - dy1 * dx2)
    if (abs(den) < 0.001):      # parallel lines
        return None
    
    t1 = ((y1 - y2) * dx2 - (x1 - x2) * dy2) / den
    x0 = int(round(x1 + t1 * dx1))
    y0 = int(round(y1 + t1 * dy1))
    t0 = t1 * sqrt(dx1 * dx1 + dy1 * dy1)
    # print(x0, y0, t0, " <> ", t1, x1, y1)
    # x0, y0 is the intersection, t0 is the distance in pixels from 'near'
    
    # return the measure between line1[near] and the closest end-point of line2
    d_near = sqrt((x1 - line2[near][0]) ** 2 + (y1 - line2[near][1]) ** 2)
    d_far  = sqrt((x1 - line2[far][0]) ** 2  + (y1 - line2[far][1]) ** 2)
    d0 = (d_near if d_near < d_far else d_far)
    
    return (x0, y0, t0, d0, abs(t0) + abs(d0))


def combine_intersections(intersections, lines):
    """
    """
    combined = []
    replace = {}
    new_intersections = {}
    
    for inter in intersections:
        if inter in combined or not intersections[inter][CLOSE_LINES]:
            continue
        for other in intersections:
            if inter == other:
                continue
            #if other in combined or not intersections[other]['closest-lines']:
            if other in combined:
                continue
            dx = inter[0] - other[0]
            dy = inter[1] - other[1]
            d2 = dx * dx + dy * dy
            if d2 < 20:
                #print("combined:", other, "replaced by", inter, "distance-squared", d2)
                combined.append(other)
                replace[other] = inter
                
                for line in intersections[other][CLOSE_LINES]:
                    if line not in intersections[inter][CLOSE_LINES]:
                        intersections[inter][CLOSE_LINES].append(line)
                
                for line in intersections[other][LINES]:
                    if line not in intersections[inter][LINES]:
                        intersections[inter][LINES].append(line)
                        
        new_intersections[inter] = intersections[inter]
    
    for line in lines:
        if line[BGN_CLOSE_INT] in combined:
            line[BGN_CLOSE_INT] = replace[line[BGN_CLOSE_INT]]
        if line[END_CLOSE_INT] in combined:
            line[END_CLOSE_INT] = replace[line[END_CLOSE_INT]]
    
    return new_intersections



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# functions that combine lines and intersections
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def determine_boundaries_and_intersections(lines):
    """
    """
    intersections = generate_joins_and_endcuts(lines)
    intersections = generate_intersections(lines, intersections)
    #print_intersections(intersections)
    
    intersections = combine_intersections(intersections, lines)
    #print_intersections(intersections)
    lines = split_lines(intersections, lines)
    #print_intersections(intersections)
    
    check_lines(lines, intersections)           # two intersections for each line
    check_intersections(lines, intersections)   #  two or more lines for each intersections
    #print_lines(lines)
    
    return lines, intersections

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# slope functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def find_slopes(lines, intersections):
    """
    # choose any line, follow around slope on 
    # then follow around slope, starting with the other end (left is now right)
    # then do the same thing for the next line
    """
    slopes = []
    
    #for line in lines:
    for num, line in enumerate(lines):
        #print("Line", num, "A")
        slopes.append(trace_slope(line[BGN_CLOSE_INT], line, RGHT, intersections))
        #print("Line", num, "B")
        slopes.append(trace_slope(line[END_CLOSE_INT], line, RGHT, intersections))
        #print("Line", num, "C")
    
    return [slope for slope in slopes if slope is not None]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def trace_slope(starting_point, starting_line, direction, intersections):
    """
    """
    if starting_line[actual_direction(starting_point, starting_line, direction)]:
        return None
    
    slope = { 'points':[], 'lines':[] }
    current_point, current_line = starting_point, starting_line
    next_point, next_line = None, None
    
    while next_point != starting_point:
        next_point, next_line = slope_next_line(current_point, current_line, direction, intersections)
        if next_point in slope['points'] or next_line in slope['lines']:
            print("slope tracing error -- should not happen!!")
            break
        slope['points'].append(next_point)
        slope['lines'].append(next_line)
        current_point, current_line = next_point, next_line
    
    for point, line in zip(slope['points'], slope['lines']):
        if line[actual_direction(point, line, direction)] is not None:
            print("line direction already set -- should not happen")
        line[actual_direction(point, line, direction)] = slope
    
    # reverse order to make right-handed
    
#    if direction is 'left':
#        slope['points'].reverse()
#        slope['lines'].reverse()
    
    perimeter = True
    for line in slope[LINES]:
        if line[TYPE] != OUTSIDE_EDGE and line[TYPE] != TREE_BOUNDARY:
            perimeter = False
    slope['perimeter'] = perimeter
    
    return slope

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def actual_direction(point, line, direction):
    """
    Because the line may be defined in either direction, need to figure this out.
    """
    if point == line[BGN_CLOSE_INT]:
        return direction
    else:
        return (LEFT if direction == RGHT else RGHT)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def slope_next_line(point, mainline, direction, intersections):
    """
    'right' means least counter-clockwise
    'left' means least clock-wise
    y axis is down in image, needs to be negative for true CCW angle from atan2()
    """
    # the angle of the main line will be the zero referent
    center = other_end_of(mainline, point)
    main_angle = atan2(center[1] - point[1], point[0] - center[0])
    angles = {}
    
    if center not in intersections:
        print("WEIRD", center, "is not in intersections")
        return None, None
    
    for line in intersections[center][CLOSE_LINES]:
        if line == mainline:
            continue
        outside = other_end_of(line, center)
        angle = atan2(center[1] - outside[1], outside[0] - center[0])
        angle -= main_angle
        while angle < 0:
            angle += 2 * pi
        while angle >= 2 * pi:
            angle -= 2 * pi
        angles[angle] = line
    
    if len(angles) == 0:
        #print("problem: there is no next line", point, mainline['label'], len(intersections[center]['closest-lines']))
        print("problem: there is no next line", point)
    
    #print(point, direction)
    
    if direction == 'right':
        angle = min(angles.keys())
    else:
        angle = max(angles.keys())
    
    return center, angles[angle]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def label_perimeters(slopes, boundaries, intersections):
    """
    Look at each slope and set the 'perimeter' tag appropriately.
    """
    # sort of done in trace_slope(), depends on OUTSIDE_EDGE labels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
