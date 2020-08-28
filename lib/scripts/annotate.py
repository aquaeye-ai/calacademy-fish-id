#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# annotate.py
# simple interactive annotation tool
# specify image to draw lines on top of

# specify image (if any), json file (esp if no image)
# output JSON file

import sys
import cv2
import json
import os.path

from enum import Enum

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## Globals
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class MODE_TYPES(Enum):
    SINGLE_POINT    = 0
    MULTI_POINT     = 1
    CONTINUOUS      = 2

MODE_TYPES_MAP = {}
MODE_TYPES_MAP[MODE_TYPES.SINGLE_POINT] = 'single-point'
MODE_TYPES_MAP[MODE_TYPES.MULTI_POINT]  = 'multi-point'
MODE_TYPES_MAP[MODE_TYPES.CONTINUOUS]  = 'continuous'

line_types_to_colors = {
                        'fish-boundary': (255, 255, 0)
                       }

generic_annotator = line_types_to_colors.keys()

line_types = generic_annotator

mTypeIndex = 0
cTypeIndex = 0
cThickness = 2
pressed = False
ix, iy = -1, -1

lines = []


## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## Functions
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def current_image():
    global original, lines, line_types_to_colors
    
    current = original.copy()
    for ln in lines:
        cColor = line_types_to_colors[ln['label']]
        cv2.line(current, tuple(ln['start']), tuple(ln['end']), cColor, ln['thickness'])
    return current


def on_mouse(event, x, y, flags, params):
    global lines, line_types, line_types_to_colors, cTypeIndex, cThickness, mTypeIndex, t0, thresh
    global image, ix, iy, pressed
    
    shift = flags & cv2.EVENT_FLAG_SHIFTKEY

    cType = line_types[cTypeIndex]
    cColor = line_types_to_colors[cType]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if mTypeIndex is MODE_TYPES.MULTI_POINT.value:
            if ix >= 0 and iy >= 0: # we've clicked somewhere valid
                image = current_image()
                cv2.line(image, (ix, iy), (x, y), cColor, cThickness)
                if ix != x or iy != y:
                    # line = { 'start': (ix,iy), 'end': (x,y), 'BGR-color': cColor, 'label': cType, 'thickness': cThickness }
                    line = {'start': (ix, iy), 'end': (x, y), 'label': cType, 'thickness': cThickness}
                    lines.append(line)

            ix, iy = x, y
        else:
            pressed = True
            if lines and shift:
                # find the closest point and the corresponding line
                # remove the line from lines
                closest_line = None
                closest_point = None
                closest_distance2 = None

                for ln in lines:
                    xx, yy = ln['start']
                    distance2 = (x - xx) ** 2 + (y - yy) ** 2
                    if (not closest_distance2) or (distance2 < closest_distance2):
                        closest_distance2 = distance2
                        closest_line = ln
                        closest_point = 'start'
                    xx, yy = ln['end']
                    distance2 = (x - xx) ** 2 + (y - yy) ** 2
                    if (distance2 < closest_distance2):
                        closest_distance2 = distance2
                        closest_line = ln
                        closest_point = 'end'

                lines.remove(closest_line)
                ix, iy = closest_line[('end' if closest_point == 'start' else 'start')]
                cTypeIndex = line_types.index(closest_line['label'])
                cThickness = closest_line['thickness']

            else:
                ix, iy = x, y
    
    if event == cv2.EVENT_MOUSEMOVE:
        if mTypeIndex is MODE_TYPES.CONTINUOUS.value:
            if pressed:
                image = current_image()
                cv2.line(image, (ix, iy), (x, y), cColor, cThickness)
                if ix != x or iy != y:
                    # line = { 'start': (ix,iy), 'end': (x,y), 'BGR-color': cColor, 'label': cType, 'thickness': cThickness }
                    line = {'start': (ix, iy), 'end': (x, y), 'label': cType, 'thickness': cThickness}
                    lines.append(line)
                ix, iy = x, y
        else:
            if pressed:
                image = current_image()
                cv2.line(image, (ix, iy), (x, y), cColor, cThickness)
    
    if event == cv2.EVENT_LBUTTONUP:
        if mTypeIndex is not MODE_TYPES.MULTI_POINT.value and pressed:
            image = current_image()
            cv2.line(image, (ix, iy), (x, y), cColor, cThickness)
            if ix != x or iy != y:
                #line = { 'start': (ix,iy), 'end': (x,y), 'BGR-color': cColor, 'label': cType, 'thickness': cThickness }
                line = { 'start': (ix,iy), 'end': (x,y), 'label': cType, 'thickness': cThickness }
                lines.append(line)
            pressed = False


def save(json_filename, lines):
    
    jj = { 
            "area" : 0,
            "boxes" : 0, 
            "circles" : 0, 
            "lines" : lines, 
            "areaByColor" : {}, 
            "metadata" : { "tags" : [] }, 
            "numShapes" : -1, 
         }
    
    # print json.dumps(jj, indent=4, separators=(',', ': '), sort_keys=True)
    f = open(json_filename, 'w')
    json.dump(jj, f)
    f.close()


def load(json_file):
    f = open(json_file)
    j = json.load(f)
    f.close()
    return j['lines']


if __name__ == "__main__":

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Initialization
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if len(sys.argv) == 1:
        print "usage: image-file [annotations-file]"
        exit()

    image_file = 'house.jpg'

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

    ann = image_file[:-len('.PNG')] # implicitly works with .jpg extension too
    annotation_file = ann + '.json'

    if len(sys.argv) > 2:
        annotation_file = sys.argv[2]

    if os.path.isfile(annotation_file):
        lines = load(annotation_file)

    original = cv2.imread(image_file)
    image = current_image()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)

    print("Line type: {}".format(line_types[cTypeIndex]))
    print("Mode type: {}".format(MODE_TYPES_MAP[MODE_TYPES(mTypeIndex)]))

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Main Loop
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    while (1):
        cv2.imshow('image', image)
        cType = line_types[cTypeIndex]
        # cv2.displayStatusBar('image', "Currently drawing: " + cType, 0)


        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        elif k == 255:
            continue
        elif k == 225:      # shift key
            continue

        elif k == 8:        # backspace to delete current or last
            if pressed:
                pressed = False
            elif lines:
                line = lines.pop()
                if len(lines) == 0:
                    ix, iy = -1, -1
                else:
                    ix, iy = line['start']
            image = current_image()

        elif k == ord('s'): # save
            save(annotation_file, lines)

        elif k == ord('t'): # change type of line
            cTypeIndex = (1 + cTypeIndex) % len(line_types)
            print("Line type: {}".format(line_types[cTypeIndex]))

        elif k == ord('m'): # line mode
            ix, iy = -1, -1
            mTypeIndex = (1 + mTypeIndex) % len(MODE_TYPES_MAP)
            print("Mode type: {}".format(MODE_TYPES_MAP[MODE_TYPES(mTypeIndex)]))

        elif k == ord('1'):
            cThickness = 1
        elif k == ord('2'):
            cThickness = 2
        elif k == ord('3'):
            cThickness = 3
        elif k == ord('4'):
            cThickness = 4
        elif k == ord('5'):
            cThickness = 5
        elif k == ord('6'):
            cThickness = 6
        elif k == ord('7'):
            cThickness = 7
        elif k == ord('8'):
            cThickness = 8
        elif k == ord('9'):
            cThickness = 9
        else:
            print k

    cv2.destroyAllWindows()
