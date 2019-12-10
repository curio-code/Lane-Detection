#Author=Vivek Kumar Jaiswal
#This program takes a Photo and returns the lanes found in the picture.

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Canny Function is to convert the image to B/W and to draw the outlines of the lanes.
def canny(image):
    gray = cv.cvtColor(copy_image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5,5),0)
    canny= cv.Canny(blur, 50,150)
    return canny

#To draw the lanes from the lines image
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
           # for x1, y1, x2, y2 in line:
            x1,y1,x2,y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

#Specify the particular area of interest
def region_of_interest(image):
    height=image.shape[0]
    polygons = np.array([
    [(200,height), (1100, height), (550, 250)]
    ])
    mask=np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image=cv.bitwise_and(image,mask)
    return masked_image

def make_coordinate(image, parameters):
    slope, intercept = parameters
    y1 = image.shape[0]
    y2 = int (y1*3/5)
    x1 = int ((y1-intercept)/slope)
    x2 = int ((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_lane_slope(image, lines):
    left_lane=[]
    right_lane=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1,x2),(y1,y2),1 )
            slope=parameters[0]
            intercept=parameters[1]
            if slope < 0 :
                left_lane.append((slope,intercept))
            else:
                right_lane.append((slope,intercept))
    
    left_lane_average=np.average(left_lane, axis=0)
    right_lane_average=np.average(right_lane, axis=0)
    left_line=make_coordinate(copy_image, left_lane_average)
    right_line=make_coordinate(copy_image, right_lane_average)
    return np.array([left_line, right_line])

image = cv.imread('test_image.jpg')
copy_image = np.copy(image) # we have to make a copy because otherwise changes would be made in the orginal image
canny=canny(copy_image)
cropped_image=region_of_interest(canny)
lines=cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_line=average_lane_slope(copy_image,lines)
line_image = display_lines(copy_image, averaged_line)
combo_image=cv.addWeighted(copy_image, 0.8, line_image, 1, 1)
cv.imshow('result', combo_image)
cv.waitKey(0)
