import numpy as np
import cv2

import sys


# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 01:22:57 2018

@author: tchang422
"""

def print_seam(rows, cols, seams, energy, square):
    
    #Print for the first and second seam
    if seams == 0 or seams == 1: 
        print("Points on seam %i:" % (seams))
        print("horizontal" if flip else "vertical")
        print("%i, %i" % (points[0], 0) if flip else (0, points[0]) )
        print("%i, %i" % (points[rows//2], rows//2) if flip else (rows//2, points[rows//2]))
        print("%i, %i" % (points[rows-1], rows-1) if flip else (rows-1,points[rows-1]))
        print("Energy of seam %i: %0.2f" % (seams, energy/len(points)) )
        print("")
        
    #Print for the last seam
    if rows == cols-1 or cols == rows-1:
        print("Points on seam %i:" % (seams))
        print("horizontal" if flip else "vertical")
        print("%i, %i" % (points[0], 0) if flip else (0, points[0]) )
        print("%i, %i" % (points[rows//2], rows//2) if flip else (rows//2, points[rows//2]))
        print("%i, %i" % (points[rows-1], rows-1) if flip else (rows-1,points[rows-1]))
        print("Energy of seam %i: %0.2f" % (seams, energy/len(points)) )
        print("")

def seam_image(W, E, seam, img, it, seams, img_name):
    
    #Create an image copy, with the column size one smaller, since we're removing one pixel from each row
    seamed_img = np.zeros([rows, cols-1, 3])
    
    #Array for holding indices of seam
    points = []
    energy = 0
    
    #Copy is the image we save with the first seam drawn on it
    if seams == 0:
        copy = img
        
    points.append(seam)
    for i in range(it, 0, -1):
        
        #Find the least cost path to the previous row, and set the next seam index
        step_value = np.argmin(W[i-1, seam-1:seam+2])
        next_seam = seam
    
        #If the least cost path index is the 0th element in the sliced array above, the index is to the left
        if step_value == 0:
            next_seam -= 1
        #The same as above, but for right
        elif step_value == 2:
            next_seam += 1
        
        #Add to the energy, so we can calculate the seam energy later
        energy += E[i, seam]
                        
        #If this is the first seam, draw on the seam pixel
        if seams == 0:
            copy[i, seam] = (0, 0, 255)
        
          
        #Slice the image into a new image, with the column size one smaller
        seamed_img[i, 0:seam, 0:3] = img[i, 0:seam, 0:3]
        seamed_img[i, seam:cols-1, 0:3] = img[i, seam+1:cols, 0:3]
            
        #Add to our list of seam indices
        points.append(next_seam)
            
        #Set the next seam index
        seam = next_seam
        
    #If this is our first seam, save the copy image with the seam drawn on
    if seams == 0:
        cv2.imwrite(img_name + "_seam.png", copy)
            
    return points, energy, seamed_img

if __name__ == "__main__":
    
    #My seam coordinates are off, but the energies are very similar
    #Also my pictures are very similar to those of the correct ones
    
    #Read the image and obtain the file name
    np.set_printoptions(suppress=True)
    img_in = "dog.jpg"
    img_name = img_in[:-4]
    img = cv2.imread(img_in)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    #Flip if the rows are bigger
    flip = rows > cols
    if rows > cols:
        img = np.rot90(img)   
    

    rows = img.shape[0]
    cols = img.shape[1]
    
    square = rows if rows < cols else cols
    seams = 0
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Keep the loop running until the image is a square
    while rows != cols:
        
        img = np.uint8(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Calculate the sobel derivatives of x and y
        dx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1)
    
        #Combine them to get the energy matrix
        E = np.abs(dx) + np.abs(dy)
    
        #Initiliaze an empty matrix in the shape of the energy matrix
        W = np.zeros([rows, cols])
        W[0, 0:cols] = E[0, 0:cols]
    
        #Set the left and right columns to very high numbers
        W[0:rows, 0] = 1000000
        W[0:rows, cols-1] = 1000000
    

        #Loop through the rows of the image, starting from the 2nd row
        #Since the first row is identical to that of the energy matrix
        for i in range(1, rows):
    
            #Create matrices with views of left origin and right, in the previous row
            origin = W[i-1, 1:cols-1]
            left = W[i-1, 0:cols-2]
            right = W[i-1, 2:cols]
            
            #Get the minimum cost value of the left origin and right values
            minimum = np.minimum(origin, right)
            minimum = np.minimum(minimum, left)

            W[i, 1:cols-1] = E[i, 1:cols-1] + minimum
    
        #Set the index for the first seam pixel we remove
        seam = np.argmin(W[square-1, 0:cols])
    
        
        
        #Call the seam image function
        #Returns list of indices of the points, the energy of the seam, and the img
        points, energy, img = seam_image(W, E, seam, img, square-1, seams, img_name)
        
        #Call our print function
        print_seam(rows, cols, seams, energy, square)
        
        
        #Change our row and column to fit the seamed image
        rows = img.shape[0]
        cols = img.shape[1]
        seams += 1
        
        

    #If we had to flip the image, flip it so that it's back to normal
    if flip:
        img = np.rot90(img, 3)
        
    #Write the final image
    cv2.imwrite(img_name + "_final.png", img)