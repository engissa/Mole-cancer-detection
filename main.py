#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:12:52 2019

@author: issa
"""
import os
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from skimage.util import img_as_bool
from skimage import data, transform
from scipy import ndimage
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import math



class ContourDetector():
    def __init__(self, image_path, clusters , gaussian_sigma = 0):
        
        self.__clusters = clusters
        self.__image_path = image_path
        self.img_raw = mpimg.imread(self.__image_path)
        self.img = self.img_raw.copy()
        self.img_mod = None
        self.im_binary = None
        self.im_binray_rot = None
        self.im_binary_rot_cr = None
        self.temp_im_rot = None
        self.mole_center_x = None
        self.mole_center_y = None
        self.top_line = None
        self.bottom_line = None
        self.right_line = None
        self.left_line = None

        self.contour_matrix = None 
        self.mole_matrix = None
        self.outer_fill = None

        self.gaussian_en = False
        self.gaussian_sigma = gaussian_sigma
        self.labels = None
        self.centroids = None
        self.cent_idx = -1

    def start(self):
        '''
        Description
        ------------
        Extracts features from a picture containing a skin mole
        Steps of Extraction:
            - Apply a gaussian filter
            - Extract clusters of 3 colors
            - Selects the darkest color cluster
            - Generate a binary image from pixels labeled with darkest color cluster
            - Removes noisy pixels
            - Rotate Image by 45 degrees to crop part of the outline borders
            - Calculate Mole Approximate center point
            - Get a bouding box around the mole
            - 
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # apply a gaussian filter over the image
        if self.gaussian_sigma !=0 : 
            self.img = ndimage.gaussian_filter(self.img,2)
        [N1,N2,N3] = self.img.shape
        image_2d = self.img.reshape((N1*N2,N3))
        self.img_mod = image_2d.copy()
        # Extract cluster centroids and labels of colors using K means clustering
        kmeans = KMeans(n_clusters=self.__clusters,random_state=0)
        kmeans.fit(image_2d)
        # calculate the mean of color
        self.centroids = kmeans.cluster_centers_.astype('uint8')
        self.labels = kmeans.labels_
        # Determine the darkest cluster index with the shortest distance to black RGB
        self.cent_idx = self._darkest_centroid()
        print(self.centroids[self.cent_idx])
        # print(self.cent_idx)
        # Show only pixels with selected label index 
        # Selected pixel is black other pixel are white
        indexes = (self.labels == self.cent_idx)
        self.img_mod[indexes,:] = 0
        self.img_mod[np.invert(indexes),:] = 255
        # Get the binary matrix
        self.im_binary = (kmeans.labels_== self.cent_idx)
        self.im_binary = self.im_binary.reshape((N1,N2))
        
        # Remove some noise like lonely pixels
        self._remove_noise(window_size=3)
        # Rotate image by 45 degree to remove image boundaries
        # Convert image to a binary matrix
        
        self.im_binray_rot = img_as_bool(transform.rotate(self.im_binary, 45.0, resize=False))
        # Calculate mole center point

        self.mole_center_x , self.mole_center_y = self._calc_mole_center()
        self.get_bounding_box()
        # Crop the image
        self._crop_image()
        
        # get contour and mole fill
        self.contour_matrix = np.zeros(self.im_binary_rot_cr.shape).astype('bool')
        # print('contour matrix',self.contour_matrix.shape)
        # Apply the flood fill algorithm

        temp_im_rot = self.im_binary_rot_cr.copy().astype('uint8')
        self._flood_fill(temp_im_rot,self.contour_matrix,(2,2),3)
#        plt.figure(figsize=(11,8))
#        plt.imshow(self.contour_matrix,interpolation=None)
        
        # get the outer filled region
        self.outer_fill = np.where(temp_im_rot==3,False,True).astype('bool')
#        plt.figure(figsize=(11,8))
#        plt.imshow(self.outer_fill,interpolation=None)
        # get filled matrix of the mole only
        self.mole_matrix = np.logical_not(self.outer_fill)
#        plt.figure(figsize=(11,8))
#        plt.imshow(self.mole_matrix,interpolation=None)
        # return contour_matrix,temp_im_rot , self.im_binary_rot_cr

    def get_params(self):
        '''
        Description
        ------------
        Calculate the Mole characteristics
        
        Parameters
        ----------
        None

        Returns
        -------
        c_mole: float
            Perimeter of the Mole
        a_mole: float
            Area of the Mole
        c_circle: float
            Perimeter of a perfect circle of Area equal to the mole
        ratio: float
            Ratio of the Perimeter of the mole with respect to the circle Perimeter
        
        '''
        c_mole = np.sum(self.contour_matrix)
        a_mole = np.sum(self.outer_fill) 
        
        r_circle = math.sqrt(a_mole/math.pi)
        c_circle = r_circle * math.pi * 2
        
        ratio = c_mole/c_circle
        
        return (c_mole,a_mole,c_circle,ratio)


    def get_bounding_box(self,precision=2):
        '''
        Description
        ------------
        Gets a bouding box around the mole.
        
        The algorithm starts from mole center point and draws a square of Left,Right columns and
        Top/Bottom rows. The boudaries of the square expand by checking the number of pixels on each side.
        The stop condition is met when each of the lines contains number a certain amount of pixels.
        
        Parameters
        ----------
        precision : int
            Number of pixels to consider a matrix row/column as empty

        Returns
        -------
        None, Modifies global top_line , bottom_line , left_line and right_line values
        
        '''
        top_line = self.mole_center_y-1
        bottom_line = self.mole_center_y+1
        right_line = self.mole_center_x+1
        left_line = self.mole_center_x-1
        edge_found = [0,0,0,0]

        while(True):

            tl = np.sum(self.im_binray_rot[top_line,left_line:right_line])
            bl = np.sum(self.im_binray_rot[bottom_line,left_line:right_line])
            rl = np.sum(self.im_binray_rot[top_line:bottom_line,right_line])
            ll = np.sum(self.im_binray_rot[top_line:bottom_line,left_line])
            
            if tl >= precision:
                top_line -=1
            else:
                edge_found[0] = 1
            if bl >= precision:
                bottom_line +=1
            else:
                edge_found[1] = 1
            if rl >= precision:
                right_line +=1
            else:
                edge_found[2] = 1
            if ll >= precision:
                left_line -=1
            else:
                edge_found[3] = 1
            if sum(edge_found)==4:
                break
        self.top_line = top_line
        self.bottom_line = bottom_line
        self.right_line = right_line
        self.left_line = left_line

    def _flood_fill(self,matrix,contour_matrix,start_point, fill_val=3):
        xsize, ysize = matrix.shape
        orig_value = matrix[start_point[0], start_point[1]]
        stack = set(((start_point[0], start_point[1]),))
        if fill_val == orig_value:
            raise ValueError("Filling region with same value")    
        while stack:
            x, y = stack.pop()
            if matrix[x,y] == 1:
                # trace mole border pixel if found
                contour_matrix[x,y] = True
            if matrix[x, y] == orig_value:
                # trace visited non mole pixels by fill val
                matrix[x, y] = fill_val
                if x > 0:
                    # go Left
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    # go Right
                    stack.add((x + 1, y))
                if y > 0:
                    # go Top
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    # go Bottom
                    stack.add((x, y + 1))

    def _remove_noise(self,window_size):
        '''
        Description
        ------------
        Removes Noise in a binary image 
        Any pixel bounded within the borders of a certain window size is considered noisy pixel
        This algorithm iterates a window matrix row by column over the whole matrix and removes noisy pixels
        
        Parameters
        ----------
        window size: int 
            The length of an NxN square matrix. Must be greater that 3
        
        Returns
        -------
        None, Modifies the binary image inplace
        '''
        x_max = self.im_binary.shape[0]
        y_max = self.im_binary.shape[1]
        for row in range(x_max - window_size):
           for col in range(y_max -window_size):
               sum1 = 0
               sum1 = np.sum(self.im_binary[row:row+window_size, col])
               sum1 += np.sum(self.im_binary[row, col:col+window_size])
               sum1 += np.sum(self.im_binary[row:row+window_size, col+window_size])
               sum1 += np.sum(self.im_binary[row+window_size, col:col+window_size])
               if sum1 <= 1:
                   self.im_binary[row+1:row+window_size-1,col+1:col+window_size-1] = False

    def _crop_image(self,dist=20):
        '''
        Description
        ------------
        Crops the rotated binary matrix

        Parameters
        ----------
        dist: int 
            The length of distance to keep outside crop line
        
        Returns
        -------
        None, Modifies the rotated binary cropped image inplace
        
        '''
        self.im_binary_rot_cr = self.im_binray_rot[self.top_line-dist:self.bottom_line+dist,
        self.left_line-dist:self.right_line+dist]

    def _darkest_centroid(self):
        '''
        Description
        ------------
        Selects the index of the darkest contour.
        Checks the color that has the closest eucledean distance to the Black RGB Color

        Parameters
        ----------
        None
        
        Returns
        -------
        idx : int
            The index of the contour
        
        '''
        min_dist = 255
        idx = -1
        for i in range(self.centroids.shape[0]):
            dist = distance.euclidean(self.centroids[i,:],[0,0,0])
            if dist <= min_dist:
                min_dist = dist
                idx =i
        return idx

    def _calc_mole_center(self):
        '''
        Description
        ------------
        Gets the approximated mole center point
        Calculate the intersection point of row and column with largest number of dark pixels

        Parameters
        ----------
        None
        
        Returns
        -------
        (x,y) : tuple
            x and y coordinates of the mole center
        '''
        x = np.argmax(np.sum(self.im_binray_rot, axis=0))
        y = np.argmax(np.sum(self.im_binray_rot, axis=1))
        
        return (x,y)

    def draw_bounding_box(self,crop=False):
        im_rot = self.im_binray_rot.copy()
        x_max = im_rot.shape[0]
        y_max = im_rot.shape[1]
        bounding_rect = None
        _,ax = plt.subplots(1,figsize=(11,8))
        if crop:
            im_rot[0:self.top_line,:]= False
            im_rot[:,self.right_line:x_max] = False
            im_rot[self.bottom_line:y_max,:]= False
            im_rot[:,0:self.left_line] = False
            ax.set_title("Bounding Box cropped Image")
        else:
            ax.set_title("Bounding Box Uncropped Image")

        bounding_rect = Rectangle((self.left_line,self.top_line),
         self.right_line-self.left_line,
         self.bottom_line-self.top_line,
         linewidth=2,edgecolor='g',facecolor='none')
        ax.imshow(im_rot)
        ax.add_patch(bounding_rect)
        plt.show()

    def show_croped_image(self):
        plt.figure(figsize=(11,8))
        plt.imshow(self.im_binary_rot_cr,interpolation=None)
        plt.title('Binary Image after Crop')
        plt.show()
    def show_clustered_image(self):
        tempImg = self.img_mod.reshape(self.img.shape)
        plt.figure(figsize=(11,8))
        plt.imshow(tempImg,interpolation=None)
        plt.title('Labeled image after K means clustering')
        plt.show()        
    def draw_mole_center(self):
        # function to plot the center of the mole on the binary image
        circ = Circle((self.mole_center_x,self.mole_center_y),5)
        _,ax = plt.subplots(1,figsize=(11,8))
        ax.set_title("Mole center point")
        ax.imshow(self.im_binray_rot)
        ax.add_patch(circ)
        plt.show()
    def show_contour(self):
        plt.imshow(self.contour_matrix)
        plt.show()    
    def trace_contour_org_img(self):
        rot_image = transform.rotate(self.img_raw, 45.0, resize=False)
        rot_image = rot_image[self.top_line-20:self.bottom_line+20,
        self.left_line-20:self.right_line+20,:]
        rot_image[self.contour_matrix,0]=124
        rot_image[self.contour_matrix,1]=252
        rot_image[self.contour_matrix,2]=0
        plt.figure(figsize=(11,8))
        plt.imshow(rot_image,interpolation=None)
#        plt.title('Contour on original image')
        plt.show()

    
                

def main():
    img_folder = 'img'
    # img_name = "melanoma_2.jpg"

    list_images = os.listdir(img_folder)
    csv = ''
    params = []
    # Choose the range of images to be analyzed
    i = 25
    j = 28
    for img_name in list_images[i:j]:
        try:
            if(img_name.split('.')[1]!='jpg'):
                continue
            img_path = os.getcwd() + '/' + img_folder + '/' + img_name 
            testObjs = ContourDetector(image_path= img_path, clusters=3, gaussian_sigma=0)
            testObjs.start()
            testObjs.show_clustered_image()
            testObjs.draw_mole_center()
            testObjs.draw_bounding_box(crop=False)
            testObjs.draw_bounding_box(crop=True)
            testObjs.show_croped_image()
            testObjs.trace_contour_org_img()
            testObjs.show_contour()
            c_mole,a_mole,c_circle,ratio = testObjs.get_params()
            img = img_name.split('.')
            csv=csv+str(c_mole)+','+str(a_mole)+','+\
            str(c_circle)+','+str(ratio)+','+img[0]+'\n';
            params.append({'c_mole':c_mole,
            'a_mole':a_mole,
            'c_circle':c_circle,
            'ratio':ratio,
            'name':img[0]})

        except Exception as e:
            print(e)
            continue
    print(csv)
    print(params)
    
if __name__ == "__main__":
   main()
    
    