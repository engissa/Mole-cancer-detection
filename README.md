# Mole-cancer-detection
## Introduction
Melanoma is a dangerous form of cancer that hits the skin cells and leads them to multiply rapidly and form malignant tumors. It often resembles moles and is caused mainly by intense UV exposure. The majority of the infected cells are black or brown but can also be skin- colored pink, red, purple, blue, or white. If melanoma is recognized and treated early, it is almost always curable; otherwise, the cancer advance and spread to other parts of the body. To diagnose melanoma, Doctors look at the ABCDE signs :
* A asymmetry
* B border
* C color
* D diameter 
* E evolution

### Objective
The focus of this lab is to analyze the borders of the Mole and extract features that will be used together with other elements of the ABCDE for Mole classification. The main objectives for the feature extraction are the following:
1. Quantize the image using the K-means algorithm to find three clusters.
1. Find contour corresponding to the mole using the darkest cluster.
1. Calculate the area and parameter of the contour corresponding to the mole.
1. Evaluate the perimeter of a perfect circle with an area equal to that of the mole.
1. Evaluate the ratio between the perimeter of the mole and the perimeter of the corre- sponding circle.

### Dataset

The dataset used contains 54 RGB images of moles in the Jpeg format and of dimension 583x583 pixels. Images are classified by a label that indicates the probability of being melanoma and are distributed as follows:


Labels | Risk Level | Number of Images
-------|-------|------
low_risk | Low | 12
medium_risk | Medium | 16
melanoma | High | 27


The sample raw images shows how the mole has different characteristics depending on the risk.

![Graph](https://github.com/engissa/Mole-cancer-detection/blob/master/Figures/raw_samples.png)

## Methodologies and Implementation
### Preprocessing and Quantization

Each image contained in the dataset was handled separately for analysis. Image ”melanoma 1” in fig 2.2 a is selected to demonstrate the implementation of the lab in this report; however, the analysis was done on the whole dataset. The image was imported using python Mpimg function that assembled a three-dimensional NdArray of 583x583 pixels consisting of three layers of RGB colors. Each pixel contains the amount of color (0-255) it holds for its layer. It is difficult to find the pixels belonging to the Mole due to the high range of possible values each pixel can carry.From the color feature, distinct patterns can be extracted to detect the label of the pixel and classify if it belongs to the Mole.

![Graph](https://github.com/engissa/Mole-cancer-detection/blob/master/Figures/quantizations.png)

The K-means clustering method aims to partition the 583x583 RGB pixels into k clusters in which each pixel belongs to the cluster with the nearest mean. It was implemented using Sklearn’s KMeans method with three as the number of intended clusters. The pixels to be provided to the method should be a 2-dimensional array, so the image matrix was reshaped into N1*N2 rows and N3 columns. The N1*N2 rows represent all the pixels of one color layer of the image, and the N3 column represents the RGB color layer of the underlying rows. After fitting the matrix of the reshaped image, the K-means clustering algorithm gave the 3x3 matrix containing the RGB colors of the centroids of the clusters and an N1*N2*N3 matrix containing the label of each pixel.

It was assumed that the Mole holds the darkest colors inside in the image, so the centroid of the darkest color is probably in the cluster that represents the Mole. The darkest cluster was selected by choosing the one with the nearest Euclidean distance with the color black (0,0,0). In this case, the chosen cluster was [88 85 85 ]. Afterward, the image was quantized by replacing the matrix of labels with the color of the selected centroid as in fig. 2.3. The three color grades are well visible in the picture where one of them represents the Mole. After that, a binary matrix containing only the pixels of the Mole was produced as in fig.2.4. The 1 represents the presence of pixel, and 0 represents the absence of pixel.
### Noise Removal

The binary image fig.2.4 showed that some pixels don’t belong to the Mole and considered as noise. An algorithm was implemented to remove them these pixels that are referred to as lonely pixels. The base of the algorithm is to form a square window matrix of dimension nxn ( n is an odd real number) and iterate through entire matrix elements. If all the elements on the first column and row and on the nth column and row are 0, then it removes all the contained pixels. In this image the (n) was chosen as 3, meaning it removes any 1 pixel that is not surrounded by any other 1 pixel. Moreover, the image was rotated by 45 degrees to eliminate some of the noise that can be found on the edge of the image. Fig. 2.5 Shows how these methods removed some of the noise in the image.

### Detection of Mole center and bounding box

The approximate center point of the Mole with coordinates ( x,y) was calculated as follows : 
* x is the index of the column holding the maximum number 1s
* y is the index of the row holding the maximum number of 1s

In other words, the center point is the intersection of the column and row that hold the maximum number of pixels in the Mole. Fig. 2.6 shows this way gave a reasonable estimation of the center point of the Mole.
A bounding box around the Mole was drawn after finding the mole center point. The bound- ing box enabled the cropping of the image to only include the Mole and eliminate any other remaining noises in the picture. The bounding box has 4 index parameters leftP, rightP, topP, and bottomP. The algorithm of drawing the bounding box is as follows :
1. Start from the center point x,y
1. Draw bounding box lines as follow:
    * Left line : ( x-1,y+topP) to (x-1,bottomP)
    * Right line : (x+1,y+topP) to (x+1,bottomP)
    * Top line: (leftP,y-1) to (rightP,y-1)
    * Bottom line: (leftP,y+1) to (rightP,y+1)
3. Calculate the number of 1s on each line
4. Check each line: if the number of 1s is greater than a certain threshold ( in this case 2) then increment the line by a step
5. Keep incrementing each line until the sum of 1s in each line reaches the threshold.
6. The bounding box is found when each of the lines has a sum equal to the threshold.

The center point of the Mole and the bounding box of the current image was found as in Fig.2.7

![Graph](https://github.com/engissa/Mole-cancer-detection/blob/master/Figures/bounding_box.png)
### Contour Detection

Fig. 2.8a shows the cropped image after the detection of the bounding box in order to facilitate the process of finding the contour of the mole. The algorithm developed for finding the edge of the Mole is a recursive algorithm based on Flood Filling as in Alg:1 . It uses a copy of the binary matrix and zeros matrix of the same size that was called the ” contour matrix.” The algorithm starts from a 0 point outside the mole on the edge of the image. Afterward, it starts filling all the 0 elements with 3 until it reaches an element equal to 1 where it stops and stores the position of that element in the ”contour matrix. In this case, the algorithm detected all the pixels surrounding the mole as in Fig. 2.8b and the contour of the mole as in Fig. 2.8c. The matrix of the surrounding pixels was transformed into a binary matrix by selecting the elements = 3. Afterward, the matrix was inverted to get the mole filled region as in Fig. 2.8d This algorithm was able to detect the contour of the without taking into consideration the gaps inside it like in Fig. 2.8a.Moreover , Fig. 2.8e shows the contour fitting the mole on the original image.
![Graph](https://github.com/engissa/Mole-cancer-detection/blob/master/Figures/algorithm.png)

### Mole Geometric Calculations

The area of the mole was calculated using the summation of the number of pixels equal to 1 in the filled matrix in fig 2.2.4. On the other side, the perimeter was calculated using the summation of the number of pixels equal to 1 in the contour matrix in fig. After calculating the area and perimeter of the mole, the perimeter of an equivalent circle of area equal to that of the mole was evaluated using eq. 2.2. Finally, the ratio between the perimeter of the Mole and the perimeter of the corresponding circle was evaluated to tell how far is the mole border indented.

![Graph](https://github.com/engissa/Mole-cancer-detection/blob/master/Figures/geometric.png)

## Results and Conclusion
The steps implemented in this lab were performed automatically on all the images in the dataset that resulted in the equivalent ratios in the table.2.2. However, it failed to process the image in the following cases: low risk 10, medium risk 1, melanoma 6, melanoma 27. The failure was due to the very high noise present in the picture where it was unable to detect the center of the circle and draw a proper bounding box as shown in fig.2.9a . Moreover, some images showed a high ratio like in fig.2.9b that is due to the irregularities of the same and the noise in the image. In general, the algorithm was able to draw the contour over the mole in most of the cases and the use of flood filling enabled a more accurate calculation of the area.
![Graph](https://github.com/engissa/Mole-cancer-detection/blob/master/Figures/results.png)