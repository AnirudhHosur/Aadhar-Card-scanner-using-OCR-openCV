# OCR of E-aadhar cards [Good quality images]

#Image quality enhancement – Improve the quality of the images using image processing libraries

# First import the librares
import cv2 
import pytesseract
import numpy as np


"""Convert the image into black and white"""
#read image
img_grey = cv2.imread("CARD.jpg", cv2.IMREAD_GRAYSCALE)
# define a threshold, 128 is the middle of black and white in grey scale
thresh = 128
# threshold the image
img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]
#save image
cv2.imwrite("CARD2.jpg",img_binary)


"""Now Noise Removal & Image Sharpening"""
bW = cv2.imread("CARD2.jpg")
bi_blur = cv2.bilateralFilter(bW, 9, 75, 75)
cv2.imwrite("CARD3.jpg",bi_blur)


final = cv2.imread("CARD3.jpg")
print(final.shape)

# Recognise the digits
custom_config = r'--oem 3 --psm 6 outputbase digits'

# Printing the details of the aadhar card
print(pytesseract.image_to_string(final, config=custom_config))













