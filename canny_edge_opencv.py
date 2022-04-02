#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

figureSize = (12,10)

image = cv2.imread("images/cat.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure("Original Image", figsize=figureSize)
plt.imshow(imageGray)
plt.set_cmap("gray")

cannyImage = cv2.Canny(np.uint8(imageGray),100,200);
plt.figure("Canny Image", figsize=figureSize)
plt.imshow(cannyImage)

## inverted 
plt.figure("Canny Image (inverted)", figsize=figureSize)
plt.imshow(255-cannyImage)


# In[ ]:




