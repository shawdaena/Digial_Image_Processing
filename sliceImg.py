import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/enoconda/Desktop/DIP/Image/tulip.png')
img1 = img[:,:,0]

bit_planes = []
planes = []

for i in range(8):
    bit_plane = ((img1 >> i)& 1) * 255
    bit_planes.append(bit_plane)
    plane = ((img1 >> i)& 1) * (2**i)
    planes.append(plane)

reconstract_img = sum(planes)
combined_img123 = planes[0]+planes[1]+planes[2]
combined_img234 = planes[1]+planes[2]+planes[3]
combined_img345 = planes[2]+planes[3]+planes[4]
combined_img456 = planes[3]+planes[4]+planes[5]
combined_img567 = planes[4]+planes[5]+planes[6]
combined_img678 = planes[5]+planes[6]+planes[7]

# Display result
plt.figure(figsize=(10, 10))
plt.subplot(4, 4, 1)
plt.imshow(img1, cmap='gray')
plt.title('Original')
plt.axis('off')

for i in range(8):
    plt.subplot(4, 4, i + 2)
    plt.imshow(bit_planes[7 - i], cmap='gray')
    plt.title(f'Bit {7 - i}')
    plt.axis('off')


plt.subplot(4,4,10)
plt.imshow(combined_img123,cmap='gray')
plt.title('combined_img123')
plt.axis('off')

plt.subplot(4,4,11)
plt.imshow(combined_img234,cmap='gray')
plt.title('combined_img234')
plt.axis('off')

plt.subplot(4,4,12)
plt.imshow(combined_img345,cmap='gray')
plt.title('combined_img345')
plt.axis('off')

plt.subplot(4,4,13)
plt.imshow(combined_img456,cmap='gray')
plt.title('combined_img456')
plt.axis('off')

plt.subplot(4,4,14)
plt.imshow(combined_img567,cmap='gray')
plt.title('combined_img567')
plt.axis('off')

plt.subplot(4,4,15)
plt.imshow(combined_img678,cmap='gray')
plt.title('combined_img678')
plt.axis('off')

plt.subplot(4,4,16)
plt.imshow(reconstract_img,cmap='gray')
plt.title('Reconstract_img')
plt.axis('off')

plt.tight_layout()
plt.show()
