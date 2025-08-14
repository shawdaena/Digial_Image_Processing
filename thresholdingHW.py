import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    img1 = cv2.imread('/home/enoconda/Desktop/DIP/Image/pic1.png', 0)
    img2 = function1(img1)
    img3 = function2(img1)
    img4 = function3(img1)

    img_set = [img1, img2, img3, img4]
    title_set = ["Original Image", "Step Function", "Function2", "Function3"]

    plt.figure(figsize=(12, 6))
    for i in range(len(img_set)):
        # Show image
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(title_set[i])
        plt.axis('off')

        # Show histogram
        plt.subplot(2, 4, i + 5)
        plt.hist(img_set[i].ravel(), bins=256, color='gray')
        plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()

def function1(img):
    T = 127
    img1 = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):    
        for j in range(img.shape[1]):   
            img1[i, j] = 255 if img[i, j] >= T else 0
    return img1

def function2(img):
    T = 127
    img1 = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):    
        for j in range(img.shape[1]):  
            if img[i, j] <= T:  
                img1[i, j] = img[i,j]*1 
            else:
                img1[i, j] = img[i,j]*0.5
    return img1

def function3(img):
    img1 = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            I = img[i, j]
            if I <= 100:  # Linear: m*I + c
                img1[i, j] = int(0.01 * I * 255 / 2)  # Adjusted scaling
            elif I <= 180: 
                img1[i, j] = I
            else:  # Multiply by 2
                img1[i, j] = min(255, int(I * 2))  # Clip to 255
    return img1

if __name__ == '__main__':
    main()
