import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    img_gray = cv2.imread("/home/enoconda/Desktop/DIP/Image/building.png", cv2.IMREAD_GRAYSCALE)

    kernel_avg = np.ones((3, 3), np.float32) / 9

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    
    prewitt_y= np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])
    
    laplace_kernel = np.array([[0,  1, 0],
                               [1, -4, 1],
                               [0,  1, 0]])
    
    img_avg = cv2.filter2D(img_gray, -1, kernel_avg)
    sobelX = cv2.filter2D(img_gray, -1, sobel_x)
    sobelY = cv2.filter2D(img_gray, -1, sobel_y)
    prewittX = cv2.filter2D(img_gray, -1, prewitt_x)
    prewittY = cv2.filter2D(img_gray, -1, prewitt_y)
    laplance = cv2.filter2D(img_gray, -1, laplace_kernel)

    img_set = [img_gray, img_avg, sobelX, sobelY, prewittX, prewittY, laplance]
    img_title = ['Original', 'Average_kernel', 'Sobel_x', 'Sobel_y', 'Prewitt_x', 'Prewitt_y', 'laplance']

    display(img_set, img_title)


def display(img_set, img_title):
    for i in range(len(img_set)):
        plt.subplot(4, 2, i+1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()