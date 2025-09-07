import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_histogram(img):
    h, w = img.shape
    pixel_array = np.zeros((256,), dtype=np.uint) 
    
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_array[pixel_value] += 1
            
    return pixel_array

def main():
    # ------- Read image -------
    img = cv2.imread("/home/enoconda/Desktop/DIP/Image/tulip.png", 0)
    
    # ------- Histogram -------
    hist = compute_histogram(img)
    
    # ------ PDF -------
    pdf = hist / hist.sum()
    
    # ------ CDF -------
    #cdf = np.cumsum(pdf)
    cdf = np.zeros_like(pdf, dtype=float)
    cdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cdf[i] = cdf[i-1] + pdf[i]
    
    # ------ Mapping Function -------
    cdf_min = cdf[np.nonzero(cdf)][0]   # first nonzero value
    mapping = np.round((cdf - cdf_min) / (1 - cdf_min) * 255).astype(np.uint8)
    
    # ------ Equalized Image -------
    h, w = img.shape
    equalized_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            equalized_img[i, j] = mapping[img[i, j]]
    
    # ------- Equalized Histogram -------
    eq_hist = compute_histogram(equalized_img)
    
    # ------- Show Results -------
    plt.figure(figsize=(12,8))
    
    # Original image
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    # Equalized image
    plt.subplot(2,2,2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title("Equalized Image")
    plt.axis("off")
    
    # Original histogram
    plt.subplot(2,2,3)
    plt.plot(hist, color='blue')
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    
    # Equalized histogram
    plt.subplot(2,2,4)
    plt.plot(eq_hist, color='red')
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
