import cv2
import matplotlib.pyplot as plt

def main():
    # ---- Read image in grayscale ----
    img = cv2.imread("/home/enoconda/Desktop/DIP/Image/img1.png", 0)
    
    # ---- Apply Canny Edge Detection ----
    img_blur = cv2.GaussianBlur(img, (5,5), 1.5)
    edges = cv2.Canny(img_blur, threshold1=100, threshold2=200)
    
    # ---- Display Results ----
    plt.figure(figsize=(10,5))
    
    # Original image
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    # Canny edges
    plt.subplot(1,2,2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
