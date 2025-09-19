import cv2
import numpy as np
import matplotlib.pyplot as plt


# Histogram Equalization Function

def hist_equalization(img):
    hist, _ = np.histogram(img.flatten(), 256, [0,256])
    pdf = hist / hist.sum()              # Probability distribution
    cdf = pdf.cumsum()                   # Cumulative distribution
    cdf_normalized = np.round(cdf * 255).astype(np.uint8)
    equalized_img = cdf_normalized[img]  # Map original pixels
    return equalized_img

# Image paths 

images = {
    "Dark Image": "/home/enoconda/Desktop/DIP/Image/DarkImg.png",
    "Bright Image": "/home/enoconda/Desktop/DIP/Image/BrightImg.png",
    "Normal Image": "/home/enoconda/Desktop/DIP/Image/flower.png"
}

# Loop through images
for img_title, path in images.items():
    # Load grayscale image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {path}")
        continue
    
    # Apply own histogram equalization
    my_eq = hist_equalization(img)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)
    
    # --------------------------
    # Plot images + histograms
    # --------------------------
    plt.figure(figsize=(16,8))
    
    # --- Images ---
    plt.subplot(3,3,1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{img_title} - Original")
    plt.axis("off")
    
    plt.subplot(3,3,2)
    plt.imshow(my_eq, cmap='gray')
    plt.title("My Equalized")
    plt.axis("off")
    
    plt.subplot(3,3,3)
    plt.imshow(clahe_img, cmap='gray')
    plt.title("CLAHE")
    plt.axis("off")
    
    # --- Histograms ---
    plt.subplot(3,3,4)
    plt.hist(img.flatten(), bins=256, color='black')
    plt.title("Original Histogram")
    
    plt.subplot(3,3,5)
    plt.hist(my_eq.flatten(), bins=256, color='black')
    plt.title("My Equalized Histogram")

    
    plt.subplot(3,3,6)
    plt.hist(clahe_img.flatten(), bins=256, color='black')
    plt.title("CLAHE Histogram")
    
    plt.tight_layout()
    plt.show()