import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Linear Operation ----------------------------#
def linear_operation1(img, alpha=1.5, beta=25):
    new_img = alpha * img.astype(np.float32) + beta
    return np.clip(new_img, 0, 255).astype(np.uint8)

def linear_operation2(img, alpha=1.1, beta=10):
    new_img = alpha * img.astype(np.float32) - beta
    return np.clip(new_img, 0, 255).astype(np.uint8)

# --------------------- Non-linear Operation (Gamma) ----------------#
def gamma_operation1(img, gamma=0.6):
    norm = img.astype(np.float32) / 255.0
    corrected = np.power(norm, float(gamma))
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def gamma_operation2(img, gamma=0.2):
    norm = img.astype(np.float32) / 255.0
    corrected = np.power(norm, float(gamma))
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

# ------------------- Divide ----------------------------------------#
def divide_image(img, s):
    h, w = img.shape
    ph, pw = h // s, w // s
    img = img[:ph * s, :pw * s]    
    parts = []
    for i in range(s):
        for j in range(s):
            y1, y2 = i * ph, (i + 1) * ph
            x1, x2 = j * pw, (j + 1) * pw
            parts.append(img[y1:y2, x1:x2])
    return parts, ph, pw

# ------------------- Combine -----------------------------------------#
def combine_image(parts, s, ph, pw):
    out = np.zeros((s * ph, s * pw), dtype=np.uint8)
    k = 0
    for i in range(s):
        for j in range(s):
            y1, y2 = i * ph, (i + 1) * ph
            x1, x2 = j * pw, (j + 1) * pw
            out[y1:y2, x1:x2] = parts[k]
            k += 1
    return out

# ------------------ HE -----------------------------------------------#
def HE(img):
    return cv2.equalizeHist(img)

# ------------------ AHE -----------------------------------------------#
def AHE(img):
    clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# ------------------- CLAHE -------------------------------------------#
def CLAHE(img, clip=2.0):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    return clahe.apply(img)

# --------------- AHE + Bilinear ---------------------------------------#
def AHE_bilinear(img):
    ahe_img = AHE(img)
    h, w = img.shape
    w2, h2 = max(1, w // 2), max(1, h // 2)
    small = cv2.resize(ahe_img, (w2, h2), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return ahe_img, up

# ---------- Main -----------------------------------------------------#
def main():
    img_path = "/home/enoconda/Desktop/DIP/Image/smogImg.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    s = 3
    parts, ph, pw = divide_image(img, s)

    processed = []
    op_names = []

    # --------- Apply 4 Operations on the tiles --------- #
    for i, p in enumerate(parts):
        if i % 4 == 0:
            out = linear_operation1(p, alpha=1.6, beta=20)
            op_names.append("Linear Operation 1")
        elif i % 4 == 1:
            out = linear_operation2(p, alpha=1.6, beta=20)
            op_names.append("Linear Operation 2")
        elif i % 4 == 2:
            out = gamma_operation1(p, gamma=0.7)
            op_names.append("Gamma Operation 1")
        else:
            out = gamma_operation2(p, gamma=0.7)
            op_names.append("Gamma Operation 2")
        processed.append(out)

    # --------- Reconstruct the image --------- #
    combined = combine_image(processed, s, ph, pw)

    # --------- Prepare Single Figure for Linear & Gamma Outputs --------- #
    plt.figure(figsize=(12, 6))
    plt.subplot(231)
    plt.imshow(processed[0], cmap='gray')
    plt.title("Linear Operation 1")
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(processed[1], cmap='gray')
    plt.title("Linear Operation 2")
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(processed[2], cmap='gray')
    plt.title("Gamma Operation 1")
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(processed[3], cmap='gray')
    plt.title("Gamma Operation 2")
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(combined, cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --------- HE / AHE / CLAHE Outputs Figure --------- #
    he = HE(img)
    ahe = AHE(img)
    clahe2 = CLAHE(img, clip=2.0)
    clahe4 = CLAHE(img, clip=4.0)
    ahe_only, ahe_bi = AHE_bilinear(img)

    plt.figure(figsize=(12, 8))
    plt.subplot(231); plt.imshow(he, cmap='gray'); plt.title('HE'); plt.axis('off')
    plt.subplot(232); plt.imshow(ahe, cmap='gray'); plt.title('AHE'); plt.axis('off')
    plt.subplot(233); plt.imshow(clahe2, cmap='gray'); plt.title('CLAHE clip=2'); plt.axis('off')
    plt.subplot(234); plt.imshow(clahe4, cmap='gray'); plt.title('CLAHE clip=4'); plt.axis('off')
    plt.subplot(235); plt.imshow(ahe_bi, cmap='gray'); plt.title('AHE + Bilinear'); plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
