import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from heapq import heappush, heappop


# --------------------------
# Step 6: Compression + Reconstruction
# --------------------------
def main():
    # Step 1: Load image (grayscale)
    img = cv2.imread("/home/enoconda/Desktop/DIP/Image/LenaImg.png", cv2.IMREAD_GRAYSCALE).astype(float)
    h, w = img.shape
    print("Image name", img)
    print(f"Original size: {h*w*8} bits")

    # Pad image to ensure both dimensions are even for DWT
    padded_h = h if h % 2 == 0 else h + 1
    padded_w = w if w % 2 == 0 else w + 1
    padded_img = np.zeros((padded_h, padded_w), dtype=float)
    padded_img[:h, :w] = img

    # Step 2: Apply 2D DWT to padded image
    LL, LH, HL, HH = haar_dwt_2d(padded_img)

    # Step 3–4: Quantize + Threshold
    # Using the same aggressive quantization as last successful run (q=(1,8,8,16), threshold=2)
    LLq, LHq, HLq, HHq = quantize_subbands(LL, LH, HL, HH, q=(1,8,8,16), threshold=2)

    # Step 5a: Run-Length Encoding
    coeffs = np.concatenate([LLq.flatten(), LHq.flatten(), HLq.flatten(), HHq.flatten()])
    rle = run_length_encode(coeffs)

    # Step 5b: Huffman Encoding
    # Count frequencies
    freqs = {}
    for val, cnt in rle:
        freqs[val] = freqs.get(val, 0) + cnt
    tree = build_huffman_tree(freqs)
    codes = build_codes(tree)
    bitstream = "".join([codes[val] * cnt for val, cnt in rle])

    # Step 6: Compute compression stats
    compressed_bits = len(bitstream)
    ratio = (h*w*8) / compressed_bits # Use original image dimensions for ratio calculation
    print(f"Compressed bits: {compressed_bits}")
    print(f"Compression Ratio: {ratio:.2f}")

    # Decompression (inverse)
    LLd, LHd, HLd, HHd = dequantize_subbands(LLq, LHq, HLq, HHq, q=(1,8,8,16))
    recon_padded = haar_idwt_2d(LLd, LHd, HLd, HHd)
    recon_padded = np.clip(recon_padded, 0, 255).astype(np.uint8)

    # Crop the reconstructed image back to original dimensions
    recon = recon_padded[:h, :w]

    # cv2.imwrite("dwt_reconstructed.png", recon)
    # print("Reconstructed image saved as dwt_reconstructed.png")

    img_set = [img.astype(np.uint8), recon]
    titles = ['Original Image', 'DWT Reconstructed Image']
    display(img_set, titles)

# --------------------------
# Step 1: 1D and 2D Haar DWT
# --------------------------
def haar_dwt_1d(signal):
    N = len(signal)
    approx, detail = [], []
    for i in range(0, N, 2):
        a = (signal[i] + signal[i+1]) / np.sqrt(2)
        d = (signal[i] - signal[i+1]) / np.sqrt(2)
        approx.append(a)
        detail.append(d)
    return np.array(approx), np.array(detail)

def haar_idwt_1d(approx, detail):
    N = len(approx)
    signal = np.zeros(2 * N)
    for i in range(N):
        signal[2*i] = (approx[i] + detail[i]) / np.sqrt(2)
        signal[2*i+1] = (approx[i] - detail[i]) / np.sqrt(2) # Fixed: Changed approx[i+1] to detail[i]
    return signal

def haar_dwt_2d(image):
    rows_approx, rows_detail = [], []
    for row in image:
        a, d = haar_dwt_1d(row)
        rows_approx.append(a)
        rows_detail.append(d)
    rows_approx = np.array(rows_approx)
    rows_detail = np.array(rows_detail)

    cols_approx, cols_detail = [], []
    for col in rows_approx.T:
        a, d = haar_dwt_1d(col)
        cols_approx.append(a)
        cols_detail.append(d)
    LL = np.array(cols_approx).T
    LH = np.array(cols_detail).T

    cols_approx, cols_detail = [], []
    for col in rows_detail.T:
        a, d = haar_dwt_1d(col)
        cols_approx.append(a)
        cols_detail.append(d)
    HL = np.array(cols_approx).T
    HH = np.array(cols_detail).T

    return LL, LH, HL, HH

def haar_idwt_2d(LL, LH, HL, HH):
    rows_approx = []
    for a_col, d_col in zip(LL.T, LH.T):
        row = haar_idwt_1d(a_col, d_col)
        rows_approx.append(row)
    rows_approx = np.array(rows_approx).T

    rows_detail = []
    for a_col, d_col in zip(HL.T, HH.T):
        row = haar_idwt_1d(a_col, d_col)
        rows_detail.append(row)
    rows_detail = np.array(rows_detail).T

    recon = []
    for a_row, d_row in zip(rows_approx, rows_detail):
        row = haar_idwt_1d(a_row, d_row)
        recon.append(row)
    return np.array(recon)


# Step 3–4: Quantization + Threshold
def quantize_subbands(LL, LH, HL, HH, q=(1,4,4,8), threshold=2):
    LLq = np.round(LL / q[0])
    LHq = np.round(LH / q[1])
    HLq = np.round(HL / q[2])
    HHq = np.round(HH / q[3])

    # Threshold: zero out small coefficients
    LLq[np.abs(LLq) < threshold] = 0
    LHq[np.abs(LHq) < threshold] = 0
    HLq[np.abs(HLq) < threshold] = 0
    HHq[np.abs(HHq) < threshold] = 0

    return LLq, LHq, HLq, HHq

def dequantize_subbands(LLq, LHq, HLq, HHq, q=(1,4,4,8)):
    LL = LLq * q[0]
    LH = LHq * q[1]
    HL = HLq * q[2]
    HH = HHq * q[3]
    return LL, LH, HL, HH


# Step 5: Entropy Coding (RLE + Huffman)
def run_length_encode(arr):
    arr = arr.flatten()
    rle = []
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            count += 1
        else:
            rle.append((arr[i-1], count))
            count = 1
    rle.append((arr[-1], count))
    return rle

def run_length_decode(rle):
    arr = []
    for val, cnt in rle:
        arr.extend([val]*cnt)
    return np.array(arr)

# --- Huffman Encoding ---
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freqs):
    heap = []
    for sym, freq in freqs.items():
        heappush(heap, Node(sym, freq))
    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook={}):
    if node is None: return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
        return
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

def display(img_set, titles, cols=2):
    rows = (len(img_set) + cols - 1) // cols
    
    for i, img in enumerate(img_set):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()