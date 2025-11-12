import cv2
import numpy as np
import math
from heapq import heappush, heappop
import matplotlib.pyplot as plt

# ---------------------- MAIN -------------------------------

def main():
    image_path = "/home/enoconda/Desktop/DIP/Image/LenaImg.png"  
    print("Image name:", image_path)
    bitstream, codes, h, w, ph, pw = compress_image(image_path)
    dct_reconstructed =  decompress_image(bitstream, codes, h, w, ph, pw)

    display_images = [cv2.imread(image_path), dct_reconstructed]
    titles = ['Original Image', 'DCT Reconstructed Image']
    display(display_images, titles)

# --------- DCT IMPLEMENTATION (1D then 2D) ----------------

def dct_1d(vector):
    """Compute 1D DCT (Type-II) manually."""
    N = len(vector)
    result = np.zeros(N)
    for u in range(N):
        alpha = math.sqrt(1/N) if u == 0 else math.sqrt(2/N)
        sum_val = 0
        for x in range(N):
            sum_val += vector[x] * math.cos(((2*x + 1) * u * math.pi) / (2 * N))
        result[u] = alpha * sum_val
    return result

def dct_2d(block):
    """Compute 2D DCT by applying 1D DCT to rows and then columns."""
    temp = np.apply_along_axis(dct_1d, axis=0, arr=block)
    result = np.apply_along_axis(dct_1d, axis=1, arr=temp)
    return result

def idct_1d(vector):
    """Compute 1D Inverse DCT (Type-III)."""
    N = len(vector)
    result = np.zeros(N)
    for x in range(N):
        sum_val = 0
        for u in range(N):
            alpha = math.sqrt(1/N) if u == 0 else math.sqrt(2/N)
            sum_val += alpha * vector[u] * math.cos(((2*x + 1) * u * math.pi) / (2 * N))
        result[x] = sum_val
    return result

def idct_2d(block):
    """Compute 2D Inverse DCT by applying 1D IDCT to rows and columns."""
    temp = np.apply_along_axis(idct_1d, axis=0, arr=block)
    result = np.apply_along_axis(idct_1d, axis=1, arr=temp)
    return result

# ----------------- QUANTIZATION MATRIX ---------------------

Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

# ------------------ ZIGZAG SCAN ----------------------------

zigzag_indices = [
 (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
 (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
 (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
 (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
 (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
 (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
 (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
 (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

def zigzag(block):
    return np.array([block[i,j] for (i,j) in zigzag_indices])

def inverse_zigzag(arr):
    block = np.zeros((8,8))
    for idx, (i,j) in enumerate(zigzag_indices):
        block[i,j] = arr[idx]
    return block

# ---------------- RUN-LENGTH ENCODING ----------------------

def run_length_encode(arr):
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
    for val, count in rle:
        arr.extend([val]*count)
    return np.array(arr)

# ---------------- HUFFMAN CODING ---------------------------

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freqs):
    heap = []
    for symbol, freq in freqs.items():
        heappush(heap, Node(symbol, freq))
    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = Node(None, n1.freq + n2.freq, n1, n2)
        heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook={}):
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
        return
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

# ---------------- COMPRESSION FUNCTION ---------------------

def compress_image(image_path):
    img = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:,:,0].astype(np.float32)

    h, w = Y.shape
    padded_h = (h + 7) // 8 * 8
    padded_w = (w + 7) // 8 * 8
    padded = np.zeros((padded_h, padded_w), np.float32)
    padded[:h, :w] = Y

    blocks = []
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            block = padded[i:i+8, j:j+8] - 128
            dct_block = dct_2d(block)
            quant = np.round(dct_block / Q)
            zz = zigzag(quant)
            blocks.append(zz)

    # Flatten & encode
    all_coeffs = np.concatenate(blocks).astype(int)
    rle = run_length_encode(all_coeffs)

    # Huffman
    freqs = {}
    for val, count in rle:
        freqs[val] = freqs.get(val, 0) + count

    tree = build_huffman_tree(freqs)
    codes = build_codes(tree)
    bitstream = ''.join([codes[val]*count for val, count in rle])

    # Compression ratio (avg bits per pixel)
    original_bits = h * w * 8
    compressed_bits = len(bitstream)
    ratio = original_bits / compressed_bits

    print(f"Compression complete")
    print(f"Original bits: {original_bits}")
    print(f"Compressed bits: {compressed_bits}")
    print(f"Compression Ratio: {ratio:.2f}")

    return (bitstream, codes, h, w, padded_h, padded_w)

# ---------------- DECOMPRESSION FUNCTION -------------------

def decompress_image(bitstream, codes, h, w, ph, pw):
    inv_codes = {v:k for k,v in codes.items()}
    curr = ""
    symbols = []
    for bit in bitstream:
        curr += bit
        if curr in inv_codes:
            symbols.append(inv_codes[curr])
            curr = ""

    arr = np.array(symbols)
    total_blocks = (ph//8)*(pw//8)
    arr = arr[:total_blocks*64]
    blocks = np.split(arr, total_blocks)

    recon = np.zeros((ph, pw))
    idx = 0
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = inverse_zigzag(blocks[idx])
            dequant = block * Q
            idct_block = idct_2d(dequant) + 128
            recon[i:i+8, j:j+8] = np.clip(idct_block, 0, 255)
            idx += 1

    final = recon[:h, :w].astype(np.uint8)
    # cv2.imwrite("dct_reconstructed.png", final)
    # print("Decompression complete → dct_reconstructed.png saved.")
    return final

# ---------------- DISPLAY FUNCTION ----------------------------

def display(img_set, titles, cols=2):
    rows = (len(img_set) + cols - 1) // cols
    plt.figure(figsize=(10, 5 * rows))

    for i, img in enumerate(img_set):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()