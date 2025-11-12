import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Step 1: Read grayscale image
    img_path = "/home/enoconda/Desktop/DIP/Image/LenaImg.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print(f"Original size: {h*w*8} bits")

    # Step 2: Apply Huffman compression
    bitstream, codes, freqs = huffman_encode(img)
    compressed_bits = len(bitstream)
    ratio = (h*w*8) / compressed_bits

    print(f"Compressed bits: {compressed_bits}")
    print(f"Compression Ratio: {ratio:.2f}")

    # Step 3: Decode back
    recon = huffman_decode(bitstream, codes, img.shape).astype(np.uint8)
    # cv2.imwrite("huffman_reconstructed.png", recon)
    # print("Reconstructed image saved as huffman_reconstructed.png")

    img_set = [img, recon]
    titles = ['Original Image', 'Huffman Reconstructed Image']
    display(img_set, titles)
    

# -----------------------------
# Huffman Coding Implementation
# -----------------------------
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
    for symbol, freq in freqs.items():
        heappush(heap, Node(symbol, freq))
    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heappush(heap, merged)
    return heap[0]

def build_huffman_codes(node, code='', codebook=None):
    if codebook is None:
        codebook = {}
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = code
    build_huffman_codes(node.left, code + '0', codebook)
    build_huffman_codes(node.right, code + '1', codebook)
    return codebook

def huffman_encode(image):
    """Compress image using Huffman coding."""
    flat = image.flatten().astype(int)
    # Count frequencies
    freqs = {}
    for val in flat:
        freqs[val] = freqs.get(val, 0) + 1
    # Build Huffman tree and codes
    root = build_huffman_tree(freqs)
    codes = build_huffman_codes(root)
    # Encode the image
    bitstream = ''.join(codes[val] for val in flat)
    return bitstream, codes, freqs

def huffman_decode(bitstream, codes, shape):
    """Decode Huffman bitstream."""
    inv_codes = {v: k for k, v in codes.items()}
    decoded = []
    code = ''
    for bit in bitstream:
        code += bit
        if code in inv_codes:
            decoded.append(inv_codes[code])
            code = ''
    return np.array(decoded).reshape(shape)

def display(img_set, titles, cols = 2):
    rows = (len(img_set) + cols - 1) // cols
   

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