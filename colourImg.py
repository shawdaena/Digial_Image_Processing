import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_img(h, w, r = 0, g = 0, b=0):
    img = np.zeros((h,w,3), dtype = np.uint8)
    img[:,:,0] = r
    img[:,:,1] = g
    img[:,:,2] = b
    return img

def main():
    h,w = 200,200
    imgs = []

    gray_vals = [0,110,180,240,255]
    for i in gray_vals:
        imgs.append(color_img(h, w,i,i,i))

    red_vals = [0,110,180,240,255]
    for i in red_vals:
        imgs.append(color_img(h, w,i,0,0))

    gre_vals = [0,110,180,240,255]
    for i in gre_vals:
        imgs.append(color_img(h, w,0,i,0))

    blue_vals = [0,110,180,240,255]
    for i in blue_vals:
        imgs.append(color_img(h, w,0,0,i))

    gre_blue_vals = [0,110,180,240,255]
    for i in gre_blue_vals:
        imgs.append(color_img(h, w,0,i,i))

    red_blue_vals = [0,110,180,240,255]
    for i in red_blue_vals:
        imgs.append(color_img(h, w,i,0,i))

    red_gre_vals = [0,110,180,240,255]
    for i in red_gre_vals:
        imgs.append(color_img(h, w,i,i,0))


    plt.figure(figsize=(10,10))
    for i, img in enumerate(imgs):
        plt.subplot(7,5,i+1)
        plt.imshow(img)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()