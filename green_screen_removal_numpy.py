
"""
# **Question 1:**  Given the following green sheet remove the green screen using numpy arrays only. And put image in front of image 2 as shown in last image.
"""


from PIL import Image
import numpy as np


inp = Image.open(r"C:\Users\FBR\Downloads\Assignment_2 (1)\Assignment_2\fr_jpg.jpg")
inp


inp = np.array(inp)


inp.shape


x = 0
for i in inp[645]:
  print(i)
  x = x+1
  if x == 900:
    break


print(inp.ravel())





news = Image.open(r"C:\Users\FBR\Downloads\Assignment_2 (1)\Assignment_2\bg_jpg.jpg")
news


news = np.array(news)


h, w = inp.shape[:2]
news = news[:h, :w]



# Step 3: Create green screen mask
# (tune the values as per your earlier findings: G=161, R/B~37)
green_mask = (inp[:, :, 1] > 150) & (inp[:, :, 0] < 80) & (inp[:, :, 2] < 80)



res_img = np.copy(news)
res_img[~green_mask] = inp[~green_mask]



final_image = Image.fromarray(res_img)
final_image.show()        # or final_image.save("output.png")









Image.fromarray(res_img)


"""
# **Question 2:** Gaussian Blurring and Downsampling with NumPy

kernel = np.array(
                   [[1,  4,  6,  4,  1],
                   [4, 16, 24, 16,  4],
                   [6, 24, 36, 24,  6],
                   [4, 16, 24, 16,  4],
                   [1,  4,  6,  4,  1]]) / 256.0


final_np = np.array(final_image)  # shape: (H, W, 3)



gray_img = (
    0.2989 * final_np[:, :, 0] +
    0.5870 * final_np[:, :, 1] +
    0.1140 * final_np[:, :, 2]
).astype(np.uint8)


gaussian_kernel = np.array(
    [[1,  4,  6,  4, 1],
     [4, 16, 24, 16, 4],
     [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4],
     [1,  4,  6,  4, 1]]
) / 256.0


gaussian_kernel = np.array(
    [[1,  4,  6,  4, 1],
     [4, 16, 24, 16, 4],
     [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4],
     [1,  4,  6,  4, 1]]
) / 256.0

def convolve_gray(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=float)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

blurred_img = convolve_gray(gray_img, gaussian_kernel)



def downsample(image):
    return image[::2, ::2]

downsampled_img = downsample(blurred_img)
 


def upsample(image, target_shape):
    h, w = target_shape
    upsampled = np.zeros((h, w), dtype=float)
    upsampled[::2, ::2] = image

    # Fill missing columns
    for i in range(0, h, 2):
        for j in range(1, w-1, 2):
            upsampled[i, j] = (upsampled[i, j-1] + upsampled[i, j+1]) / 2

    # Fill missing rows
    for i in range(1, h-1, 2):
        for j in range(w):
            upsampled[i, j] = (upsampled[i-1, j] + upsampled[i+1, j]) / 2

    return upsampled

upsampled_img = upsample(downsampled_img, gray_img.shape)



def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

error = mse(gray_img, upsampled_img)
print("Original shape:", gray_img.shape)
print("Downsampled shape:", downsampled_img.shape)
print("Reconstructed shape:", upsampled_img.shape)
print("Reconstruction Error (MSE):", error)



print("\nOriginal pixels at row 100:", gray_img[100, 100:110].astype(int))
print("Reconstructed pixels at row 100:", upsampled_img[100, 100:110].astype(int))





