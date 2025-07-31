"""
## Instructions
• All questions must be answered within a single notebook or .py file.

• Follow the file naming conventions: Name your submission file as RollNo.ipynb or RollNo.py (e.g., i22_xxxx_S.ipynb, where xxxx is your Roll Number and S is your Section).

• Ensure that your solutions are free of errors.

• Late submissions will not be accepted and will be given a zero.

• No external libraries are allowed for this assignment.

• Any form of plagiarism will result in a zero for both parties involved.

• AI-generated content is prohibited. Detection of such content will lead to a zero score.
"""


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
###Problem Description:
In this assignment, you will learn how to apply Gaussian blurring and perform downsampling on a grayscale image using only NumPy. You will also learn how to reconstruct the image from the downsampled version. This exercise will introduce the concept of convolution, a key operation in image processing.

##Task:
###Write a Python program using NumPy to:
Apply a Gaussian blur to an image (task1 output image) using a 5x5 kernel.
Downsample the image by a factor of 2.
Upsample the downsampled image to the original size.
Compare the reconstructed image with the original image.
Algorithm Explanation:
####Step 1: Convolution and Gaussian Blurring
Convolution is a mathematical operation where an image is processed by a filter (also called a kernel). The idea is to slide the kernel across the image, multiply the kernel's values by the underlying pixel values, and sum the result. This gives a weighted sum that replaces the original pixel in the output image.

In image processing, convolution is often used to apply effects like blurring, sharpening, and edge detection.

What is a kernel?

A kernel is a small matrix (e.g., 3x3 or 5x5) used to process the image. In this case, we’ll use a Gaussian kernel to smooth (blur) the image, reducing noise or fine details.
Here's a 5x5 Gaussian kernel you will use:

##kernel = np.array(
                   [[1,  4,  6,  4,  1],
                   [4, 16, 24, 16,  4],
                   [6, 24, 36, 24,  6],
                   [4, 16, 24, 16,  4],
                   [1,  4,  6,  4,  1]]) / 256.0
How does convolution work?

Slide the kernel over the image (left to right, top to bottom).
For each position of the kernel, multiply the kernel values by the image values it covers.
Sum these products to get the new pixel value.
Replace the original pixel with the new value.
Example of Convolution:
Let's say we have a 3x3 region of the image and we slide a 3x3 kernel over it.

**For this image region:**

### Image region:
                  [[12, 15, 20],
                  [18, 19, 25],
                  [10, 16, 14]]
###And this kernel:

                  [[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]]
To apply convolution, we multiply the kernel element-wise with the image region:


              (12 * 1) + (15 * 0) + (20 * -1) +
              (18 * 1) + (19 * 0) + (25 * -1) +
              (10 * 1) + (16 * 0) + (14 * -1)
Which gives the sum:


                12 + 0 - 20 + 18 + 0 - 25 + 10 + 0 - 14 = -19
**So, the output pixel value will be -19.**

#### Step 2: Gaussian Blurring
Apply a Gaussian blur to smooth the image. This helps reduce noise by averaging neighboring pixel values.
We’ll perform convolution using a 5x5 Gaussian kernel. Every pixel will be replaced with a weighted average of itself and its neighbors.
#### Step 3: Downsampling
After applying the blur, downsample the image by keeping only every second row and column.
This effectively reduces the image size by half.
#### Step 4: Upsampling
Upsample the image by restoring it to its original size. Place the downsampled values in the correct positions and fill in the missing pixels using interpolation.
** Steps to Solve the Problem:**
#####Convolution Implementation:

Write a function to apply convolution between the image and a kernel.
Loop through each pixel of the image, apply the kernel, and replace the pixel with the new value.
Gaussian Blur:

Implement convolution with the Gaussian kernel to smooth the image.
Downsampling:

Reduce the size of the image by taking every second pixel in both rows and columns.
Upsampling:

Expand the image back to its original size, inserting zeros and using neighboring pixels to fill the gaps.
Comparison:

Compare the reconstructed image to the original.
"""


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





