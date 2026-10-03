import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

#________________6a______________________
@cuda.jit
def binarize_image(src, dst, width, height, threshold, white_value):

    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x < width and y < height:
        binary = (float(src[y, x, 0]) + float(src[y, x, 1]) + float(src[y, x, 2])) / 3.0

        if binary < threshold:
            dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = 0
        else:
            dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = white_value


image = plt.imread(r"C:\Users\lapla\OneDrive\Pictures\sakura.jpg")
image = np.ascontiguousarray(image[:, :, :3]).copy()
height, width, _ = image.shape
threshold = 128
white_value = 255
src_gpu = cuda.to_device(image)
dst_gpu = cuda.device_array_like(src_gpu)

block_size = (16, 16)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
)

binarize_image[grid_size, block_size](
    src_gpu, dst_gpu, width, height, threshold, white_value
)
cuda.synchronize()

binary_image = dst_gpu.copy_to_host()
plt.imshow(binary_image)
plt.title("Binary image")
plt.show()

#________________6b______________________
@cuda.jit
def brightness_image(src, dst, width, height,increase, brightness):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < width and y < height:
        for channel in range(3):
            value = float(src[y, x, channel])
            if increase:
                value += brightness
                if value > 255.0:
                    value = 255.0
            else:
                value -= brightness
                if value < 0.0:
                    value = 0.0

            dst[y, x, channel] = value

image = plt.imread(r"C:\Users\lapla\OneDrive\Pictures\sakura.jpg")
image = np.ascontiguousarray(image[:, :, :3]).copy()
height, width, _ = image.shape
brightness = 100
src_gpu = cuda.to_device(image)
dst_gpu = cuda.device_array_like(src_gpu)   
block_size = (16, 16)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
)

brightness_image[grid_size, block_size](
    src_gpu, dst_gpu, width, height, True, brightness
)
cuda.synchronize()

brightened_image = dst_gpu.copy_to_host()
plt.imshow(brightened_image)
plt.title("Brightened image")
plt.show()
