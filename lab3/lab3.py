import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import time
img = plt.imread(r"C:\Users\lapla\OneDrive\Pictures\ace.jpg")
height, width, channels = img.shape
pixels = img[:, :, :3].reshape(height * width, 3)
plt.imshow(img)
plt.show()

#_______________________CPU__________________________________
def grayscale_cpu(pixels):
     result = np.empty_like(pixels)
     for i in range(len(pixels)):
          r, g, b = pixels[i]
          gray = (float(r) + float(g) + float(b)) / 3
          result[i] = (gray, gray, gray)
     return result
gray_pixels = grayscale_cpu(pixels)
start_time = time.time()
gray_image = gray_pixels.reshape(height, width, 3)
end_time = time.time()
print(f"CPU execution time: {end_time - start_time:.4f} seconds")
plt.imshow(gray_image)   
plt.show()

#______________________GPU__________________________________
@cuda.jit
def grayscale_gpu(src, dst):
    # where are we in the input?
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g


pixelCount = width * height
blockSize = 128
gridSize = pixelCount // blockSize
src_gpu = cuda.to_device(pixels) #This one is the input
dst_gpu = cuda.device_array_like(src_gpu) #This one is the output

start_time = time.time()
grayscale_gpu[gridSize, blockSize](src_gpu, dst_gpu)
end_time = time.time()
print(f"GPU execution time: {end_time - start_time:.4f} seconds")

result = dst_gpu.copy_to_host()
plt.imshow(result.reshape(height, width, 3))
plt.show()
