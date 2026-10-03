from numba import cuda
import matplotlib.pyplot as plt
import numpy as np

gaussian_filter = np.array([[0, 0, 1, 2, 1, 0, 0],
                            [0, 3, 13, 22, 13, 3, 0],
                            [1, 13, 59, 97, 59, 13, 1],
                            [2, 22, 97, 159, 97, 22, 2],
                            [1, 13, 59, 97, 59, 13, 1],
                            [0, 3, 13, 22, 13, 3, 0],
                            [0, 0, 1, 2, 1, 0, 0]], dtype=np.float32)
gaussian_filter = np.ascontiguousarray(gaussian_filter)
filter_sum = float(gaussian_filter.sum())

@cuda.jit
def gaussian_blur(src, dst, width, height, gaussian_filter, filter_sum):
     x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
     y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
     if x < width and y < height:
          sum_r = 0.0
          sum_g = 0.0
          sum_b = 0.0
          for ky in range(7):
               for kx in range(7):
                    pixel_x = min(max(x + kx - 3, 0), width - 1)
                    pixel_y = min(max(y + ky - 3, 0), height - 1)
                    weight = gaussian_filter[ky, kx]
                    sum_r += float(src[pixel_y, pixel_x, 0]) * weight
                    sum_g += float(src[pixel_y, pixel_x, 1]) * weight
                    sum_b += float(src[pixel_y, pixel_x, 2]) * weight

          dst[y, x, 0] = min(max(sum_r / filter_sum, 0.0), 255.0)
          dst[y, x, 1] = min(max(sum_g / filter_sum, 0.0), 255.0)
          dst[y, x, 2] = min(max(sum_b / filter_sum, 0.0), 255.0)

image = plt.imread(r"C:\Users\lapla\OneDrive\Pictures\sakura.jpg")
image = np.ascontiguousarray(image[:, :, :3]).copy()
height, width, _ = image.shape
src_gpu = cuda.to_device(image)
dst_gpu = cuda.device_array_like(src_gpu)
block_size = (16, 16)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
)

gaussian_blur[grid_size, block_size](
    src_gpu, dst_gpu, width, height, gaussian_filter, filter_sum
)
blurred_image = dst_gpu.copy_to_host()  
plt.imshow(blurred_image)
plt.title("Gaussian Blur")  
plt.show()  
