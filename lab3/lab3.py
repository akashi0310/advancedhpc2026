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

#_____________________________________________________________
block_sizes = [32, 64, 128, 256, 512, 1024]
execution_times = []

src_gpu = cuda.to_device(pixels) 
dst_gpu = cuda.device_array_like(src_gpu)
for threads_per_block in block_sizes:
    blocks = (
        len(pixels) + threads_per_block - 1
    ) // threads_per_block

    start_time = time.perf_counter()

    grayscale_gpu[blocks, threads_per_block](src_gpu, dst_gpu)
    cuda.synchronize()  # Wait until the GPU finishes

    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # milliseconds

    execution_times.append(execution_time)

    print(
        f"Block size: {threads_per_block}, "
        f"Time: {execution_time:.4f} ms"
    )

plt.plot(block_sizes, execution_times, marker="o", linewidth=2)

plt.title("CUDA Block Size vs Execution Time")
plt.xlabel("Threads per Block")
plt.ylabel("Execution Time (ms)")
plt.xticks(block_sizes)
plt.grid(True)
plt.show()
