import time
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda


img = plt.imread(r"C:\Users\lapla\OneDrive\Pictures\ace.jpg")
height, width = img.shape[:2]
pixels = np.ascontiguousarray(img[:, :, :3].reshape(height * width, 3))

def grayscale_cpu(src):
    """CPU reference implementation. One row in src represents one pixel."""
    result = np.empty_like(src)
    for i in range(src.shape[0]):
        gray = (float(src[i, 0]) + float(src[i, 1]) + float(src[i, 2])) / 3.0
        result[i, 0] = result[i, 1] = result[i, 2] = gray
    return result


@cuda.jit
def grayscale_1d(src, dst, pixel_count):
    """One-dimensional grid: each thread handles one flattened pixel."""
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < pixel_count:
        gray = (float(src[i, 0]) + float(src[i, 1]) + float(src[i, 2])) / 3.0
        dst[i, 0] = dst[i, 1] = dst[i, 2] = gray


@cuda.jit
def grayscale_2d(src, dst, width, height):
    """Two-dimensional grid: each thread's (x, y) identifies one pixel."""
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if tidx < width and tidy < height:
        i = tidy * width + tidx
        gray = (float(src[i, 0]) + float(src[i, 1]) + float(src[i, 2])) / 3.0
        dst[i, 0] = dst[i, 1] = dst[i, 2] = gray

pixel_count = height * width
src_gpu = cuda.to_device(pixels)
dst_1d_gpu = cuda.device_array_like(src_gpu)
dst_2d_gpu = cuda.device_array_like(src_gpu)

start = time.perf_counter()
cpu_result = grayscale_cpu(pixels)
cpu_time_ms = (time.perf_counter() - start) * 1000.0
print(f"CPU time: {cpu_time_ms:.4f} ms")


block_sizes_2d = [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16), (32, 32)]
times_1d_ms = []
times_2d_ms = []
repetitions = 20

for block_size in block_sizes_2d:
    threads_1d = block_size[0] * block_size[1]
    blocks_1d = (pixel_count + threads_1d - 1) // threads_1d
    blocks_2d = (
        (width + block_size[0] - 1) // block_size[0],
        (height + block_size[1] - 1) // block_size[1],
    )

    grayscale_1d[blocks_1d, threads_1d](src_gpu, dst_1d_gpu, pixel_count)
    grayscale_2d[blocks_2d, block_size](src_gpu, dst_2d_gpu, width, height)
    cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repetitions):
        grayscale_1d[blocks_1d, threads_1d](src_gpu, dst_1d_gpu, pixel_count)
    cuda.synchronize()
    elapsed_1d_ms = (time.perf_counter() - start) * 1000.0 / repetitions
    times_1d_ms.append(elapsed_1d_ms)

    start = time.perf_counter()
    for _ in range(repetitions):
        grayscale_2d[blocks_2d, block_size](src_gpu, dst_2d_gpu, width, height)
    cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repetitions
    times_2d_ms.append(elapsed_ms)
    print(f"Block {block_size}: 1D={elapsed_1d_ms:.4f} ms, 2D={elapsed_ms:.4f} ms")

result_1d = dst_1d_gpu.copy_to_host()
result_2d = dst_2d_gpu.copy_to_host()
np.testing.assert_allclose(result_1d, result_2d, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(result_1d, cpu_result, rtol=1e-5, atol=1e-5)
plt.imshow(result_2d.reshape(height, width, 3))
plt.title("Grayscale image (2D CUDA kernel)")
plt.axis("off")
plt.show()

block_labels = [f"{x}x{y}" for x, y in block_sizes_2d]
plt.figure(figsize=(9, 5))
plt.plot(block_labels, times_1d_ms, marker="o", linewidth=2, label="1D grid")
plt.plot(block_labels, times_2d_ms, marker="s", linewidth=2, label="2D grid")
plt.title("1D vs 2D CUDA grayscale kernel time")
plt.xlabel("Threads per block (x × y for 2D)")
plt.ylabel("Average kernel time (ms)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

