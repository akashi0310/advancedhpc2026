from numba import cuda
print(cuda.detect())

gpu = cuda.select_device(0)
print("GPU name:", gpu.name)

print("Multiprocessor count:", gpu.MULTIPROCESSOR_COUNT)
print("Compute Capability:",gpu.compute_capability)
if gpu.compute_capability == (8,6):
     print("Cuda core count:", gpu.MULTIPROCESSOR_COUNT * 128)

free_memory, total_memory = cuda.current_context().get_memory_info()
print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GiB")
print(f"Free GPU memory: {free_memory / (1024 ** 3):.2f} GiB")
