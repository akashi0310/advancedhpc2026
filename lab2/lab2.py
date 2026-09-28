from numba import cuda
print(cuda.detect())

gpu = cuda.select_device(0)
print("GPU name:", gpu.name)