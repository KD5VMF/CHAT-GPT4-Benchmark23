import torch
import time
import os
import platform
import math
import numpy as np

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def nearest_power_of_2(n):
    return 2**round(math.log2(n))

def format_duration(duration):
    return f"{duration * 1000:.6f} ms"

def test_device(device, matrix_sizes, use_tensor_cores=False):
    results = []

    for size in matrix_sizes:
        mat_a = torch.randn(size, size, device=device, dtype=torch.float64)
        mat_b = torch.randn(size, size, device=device, dtype=torch.float64)

        if use_tensor_cores:
            mat_a = mat_a.to(torch.float16)
            mat_b = mat_b.to(torch.float16)

        # Warm-up
        for _ in range(5):
            torch.matmul(mat_a, mat_b)

        # Time the matrix multiplications
        start_time = time.perf_counter()
        for _ in range(50):
            torch.matmul(mat_a, mat_b)
        end_time = time.perf_counter()

        duration = end_time - start_time
        results.append((size, duration))

    return results

def main():
    clear_screen()

    print("Matrix Multiplication Performance Test")

    max_power = 17
    print("\nSelect a matrix size (power of 2):")
    for i in range(6, max_power + 1):
        print(f"{i - 5}. {2 ** i}")

    user_choice = int(input("\nEnter your choice: "))
    user_choice = min(user_choice, max_power - 5)
    user_size = 2 ** (user_choice + 5)
    matrix_sizes = [2 ** i for i in range(6, user_size.bit_length() + 1)]

    cpu_device = torch.device("cpu")
    print(f"\nRunning tests on CPU: {torch.get_num_threads()} threads")
    cpu_results = test_device(cpu_device, matrix_sizes)

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        print(f"Running tests on GPU using CUDA: {torch.cuda.get_device_name(gpu_device)}")
        gpu_results = test_device(gpu_device, matrix_sizes)

        if gpu_device.type == "cuda":
            print(f"Running tests on GPU using Tensor Cores: {torch.cuda.get_device_name(gpu_device)}")
            gpu_tc_results = test_device(gpu_device, matrix_sizes, use_tensor_cores=True)
        else:
            print("GPU does not support Tensor Cores")
            gpu_tc_results = None
    else:
        print("CUDA is not available, GPU test skipped")
        gpu_results = None
        gpu_tc_results = None

    print("\nMatrix multiplication test results:")
    print(f"{'Size':<8}{'CPU Time':<24}{'GPU Time (CUDA)':<24}{'GPU Time (Tensor Cores)':<24}{'Faster Device'}")

    for i, size in enumerate(matrix_sizes):
        cpu_duration = format_duration(cpu_results[i][1])
        gpu_duration = format_duration(gpu_results[i][1]) if gpu_results else "N/A"
        gpu_tc_duration = format_duration(gpu_tc_results[i][1]) if gpu_tc_results else "N/A"
        if gpu_results and gpu_tc_results:
            durations = [cpu_results[i][1], gpu_results[i][1], gpu_tc_results[i][1]]
            faster_device = ["CPU", "GPU (CUDA)", "GPU (Tensor Cores)"][np.argmin(durations)]
            time_difference = format_duration(abs(min(durations) - max(durations)))
        elif gpu_results:
            faster_device = "CPU" if cpu_results[i][1] < gpu_results[i][1] else "GPU (CUDA)"
            time_difference = format_duration(abs(cpu_results[i][1] - gpu_results[i][1]))
        else:
            faster_device = "N/A"
            time_difference = "N/A"
        print(f"{size:<8}{cpu_duration:<24}{gpu_duration:<24}{gpu_tc_duration:<24}{faster_device} by {time_difference}")

    print("\nThank you for running the Matrix Multiplication Performance Test!")
    print("If you have any questions or need further assistance, please don't hesitate to ask.")
    print("Written by: CHAT-GPT4, on 04/30/2023.")

if __name__ == "__main__":
    main()


