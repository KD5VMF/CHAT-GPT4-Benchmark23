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

    # Step 1: Ask user what to use for benchmarking
    print("\nSelect computation method:")
    print("1. CPU only")
    print("2. GPU using CUDA (default precision)")
    print("3. GPU using Tensor Cores (half precision)")
    print("4. GPU (CUDA) and Tensor Cores")
    print("5. All (CPU, CUDA, Tensor Cores)")

    benchmark_type = int(input("\nEnter your choice (1/2/3/4/5): "))

    # Step 2: Select matrix size
    max_power = 17
    print("\nSelect a matrix size (power of 2):")
    for i in range(6, max_power + 1):
        print(f"{i - 5}. {2 ** i}")

    user_choice = int(input("\nEnter your choice: "))
    user_choice = min(user_choice, max_power - 5)
    user_size = 2 ** (user_choice + 5)
    matrix_sizes = [2 ** i for i in range(6, user_size.bit_length() + 1)]

    # Step 3: Run tests based on the user's choice
    cpu_results = None
    gpu_results = None
    gpu_tc_results = None

    if benchmark_type == 1 or benchmark_type == 5:
        # Run CPU test
        cpu_device = torch.device("cpu")
        print(f"\nRunning tests on CPU: {torch.get_num_threads()} threads")
        cpu_results = test_device(cpu_device, matrix_sizes)

    if benchmark_type in [2, 3, 4, 5] and torch.cuda.is_available():
        gpu_device = torch.device("cuda")

        if benchmark_type in [2, 4, 5]:
            # Run GPU (CUDA) test
            print(f"\nRunning tests on GPU using CUDA: {torch.cuda.get_device_name(gpu_device)}")
            gpu_results = test_device(gpu_device, matrix_sizes)

        if benchmark_type in [3, 4, 5]:
            # Run GPU (Tensor Cores) test
            print(f"Running tests on GPU using Tensor Cores: {torch.cuda.get_device_name(gpu_device)}")
            gpu_tc_results = test_device(gpu_device, matrix_sizes, use_tensor_cores=True)

    elif benchmark_type in [2, 3, 4, 5]:
        print("\nCUDA is not available, GPU test skipped")

    # Step 4: Display results
    print("\nMatrix multiplication test results:")
    header = f"{'Size':<8}{'CPU Time':<24}{'GPU Time (CUDA)':<24}{'GPU Time (Tensor Cores)':<24}{'Faster Device'}"
    print(header)

    for i, size in enumerate(matrix_sizes):
        cpu_duration = format_duration(cpu_results[i][1]) if cpu_results else "N/A"
        gpu_duration = format_duration(gpu_results[i][1]) if gpu_results else "N/A"
        gpu_tc_duration = format_duration(gpu_tc_results[i][1]) if gpu_tc_results else "N/A"

        durations = []
        device_labels = []

        if cpu_results:
            durations.append(cpu_results[i][1])
            device_labels.append("CPU")
        if gpu_results:
            durations.append(gpu_results[i][1])
            device_labels.append("GPU (CUDA)")
        if gpu_tc_results:
            durations.append(gpu_tc_results[i][1])
            device_labels.append("GPU (Tensor Cores)")

        if len(durations) > 1:
            # Determine the fastest device
            min_index = np.argmin(durations)
            faster_device = device_labels[min_index]
            time_difference = format_duration(abs(durations[min_index] - max(durations)))
        else:
            # Only one type of result available
            faster_device = device_labels[0] if device_labels else "N/A"
            time_difference = "N/A"

        print(f"{size:<8}{cpu_duration:<24}{gpu_duration:<24}{gpu_tc_duration:<24}{faster_device} by {time_difference}")

    print("\nThank you for running the Matrix Multiplication Performance Test!")
    print("If you have any questions or need further assistance, please don't hesitate to ask.")
    print("Written by: CHAT-GPT4, on 04/30/2023.")

if __name__ == "__main__":
    main()
