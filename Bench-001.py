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

def format_duration(duration):
    return f"{duration * 1000:.6f} ms"

def test_device(device, matrix_sizes, use_tensor_cores=False):
    """Run the matrix multiplication test on the given device.
    
    For each matrix size, a random square matrix is created and multiplied 50 times.
    Returns a list of tuples: (matrix_size, total_duration).
    """
    results = []
    for size in matrix_sizes:
        # Create two random matrices on the given device.
        mat_a = torch.randn(size, size, device=device, dtype=torch.float64)
        mat_b = torch.randn(size, size, device=device, dtype=torch.float64)
        if use_tensor_cores:
            # Convert to half precision for Tensor Core usage.
            mat_a = mat_a.to(torch.float16)
            mat_b = mat_b.to(torch.float16)
        # Warm-up runs
        for _ in range(5):
            torch.matmul(mat_a, mat_b)
        # Timed runs
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
    print("----------------------------------------")

    # Step 1: Ask user which computation method(s) to run.
    print("\nSelect computation method:")
    print("1. CPU only")
    print("2. GPU using CUDA (default precision, single GPU)")
    print("3. GPU using Tensor Cores (half precision, single GPU)")
    print("4. GPU (CUDA) and Tensor Cores (single GPU)")
    print("5. All (CPU, GPU (CUDA), GPU (Tensor Cores))")
    print("6. Multi-GPU test (specify comma-separated GPU indices)")
    try:
        benchmark_type = int(input("\nEnter your choice (1-6): "))
    except ValueError:
        print("Invalid input. Defaulting to CPU only.")
        benchmark_type = 1

    # Step 2: Select matrix sizes.
    max_power = 17  # maximum power of 2 to consider.
    print("\nSelect a matrix size (power of 2):")
    # Show options from 2^6 up to 2^(max_power)
    for i in range(6, max_power + 1):
        print(f"{i - 5}. {2 ** i}")
    try:
        user_choice = int(input("\nEnter your choice (number corresponding to 2^(n)): "))
    except ValueError:
        user_choice = 6  # default
    user_choice = min(user_choice, max_power - 5)
    user_size = 2 ** (user_choice + 5)
    # Build a list of matrix sizes from 2^6 up to the selected size.
    matrix_sizes = [2 ** i for i in range(6, user_size.bit_length() + 1)]

    # Step 3: Run tests based on the selected benchmark type.
    cpu_results = None
    gpu_results = None
    gpu_tc_results = None
    multi_gpu_results = None  # For Option 6

    # Option 1 (or part of Option 5): CPU test.
    if benchmark_type in [1, 5]:
        cpu_device = torch.device("cpu")
        print(f"\nRunning tests on CPU ({torch.get_num_threads()} threads)...")
        cpu_results = test_device(cpu_device, matrix_sizes)

    # Options 2, 3, 4, or 5: Single GPU tests.
    if benchmark_type in [2, 3, 4, 5]:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # If more than one GPU, let user select one.
            if num_gpus > 1:
                print("\nMultiple GPUs detected:")
                for i in range(num_gpus):
                    print(f"  {i}: {torch.cuda.get_device_name(i)}")
                gpu_choice = input("Enter the GPU index to use (default 0): ").strip()
                try:
                    device_index = int(gpu_choice) if gpu_choice != "" else 0
                    if device_index < 0 or device_index >= num_gpus:
                        print("Invalid GPU index. Defaulting to GPU 0.")
                        device_index = 0
                except ValueError:
                    print("Invalid input. Defaulting to GPU 0.")
                    device_index = 0
            else:
                device_index = 0
            gpu_device = torch.device(f"cuda:{device_index}")
            if benchmark_type in [2, 4, 5]:
                print(f"\nRunning tests on GPU using CUDA (Device {device_index}: {torch.cuda.get_device_name(gpu_device)})...")
                gpu_results = test_device(gpu_device, matrix_sizes)
            if benchmark_type in [3, 4, 5]:
                print(f"\nRunning tests on GPU using Tensor Cores (Device {device_index}: {torch.cuda.get_device_name(gpu_device)})...")
                gpu_tc_results = test_device(gpu_device, matrix_sizes, use_tensor_cores=True)
        else:
            print("\nCUDA is not available; GPU tests will be skipped.")

    # Option 6: Multi-GPU test.
    if benchmark_type == 6:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print("\nAvailable GPUs:")
            for i in range(num_gpus):
                print(f"  {i}: {torch.cuda.get_device_name(i)}")
            gpu_indices_input = input("Enter the GPU indices to use (comma-separated, e.g. 0,1,2): ").strip()
            try:
                gpu_indices = [int(x.strip()) for x in gpu_indices_input.split(',') if x.strip().isdigit()]
                gpu_indices = [i for i in gpu_indices if 0 <= i < num_gpus]
                if not gpu_indices:
                    print("No valid GPU indices selected. Defaulting to GPU 0.")
                    gpu_indices = [0]
            except Exception:
                print("Invalid input. Defaulting to GPU 0.")
                gpu_indices = [0]
            multi_gpu_results = {}
            for gpu_idx in gpu_indices:
                device = torch.device(f"cuda:{gpu_idx}")
                print(f"\nRunning tests on GPU {gpu_idx} using CUDA (Device {gpu_idx}: {torch.cuda.get_device_name(device)})...")
                multi_gpu_results[gpu_idx] = test_device(device, matrix_sizes)
        else:
            print("\nCUDA is not available; multi-GPU test skipped.")

    # Step 4: Display results.
    if benchmark_type == 6 and multi_gpu_results is not None:
        print("\nMatrix Multiplication Test Results (Multi-GPU):")
        # Build header with one column per GPU.
        header = "Size".ljust(10)
        selected_gpus = sorted(multi_gpu_results.keys())
        for gpu_idx in selected_gpus:
            header += f"GPU {gpu_idx} Time".ljust(28)
        header += "Ranking (Fastest -> Slowest)"
        print(header)
        for i, size in enumerate(matrix_sizes):
            line = f"{size:<10}"
            durations = []
            for gpu_idx in selected_gpus:
                duration = multi_gpu_results[gpu_idx][i][1]
                durations.append((gpu_idx, duration))
                line += f"{format_duration(duration):<28}"
            # Sort GPUs by their duration (fastest first)
            sorted_durations = sorted(durations, key=lambda x: x[1])
            ranking_str = " -> ".join([f"GPU {gpu}" for gpu, _ in sorted_durations])
            # Compute time difference between fastest and slowest.
            time_diff = format_duration(sorted_durations[-1][1] - sorted_durations[0][1])
            ranking_str += f" (Diff: {time_diff})"
            line += ranking_str
            print(line)
    else:
        # Display results for options 1,2,3,4,5 in a common table.
        print("\nMatrix Multiplication Test Results:")
        header = f"{'Size':<10}{'CPU Time':<28}{'GPU Time (CUDA)':<28}{'GPU Time (Tensor Cores)':<28}{'Fastest Device'}"
        print(header)
        for i, size in enumerate(matrix_sizes):
            cpu_duration = format_duration(cpu_results[i][1]) if cpu_results else "N/A"
            gpu_duration = format_duration(gpu_results[i][1]) if gpu_results else "N/A"
            gpu_tc_duration = format_duration(gpu_tc_results[i][1]) if gpu_tc_results else "N/A"
            durations = []
            labels = []
            if cpu_results:
                durations.append(cpu_results[i][1])
                labels.append("CPU")
            if gpu_results:
                durations.append(gpu_results[i][1])
                labels.append("GPU (CUDA)")
            if gpu_tc_results:
                durations.append(gpu_tc_results[i][1])
                labels.append("GPU (Tensor Cores)")
            if len(durations) > 0:
                min_index = np.argmin(durations)
                fastest = labels[min_index]
                diff = format_duration(abs(durations[min_index] - max(durations)))
            else:
                fastest = "N/A"
                diff = "N/A"
            print(f"{size:<10}{cpu_duration:<28}{gpu_duration:<28}{gpu_tc_duration:<28}{fastest} by {diff}")

    print("\nThank you for running the Matrix Multiplication Performance Test!")
    print("Written by: CHAT-GPT4, on 04/30/2023.")

if __name__ == "__main__":
    main()
