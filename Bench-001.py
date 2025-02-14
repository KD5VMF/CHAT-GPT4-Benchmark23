"""
Title: Matrix Multiplication Performance Test
Author: CHAT-GPT o3-mini-high
Date: 02/14/2025

About:
This program benchmarks matrix multiplication performance on CPU and GPU devices.
It provides a main menu for selecting:
  1. CPU-only testing,
  2. Single GPU testing, or
  3. Multi-GPU testing.
  
For GPU testing, users can choose between:
  - CUDA mode (default precision),
  - Tensor Cores mode (half precision), or
  - Both (to compare CUDA vs. Tensor Cores).

The program automatically checks for Tensor Core support (available on NVIDIA GPUs
with compute capability 7.0 or higher). If the selected GPU does not support Tensor Cores,
the program informs the user and skips the Tensor Core test.

The program includes detailed comments explaining each section for clarity.
"""

import torch
import time
import os
import platform
import math
import numpy as np

def clear_screen():
    """
    Clears the console screen.
    Uses 'cls' on Windows and 'clear' on other platforms.
    """
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def format_duration(duration):
    """
    Formats a duration (in seconds) to a string in milliseconds.
    """
    return f"{duration * 1000:.6f} ms"

def has_tensor_cores(device_index):
    """
    Checks if the GPU at the given device_index supports Tensor Cores.
    NVIDIA GPUs with compute capability >= 7.0 (Volta architecture and later) have Tensor Cores.
    
    Parameters:
        device_index (int): The index of the CUDA device.
        
    Returns:
        bool: True if the device supports Tensor Cores, otherwise False.
    """
    cap = torch.cuda.get_device_capability(device_index)
    return cap[0] >= 7

def test_device(device, matrix_sizes, use_tensor_cores=False):
    """
    Benchmarks matrix multiplication on the specified device for various matrix sizes.
    
    For each size in matrix_sizes:
      - Creates two random square matrices.
      - Performs 5 warm-up multiplications.
      - Times 50 multiplications.
    
    For CUDA devices, the function synchronizes before and after the timed loop to ensure
    accurate timing.
    
    Parameters:
        device (torch.device): The device (CPU or CUDA) on which to run the test.
        matrix_sizes (list): A list of matrix dimensions (e.g., [64, 128, ...]).
        use_tensor_cores (bool): If True, converts matrices to half precision (FP16) to
                                 potentially utilize Tensor Cores.
    
    Returns:
        list: A list of tuples, each containing (matrix_size, total_duration).
    """
    results = []
    for size in matrix_sizes:
        # Create two random matrices on the selected device with double precision.
        mat_a = torch.randn(size, size, device=device, dtype=torch.float64)
        mat_b = torch.randn(size, size, device=device, dtype=torch.float64)
        
        # If using Tensor Cores, convert matrices to half precision.
        if use_tensor_cores:
            mat_a = mat_a.to(torch.float16)
            mat_b = mat_b.to(torch.float16)
        
        # Perform 5 warm-up runs (not timed) to stabilize performance.
        for _ in range(5):
            torch.matmul(mat_a, mat_b)
        
        # Synchronize CUDA devices to ensure all warm-up operations are complete.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
        # Start timing.
        start_time = time.perf_counter()
        for _ in range(50):
            torch.matmul(mat_a, mat_b)
        # Synchronize CUDA to ensure all operations have finished before stopping the timer.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        
        # Record the total duration for this matrix size.
        results.append((size, end_time - start_time))
    return results

def main():
    """
    Main function that provides the user interface:
      - Displays the About information.
      - Allows selection of CPU-only, Single GPU, or Multi-GPU tests.
      - For GPU tests, prompts for the test mode (CUDA, Tensor Cores, or Both).
      - For Multi-GPU tests, allows entering a comma-separated list of GPU indices.
      - Runs the appropriate tests and displays the results.
    """
    clear_screen()
    print("Matrix Multiplication Performance Test")
    print("----------------------------------------")
    
    # --- Main Menu: Select the device type for benchmarking ---
    print("\nMain Menu:")
    print("1. CPU Only")
    print("2. Single GPU Test")
    print("3. Multi-GPU Test")
    try:
        main_choice = int(input("\nEnter your choice (1-3): "))
    except ValueError:
        main_choice = 1  # Default to CPU Only if invalid input
    
    # --- Matrix Size Selection ---
    # Allow the user to select the maximum matrix size (as a power of 2)
    max_power = 17  # Test up to 2^17
    print("\nSelect a matrix size (power of 2):")
    for i in range(6, max_power + 1):
        print(f"{i - 5}. {2 ** i}")
    try:
        user_choice = int(input("\nEnter your choice (number corresponding to 2^(n)): "))
    except ValueError:
        user_choice = 6  # Default if input is invalid
    user_choice = min(user_choice, max_power - 5)
    user_size = 2 ** (user_choice + 5)
    # Create a list of matrix sizes from 2^6 up to the chosen maximum.
    matrix_sizes = [2 ** i for i in range(6, user_size.bit_length() + 1)]
    
    # --- Option 1: CPU-Only Test ---
    if main_choice == 1:
        cpu_device = torch.device("cpu")
        print(f"\nRunning CPU test using {torch.get_num_threads()} threads...")
        cpu_results = test_device(cpu_device, matrix_sizes)
        print("\nCPU Test Results:")
        print(f"{'Size':<10}{'Time':<28}")
        for size, duration in cpu_results:
            print(f"{size:<10}{format_duration(duration):<28}")
    
    # --- Option 2: Single GPU Test ---
    elif main_choice == 2:
        if not torch.cuda.is_available():
            print("\nCUDA is not available.")
            return
        num_gpus = torch.cuda.device_count()
        
        # --- GPU Test Type Menu for Single GPU ---
        print("\nSelect GPU Test Type for Single GPU:")
        print("1. CUDA (default precision)")
        print("2. Tensor Cores (half precision)")
        print("3. Both (compare CUDA and Tensor Cores)")
        try:
            gpu_test_type = int(input("\nEnter your choice (1-3): "))
        except ValueError:
            gpu_test_type = 1
        
        # If more than one GPU is available, prompt for the GPU index.
        if num_gpus > 1:
            print("\nMultiple GPUs detected:")
            for i in range(num_gpus):
                print(f"  {i}: {torch.cuda.get_device_name(i)}")
            try:
                gpu_index = int(input("Enter GPU index to use (default 0): ") or "0")
            except ValueError:
                gpu_index = 0
        else:
            gpu_index = 0
        gpu_device = torch.device(f"cuda:{gpu_index}")
        gpu_name = torch.cuda.get_device_name(gpu_index)
        
        # Run the test based on the selected mode.
        if gpu_test_type == 1:
            print(f"\nRunning Single GPU Test (CUDA) on GPU {gpu_index}: {gpu_name}")
            gpu_results = test_device(gpu_device, matrix_sizes)
            print("\nSingle GPU Test Results (CUDA):")
            print(f"{'Size':<10}{'Time':<28}")
            for size, duration in gpu_results:
                print(f"{size:<10}{format_duration(duration):<28}")
        elif gpu_test_type == 2:
            # Check for Tensor Core support.
            if not has_tensor_cores(gpu_index):
                print(f"\nNo Tensor Cores found on GPU {gpu_index}: {gpu_name}.")
                return
            print(f"\nRunning Single GPU Test (Tensor Cores) on GPU {gpu_index}: {gpu_name}")
            gpu_tc_results = test_device(gpu_device, matrix_sizes, use_tensor_cores=True)
            print("\nSingle GPU Test Results (Tensor Cores):")
            print(f"{'Size':<10}{'Time':<28}")
            for size, duration in gpu_tc_results:
                print(f"{size:<10}{format_duration(duration):<28}")
        elif gpu_test_type == 3:
            print(f"\nRunning Single GPU Test (Both) on GPU {gpu_index}: {gpu_name}")
            gpu_results = test_device(gpu_device, matrix_sizes)
            if has_tensor_cores(gpu_index):
                gpu_tc_results = test_device(gpu_device, matrix_sizes, use_tensor_cores=True)
            else:
                print(f"\nNo Tensor Cores found on GPU {gpu_index}: {gpu_name}. Skipping Tensor Cores test.")
                gpu_tc_results = None
            print("\nSingle GPU Test Results (Comparison):")
            print(f"{'Size':<10}{'CUDA Time':<28}{'Tensor Cores Time':<28}{'Faster'}")
            for i, size in enumerate(matrix_sizes):
                cuda_time = gpu_results[i][1]
                if gpu_tc_results:
                    tensor_time = gpu_tc_results[i][1]
                    faster = "CUDA" if cuda_time < tensor_time else "Tensor Cores"
                    diff = format_duration(abs(cuda_time - tensor_time))
                    tensor_str = format_duration(tensor_time)
                else:
                    faster = "CUDA (only)"
                    diff = "N/A"
                    tensor_str = "N/A"
                print(f"{size:<10}{format_duration(cuda_time):<28}{tensor_str:<28}{faster} by {diff}")
    
    # --- Option 3: Multi-GPU Test ---
    elif main_choice == 3:
        if not torch.cuda.is_available():
            print("\nCUDA is not available.")
            return
        num_gpus = torch.cuda.device_count()
        
        # --- Multi-GPU Test Mode Menu ---
        print("\nSelect Multi-GPU Test Type:")
        print("1. CUDA (default precision)")
        print("2. Tensor Cores (half precision)")
        print("3. Both (separate tests for CUDA and Tensor Cores)")
        try:
            multi_test_type = int(input("\nEnter your choice (1-3): "))
        except ValueError:
            multi_test_type = 1
        
        # Prompt for GPU indices to use.
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
        # Build a dictionary mapping GPU indices to their device names.
        gpu_names = {idx: torch.cuda.get_device_name(idx) for idx in gpu_indices}
        
        # --- Multi-GPU Test for CUDA or Tensor Cores only ---
        if multi_test_type in [1, 2]:
            mode_str = "CUDA" if multi_test_type == 1 else "Tensor Cores"
            print(f"\nRunning Multi-GPU Test ({mode_str}) on GPUs: {', '.join(str(i) for i in gpu_indices)}")
            multi_results = {}
            for idx in gpu_indices:
                device = torch.device(f"cuda:{idx}")
                if multi_test_type == 1:
                    multi_results[idx] = test_device(device, matrix_sizes)
                else:
                    if has_tensor_cores(idx):
                        multi_results[idx] = test_device(device, matrix_sizes, use_tensor_cores=True)
                    else:
                        print(f"\nNo Tensor Cores found on GPU {idx}: {gpu_names[idx]}. Skipping this GPU for Tensor Cores test.")
                        multi_results[idx] = None
            # Display the multi-GPU results table.
            print(f"\nMulti-GPU Test Results ({mode_str}):")
            header = "Size".ljust(10)
            for idx in gpu_indices:
                header += f"GPU {idx} Time".ljust(28)
            header += "Winner".ljust(40) + "Diff (Fastest vs 2nd)".ljust(28)
            print(header)
            for i, size in enumerate(matrix_sizes):
                line = f"{size:<10}"
                durations = []
                for idx in gpu_indices:
                    result = multi_results[idx]
                    if result is not None:
                        duration = result[i][1]
                        durations.append((idx, duration))
                        line += f"{format_duration(duration):<28}"
                    else:
                        line += "No Tensor Cores".ljust(28)
                if len(durations) > 1:
                    # Sort by duration (lower is better)
                    sorted_durations = sorted(durations, key=lambda x: x[1])
                    winner_idx, winner_time = sorted_durations[0]
                    runner_up_idx, runner_up_time = sorted_durations[1]
                    diff = runner_up_time - winner_time
                    winner_str = f"{gpu_names[winner_idx]} (GPU {winner_idx})"
                    diff_str = format_duration(diff)
                elif durations:
                    winner_idx, _ = durations[0]
                    winner_str = f"{gpu_names[winner_idx]} (GPU {winner_idx})"
                    diff_str = "N/A"
                else:
                    winner_str = "N/A"
                    diff_str = "N/A"
                line += winner_str.ljust(40) + diff_str.ljust(28)
                print(line)
        
        # --- Multi-GPU Test for Both CUDA and Tensor Cores ---
        elif multi_test_type == 3:
            print(f"\nRunning Multi-GPU Test (Both CUDA and Tensor Cores) on GPUs: {', '.join(str(i) for i in gpu_indices)}")
            multi_results_cuda = {}
            multi_results_tensor = {}
            for idx in gpu_indices:
                device = torch.device(f"cuda:{idx}")
                multi_results_cuda[idx] = test_device(device, matrix_sizes)
                if has_tensor_cores(idx):
                    multi_results_tensor[idx] = test_device(device, matrix_sizes, use_tensor_cores=True)
                else:
                    print(f"\nNo Tensor Cores found on GPU {idx}: {gpu_names[idx]}. Skipping Tensor Cores test for this GPU.")
                    multi_results_tensor[idx] = None
            # Display CUDA results.
            print("\nMulti-GPU Test Results (CUDA):")
            header = "Size".ljust(10)
            for idx in gpu_indices:
                header += f"GPU {idx} Time".ljust(28)
            header += "Winner".ljust(40) + "Diff (Fastest vs 2nd)".ljust(28)
            print(header)
            for i, size in enumerate(matrix_sizes):
                line = f"{size:<10}"
                durations = []
                for idx in gpu_indices:
                    duration = multi_results_cuda[idx][i][1]
                    durations.append((idx, duration))
                    line += f"{format_duration(duration):<28}"
                if len(durations) > 1:
                    sorted_durations = sorted(durations, key=lambda x: x[1])
                    winner_idx, winner_time = sorted_durations[0]
                    runner_up_idx, runner_up_time = sorted_durations[1]
                    diff = runner_up_time - winner_time
                    winner_str = f"{gpu_names[winner_idx]} (GPU {winner_idx})"
                    diff_str = format_duration(diff)
                elif durations:
                    winner_idx, _ = durations[0]
                    winner_str = f"{gpu_names[winner_idx]} (GPU {winner_idx})"
                    diff_str = "N/A"
                else:
                    winner_str = "N/A"
                    diff_str = "N/A"
                line += winner_str.ljust(40) + diff_str.ljust(28)
                print(line)
            # Display Tensor Cores results.
            print("\nMulti-GPU Test Results (Tensor Cores):")
            header = "Size".ljust(10)
            for idx in gpu_indices:
                header += f"GPU {idx} Time".ljust(28)
            header += "Winner".ljust(40) + "Diff (Fastest vs 2nd)".ljust(28)
            print(header)
            for i, size in enumerate(matrix_sizes):
                line = f"{size:<10}"
                durations = []
                for idx in gpu_indices:
                    result = multi_results_tensor[idx]
                    if result is not None:
                        duration = result[i][1]
                        durations.append((idx, duration))
                        line += f"{format_duration(duration):<28}"
                    else:
                        line += "No Tensor Cores".ljust(28)
                if len(durations) > 1:
                    sorted_durations = sorted(durations, key=lambda x: x[1])
                    winner_idx, winner_time = sorted_durations[0]
                    runner_up_idx, runner_up_time = sorted_durations[1]
                    diff = runner_up_time - winner_time
                    winner_str = f"{gpu_names[winner_idx]} (GPU {winner_idx})"
                    diff_str = format_duration(diff)
                elif durations:
                    winner_idx, _ = durations[0]
                    winner_str = f"{gpu_names[winner_idx]} (GPU {winner_idx})"
                    diff_str = "N/A"
                else:
                    winner_str = "N/A"
                    diff_str = "N/A"
                line += winner_str.ljust(40) + diff_str.ljust(28)
                print(line)
    
    print("\nThank you for running the Matrix Multiplication Performance Test!")
    print("Written by: CHAT-GPT o3-mini-high, on 02/14/2025.")

if __name__ == "__main__":
    main()
