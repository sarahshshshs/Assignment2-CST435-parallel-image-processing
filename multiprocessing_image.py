"""
CST435: Parallel and Cloud Computing
Assignment 2

Multiprocessing Module Implementation (Python)

This script applies an image processing pipeline to a subset of the
Food-101 dataset using Python's multiprocessing module. A subset of the Food-101 dataset is
processed by applying the following filters:

1. Grayscale conversion using luminance formula
2. Gaussian blur using a 3×3 kernel
3. Sobel edge detection
4. Image sharpening
5. Brightness adjustment

The program measures execution time, speedup, and efficiency
for different numbers of processes when executed on Google Cloud Platform.

Author: Sarah Nur Dinie
"""


import os
import time
import cv2  # OpenCV is required for specific filters like Sobel/3x3 Kernels
import numpy as np
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
INPUT_DIR = "dataset/food101_subset"
OUTPUT_DIR = "output_images_mp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. PREPARE TASKS ---
image_tasks = []
for category in os.listdir(INPUT_DIR):
    category_path = os.path.join(INPUT_DIR, category)

    if os.path.isdir(category_path):
        output_category = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_category, exist_ok=True)

        for file in os.listdir(category_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith("._"):
                image_tasks.append((category, file))

# --- 2. THE WORKER FUNCTION ---
def process_image(task):
    category, filename = task
    
    input_path = os.path.join(INPUT_DIR, category, filename)
    output_path = os.path.join(OUTPUT_DIR, category, filename)

    # Load Image
    img = cv2.imread(input_path)
    if img is None:
        return None

    # --- REQUIREMENT 1: Grayscale Conversion ---
    # "Convert RGB images to grayscale using luminance formula"
    # cv2.cvtColor uses the standard ITU-R 601-2 luminance formula (Y = 0.299R + 0.587G + 0.114B)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- REQUIREMENT 2: Gaussian Blur  ---
    # "Apply 3x3 Gaussian kernel for smoothing"
    # (3, 3) to explicitly set the kernel size.
    # We apply this to the 'gray' image to prepare it for edge detection.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- REQUIREMENT 3: Edge Detection  ---
    # "Sobel filter to detect edges"
    # We must calculate the gradient in X and Y directions separately and combine them.
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude (strength) of the edges
    edges = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(edges)

    # --- REQUIREMENT 4: Image Sharpening ---
    # "Enhance edges and details"
    # We apply a sharpening kernel to the edges to make them crisp.
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpened = cv2.filter2D(edges, -1, kernel_sharp)

    # --- REQUIREMENT 5: Brightness Adjustment ---
    # "Increase or decrease image brightness"
    # We add a constant value (50) to the pixels.
    brightness_matrix = np.ones(sharpened.shape, dtype="uint8") * 50
    final_result = cv2.add(sharpened, brightness_matrix)

    # Save the Final Result
    cv2.imwrite(output_path, final_result)

    return filename

# --- 3. MULTIPROCESSING MANAGER ---
def run_multiprocessing(workers):
    start = time.time()

    with Pool(processes=workers) as pool:
        pool.map(process_image, image_tasks)

    end = time.time()
    return end - start

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    if not image_tasks:
        print("No images found! Check your INPUT_DIR path.")
    else:
        print(f"Found {len(image_tasks)} images.")
        
        # Run Baseline (1 process)
        baseline_time = run_multiprocessing(1)
        print(f"\nBaseline (1 process): {baseline_time:.2f} seconds")

        # Run Parallel Tests
        # We test 2, 4, and Max Cores as typical benchmarks
        test_counts = sorted(list(set([2, 4, cpu_count()])))
        
        for workers in test_counts:
            exec_time = run_multiprocessing(workers)
            
            # Calculate Metrics 
            speedup = baseline_time / exec_time
            efficiency = speedup / workers

            print(f"\nProcesses: {workers}")
            print(f"Execution time: {exec_time:.2f} seconds")
            print(f"Speedup: {speedup:.2f}")
            print(f"Efficiency: {efficiency:.2f}")
