"""
CST435: Parallel and Cloud Computing
Assignment 2

concurrent.futures Implementation (Python)

This script applies an image processing pipeline to a subset of the
Food-101 dataset using Python's concurrent.futures. A subset of the Food-101 dataset is
processed by applying the following filters:

1. Grayscale conversion using luminance formula
2. Gaussian blur using a 3×3 kernel
3. Sobel edge detection
4. Image sharpening
5. Brightness adjustment

The program measures execution time, speedup, and efficiency
for different numbers of processes when executed on Google Cloud Platform.

Author: Nur Rabiatul Adawiyah 
"""
import os
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
INPUT_DIR = "dataset/food101_subset"   # Your dataset folder
OUTPUT_DIR = "output_images_cf"         # Output folder for processed images

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. PREPARE TASKS ---
image_tasks = []
for category in os.listdir(INPUT_DIR):
    category_path = os.path.join(INPUT_DIR, category)
    
    if os.path.isdir(category_path):
        # Create category folder in output
        output_category = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_category, exist_ok=True)
        
        # Add image tasks
        for file in os.listdir(category_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")) and not file.startswith("._"):
                image_tasks.append((category, file))

# --- 2. IMAGE PROCESSING FUNCTION ---
def process_image(task):
    category, filename = task
    
    input_path = os.path.join(INPUT_DIR, category, filename)
    output_path = os.path.join(OUTPUT_DIR, category, filename)
    
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error processing {filename}: cannot read file")
        return None
    
    # 1️⃣ Grayscale Conversion (luminance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2️⃣ Gaussian Blur (3x3 kernel)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3️⃣ Edge Detection (Sobel)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(edges)
    
    # 4️⃣ Image Sharpening
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpened = cv2.filter2D(edges, -1, kernel_sharp)
    
    # 5️⃣ Brightness Adjustment (+50)
    brightness_matrix = np.ones(sharpened.shape, dtype="uint8") * 50
    final_result = cv2.add(sharpened, brightness_matrix)
    
    # Save to corresponding category folder
    cv2.imwrite(output_path, final_result)
    
    return filename
  
  # --- 3. RUN CONCURRENT PROCESSING ---
  def run_concurrent(workers):
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_image, image_tasks))
    
    end = time.time()
    return end - start

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Fix: Define cpu_cores by using os.cpu_count()
    cpu_cores = os.cpu_count()
    
    if not image_tasks:
        print("No images found! Check your INPUT_DIR path.")
    else:
        print(f"Found {len(image_tasks)} images.")
        
        # Baseline: 1 worker
        baseline_time = run_concurrent(1)
        print(f"\nBaseline (1 worker): {baseline_time:.2f} seconds")
        
       # Test 2, 4, and max CPU cores (same as multiprocessing)
        test_counts = sorted(list(set([2, 4, cpu_cores])))

        for workers in test_counts:
            exec_time = run_concurrent(workers)

            speedup = baseline_time / exec_time
            efficiency = speedup / workers

            print(f"\nWorkers: {workers}")
            print(f"Execution time: {exec_time:.2f} seconds")
            print(f"Speedup: {speedup:.2f}")
            print(f"Efficiency: {efficiency:.2f}")

