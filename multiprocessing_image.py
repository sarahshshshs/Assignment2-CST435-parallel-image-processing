"""
CST435: Parallel and Cloud Computing
Assignment 2

Multiprocessing Implementation (Python)

This script applies an image processing pipeline to a subset of the
Food-101 dataset using Python's multiprocessing module.

Operations:
1. Grayscale conversion
2. Gaussian blur
3. Edge detection
4. Image sharpening
5. Brightness adjustment

The program measures execution time, speedup, and efficiency
for different numbers of processes when executed on Google Cloud Platform.

Author: Sarah Nur Dinie
"""


import os
import time
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageFilter, ImageEnhance

INPUT_DIR = "dataset/food101_subset"
OUTPUT_DIR = "output_images_mp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_tasks = []

for category in os.listdir(INPUT_DIR):
    category_path = os.path.join(INPUT_DIR, category)

    if os.path.isdir(category_path):
        output_category = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_category, exist_ok=True)

        for file in os.listdir(category_path):
            if file.lower().endswith(".jpg") and not file.startswith("._"):
                image_tasks.append((category, file))

def process_image(task):
    category, filename = task

    input_path = os.path.join(INPUT_DIR, category, filename)
    output_path = os.path.join(OUTPUT_DIR, category, filename)

    img = Image.open(input_path)

    # 1. Grayscale
    img = img.convert("L")

    # 2. Gaussian Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # 3. Edge Detection
    img = img.filter(ImageFilter.FIND_EDGES)

    # 4. Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # 5. Brightness
    img = ImageEnhance.Brightness(img).enhance(1.2)

    img.save(output_path)

    return filename

def run_multiprocessing(workers):
    start = time.time()

    with Pool(processes=workers) as pool:
        pool.map(process_image, image_tasks)

    end = time.time()
    return end - start

if __name__ == "__main__":
    baseline_time = run_multiprocessing(1)
    print(f"\nBaseline (1 process): {baseline_time:.2f} seconds")

    for workers in [2, 4, cpu_count()]:
        exec_time = run_multiprocessing(workers)
        speedup = baseline_time / exec_time
        efficiency = speedup / workers

        print(f"\nProcesses: {workers}")
        print(f"Execution time: {exec_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}")
        print(f"Efficiency: {efficiency:.2f}")

