# Assignment 2-CST435-parallel-image-processing
Parallel Image Processing using Python (multiprocessing module; concurrent.futures) on GCP

# CST435 – Parallel Image Processing on GCP

## Project Overview
This project implements a parallel image processing system using Python.
A subset of the Food-101 dataset is processed using two different
parallel programming paradigms and executed on Google Cloud Platform (GCP).

The Objective of this project are:
1. To design and implement parallel programs using multiple parallel computing paradigms
2. To deploy and execute parallel applications on Google Cloud Platform (GCP)
3. To analyze performance characteristics of parallel algorithms through speedup and
efficiency metrics

The system applies a sequence of image filters to each image:
1. **Grayscale Conversion**
   - RGB images are converted to grayscale using the luminance formula
     (ITU-R BT.601 standard).

2. **Gaussian Blur**
   - A 3×3 Gaussian kernel is applied for image smoothing.

3. **Edge Detection**
   - Sobel filters are applied in the horizontal and vertical directions.
   - Edge magnitude is computed from the Sobel gradients.

4. **Image Sharpening**
   - A sharpening kernel is applied to enhance edges and details.

5. **Brightness Adjustment**
   - Pixel intensity values are increased to improve brightness.
---

## Parallel Implementations

### 1. Multiprocessing Module
- Implemented using Python’s `multiprocessing.Pool`
- Each process handles one image at a time
- Execution time, speedup, and efficiency are measured for different
  numbers of processes

### 2. Concurrent Futures
- Implemented using Python’s concurrent.futures.ProcessPoolExecutor
- Image processing tasks are automatically scheduled and distributed among worker processes
- Execution time, speedup, and efficiency are evaluated for different numbers of worker processes (1, 2, 4)

---

## Dataset
A small subset of the Food-101 dataset was used for testing:
- baklava
- bibimbap
- cannoli

Each class contains 50 images. the dataet is in the zip file

---

## Deployment on Google Cloud Platform
Both implementations were deployed and executed on a Google Cloud
Compute Engine virtual machine running Debian GNU/Linux 12.
Performance measurements were collected by varying the number of
processes to analyze scalability, speedup, and efficiency.

---

## How to Run (Google Cloud Platform)

This project was executed on a Google Cloud Compute Engine virtual machine.
The steps below describe how to run the implementations on GCP.

### 1. Create a Compute Engine VM
- Go to Google Cloud Console
- Navigate to **Compute Engine → VM Instances**
- Create a new VM with the following recommended settings:
  - Machine type: e2-standard-4 (4 vCPUs)
  - Operating system: Debian GNU/Linux 12
- Start the VM and connect using the **SSH** button.

---

### 2. Update System and Install Python
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip unzip -y

---
### 3. Create and Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate

---
### 4. Install Required Python Libraries
pip install opencv-python numpy

---
### 5. Upload Project Files

Upload the following files to the VM:
- multiprocessing_image.py
- concurrent_image_processing.py
- dataset.zip (Food-101 subset)
after uploading extract the dataset: unzip dataset.zip

---
### 6.Run the Multiprocessing Implementation
python concurrent_image.py

---
### 7.Run the Concurrent Futures Implementation
python concurrent_image_processing.py




