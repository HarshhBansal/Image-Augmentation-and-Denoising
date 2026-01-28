# Image Augmentation and Denoising

## 📌 Overview
This project implements a **classical image processing pipeline** using **Python and OpenCV**.  
It demonstrates fundamental operations such as image loading, RGB channel analysis, preprocessing, data augmentation, noise modeling, and noise reduction using spatial filtering techniques.

The project is intended for **learning, experimentation, and demonstration of core image preprocessing concepts** commonly used in computer vision pipelines.

## 🛠️ Technologies Used
- Python
- OpenCV (cv2)
- NumPy
- Matplotlib
  
## 📂 Project Workflow

### 1. Image Loading and Visualization
- Load an image using OpenCV
- Convert from BGR to RGB format
- Display the original RGB image

### 2. Channel Analysis
- Detect the number of image channels
- Split the image into:
  - Red channel
  - Green channel
  - Blue channel
- Visualize each channel independently

### 3. Image Preprocessing
- Resize image to **256 × 256**
- Convert RGB image to **grayscale**
- Normalize grayscale image values to the range **[0, 1]**
  
### 4. Data Augmentation Techniques
The following augmentation techniques are applied to the color image:

- Rotation (30 degrees)
- Horizontal flipping
- Zooming
- Translation (x and y shift)
- Brightness enhancement using HSV color space

All augmented images are visualized for comparison.

### 5. Noise Modeling (Grayscale Image)
Different types of noise are artificially added to the grayscale image:

- Gaussian Noise
- Salt & Pepper Noise
- Speckle Noise
- Poisson Noise
- Uniform Noise

Each noise type is visualized to analyze its effect on image quality.

### 6. Noise Filtering and Denoising
To remove noise, classical spatial filters are applied:

- **Gaussian Blur**
  - Applied to Gaussian noise
  - Applied to Speckle noise
  - Applied to Poisson noise
  - Applied to Uniform noise
- **Median Filter**
  - Applied to Gaussian noise
  - Applied to Salt & Pepper noise (most effective)

Side-by-side visual comparisons are shown between noisy and filtered images.

## 📊 Outputs
The project generates multiple visual results including:
- Original RGB image(results/Grayscale_results.png)
- Augmented images
- Noisy grayscale images

These outputs help in understanding the behavior of noise and the effectiveness of filtering techniques.


## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Image-Augmentation-and-Denoising.git

pip install opencv-python numpy matplotlib

Place an image file (e.g., Img-01.jpg) in the project directory.
python image_processing.py
