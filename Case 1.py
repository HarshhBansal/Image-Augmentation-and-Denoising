import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load original image ---
image_path = "Img-01.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found. Check the file path.")

# --- Original RGB Image ---
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original RGB Image")
plt.axis("off")
plt.show()

# --- Check number of channels ---
if len(img.shape) == 2:
    channels = 1  
else:
    channels = img.shape[2]  
print("Number of channels:", channels)

# --- Split and plot RGB channels ---
b, g, r = cv2.split(img) 
print("Red channel shape:", r.shape)
print("Blue channel shape:", b.shape)
print("Green channel shape:", g.shape)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(r, cmap='Reds')
plt.title("Red Channel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(g, cmap='Greens')
plt.title("Green Channel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(b, cmap='Blues')
plt.title("Blue Channel")
plt.axis("off")

plt.show()

# --- Resize and Grayscale ---
img_resized = cv2.resize(img, (256, 256))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_norm = img_gray / 255.0  # normalize to 0-1

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title("Resized RGB (256x256)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")
plt.show()

# --- Data Augmentation (Color Images) ---
h, w = img.shape[:2]
center = (w // 2, h // 2)

# Rotation
M_rot = cv2.getRotationMatrix2D(center, 30, 1.0)
img_rotated = cv2.warpAffine(img, M_rot, (w, h))

# Flip
img_flipped = cv2.flip(img, 1)

# Zoom 
zoom_factor = 1.2
new_h, new_w = int(h*zoom_factor), int(w*zoom_factor)
img_zoomed = cv2.resize(img, (new_w, new_h))
start_x = (new_w - w) // 2
start_y = (new_h - h) // 2
img_zoomed = img_zoomed[start_y:start_y+h, start_x:start_x+w]

# Translation
M_trans = np.float32([[1, 0, 30], [0, 1, 20]])
img_translated = cv2.warpAffine(img, M_trans, (w, h))

# Brightness increase
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.3, 0, 255)
img_bright = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# Display Augmented Color Images 
plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
plt.title("Flipped")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2RGB))
plt.title("Zoomed")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB))
plt.title("Translated")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB))
plt.title("Brightness Increased")
plt.axis("off")
plt.show()

# Noise Addition (Grayscale) 
# Gaussian Noise 
mean = 0
std_dev = 0.05
gaussian_noise = np.random.normal(mean, std_dev, img_gray_norm.shape)
img_gaussian = np.clip(img_gray_norm + gaussian_noise, 0, 1)
print("Gaussian noise added with mean=0 and std=0.05")

# Salt & Pepper Noise 
prob = 0.02
img_sp = np.copy(img_gray_norm)
rand = np.random.rand(*img_gray_norm.shape)
img_sp[rand < (prob/2)] = 0
img_sp[rand > 1 - (prob/2)] = 1
print("Salt & Pepper noise added with probability=0.02")

# Speckle Noise
speckle_noise = np.random.normal(0, 0.1, img_gray_norm.shape)
img_speckle = np.clip(img_gray_norm + img_gray_norm * speckle_noise, 0, 1)
print("Speckle noise added with Gaussian factor std=0.1")

# Poisson Noise
img_poisson = np.random.poisson(img_gray) / 255.0
img_poisson = np.clip(img_poisson, 0, 1)
print("Poisson noise added")

# Uniform Noise
a, b = -0.1, 0.1
uniform_noise = np.random.uniform(a, b, img_gray_norm.shape)
img_uniform = np.clip(img_gray_norm + uniform_noise, 0, 1)
print("Uniform noise added with range [-0.1, 0.1]")

#  Display Noise Images 
plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_gaussian, cmap='gray')
plt.title("Gaussian Noise")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_sp, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img_speckle, cmap='gray')
plt.title("Speckle Noise")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(img_poisson, cmap='gray')
plt.title("Poisson Noise")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(img_uniform, cmap='gray')
plt.title("Uniform Noise")
plt.axis("off")
plt.show()

# Filtration 
# Convert normalized image to 0-255 uint8 for filtering
img_gaussian_uint8 = (img_gaussian * 255).astype(np.uint8)

#  Apply Gaussian Blur on Gaussian noise 
img_gaussian_denoised = cv2.GaussianBlur(img_gaussian_uint8, (5, 5), 0)

# Apply Median Blur on Gaussian noise  
img_median_denoised = cv2.medianBlur(img_gaussian_uint8, 5)  


# Display the results
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img_gaussian, cmap='gray')
plt.title("Original Gaussian Noise")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_gaussian_denoised, cmap='gray')
plt.title("After Gaussian Blur")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_median_denoised, cmap='gray')
plt.title("After Median Blur")
plt.axis("off")

plt.show()

#  Gaussian Blur on Salt & Pepper Noise

# Convert normalized image to 0-255 uint8 (if needed)
img_sp_uint8 = (img_sp * 255).astype(np.uint8)  # skip if already uint8

# Apply Median Blur
median_filtered_sp = cv2.medianBlur(img_sp_uint8, 3)  # 3x3 kernel

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_sp, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(median_filtered_sp, cmap='gray')
plt.title("After Median Filter")
plt.axis("off")

plt.show()

# Gaussian Blur on Speckle noise
# Convert normalized image to uint8 (if needed)
img_speckle_uint8 = (img_speckle * 255).astype(np.uint8)  

# Apply Gaussian Blur
speckle_filtered = cv2.GaussianBlur(img_speckle_uint8, (3, 3), 0)  

# Display results
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_speckle, cmap='gray')
plt.title("Speckle Noise")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(speckle_filtered, cmap='gray')
plt.title("After Gaussian Blur")
plt.axis("off")

plt.show()

#  Gaussian Blur on Poisson NOISE

# Convert normalized image to uint8 (if needed)
img_poisson_uint8 = (img_poisson * 255).astype(np.uint8)  

# Apply Gaussian Blur
poisson_filtered = cv2.GaussianBlur(img_poisson_uint8, (3,3), 0) 

# Display results
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_poisson, cmap='gray')
plt.title("Poisson Noise")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(poisson_filtered, cmap='gray')
plt.title("After Gaussian Blur")
plt.axis("off")

plt.show()

# Gaussian blur on noisy image

img_noisy = np.clip(img_uniform , 0, 1)  # keep values in [0,1]

# Convert to uint8 for OpenCV functions
img_noisy_uint8 = (img_noisy * 255).astype(np.uint8)

# Apply Gaussian Blur
img_blurred = cv2.GaussianBlur(img_noisy_uint8, (5, 5), sigmaX=1)

# Plot the results

plt.figure(figsize=(15,5))

# Original
plt.subplot(1,3,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis("off")

# Noisy
plt.subplot(1,3,2)
plt.imshow(img_uniform, cmap='gray')
plt.title("Uniform Noise Added")
plt.axis("off")

# Gaussian Blur
plt.subplot(1,3,3)
plt.imshow(img_blurred, cmap='gray')
plt.title("Gaussian Blur on Noisy Image")
plt.axis("off")

plt.show()

