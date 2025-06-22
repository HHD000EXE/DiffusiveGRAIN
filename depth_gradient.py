import cv2
import numpy as np

# --- Load or create your depth image here ---
depth_image = cv2.imread('train_images(RawD)/x10y20trial3_depth_gray_21.png', cv2.IMREAD_ANYDEPTH)
depth_float = depth_image.astype(np.float32)

# --- 1. Compute gradient magnitude using Sobel ---
grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)

# Normalize so it's in [0, 255] range (for visualization)
grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
grad_mag_uint8 = grad_mag_norm.astype(np.uint8)

# --- 2. Amplify each pixel value by 3 (use higher precision so no wrap-around) ---
amplified = grad_mag_uint8.astype(np.uint16) * 5

# --- 3. Threshold: if amplified value > 127.5 => set to 0 ---
threshold = 125
mask = amplified > threshold
amplified[mask] = 0

# --- 4. Convert back to 8-bit and save/show ---
thresholded_image = amplified.astype(np.uint8)
cv2.imwrite('gradient_magnitude_thresholded.png', thresholded_image)

# If desired, visualize:
# cv2.imshow("Thresholded Gradient", thresholded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
#
# # 1. Read depth image (8-bit or 16-bit). Convert to float to avoid overflow issues.
# depth_image = cv2.imread('train_images(RawD)/x10y20trial3_depth_gray_21.png', cv2.IMREAD_ANYDEPTH)
#
# # 2. Compute the vertical gradient using Sobel:
# #    (dx=0, dy=1) => gradient from top to bottom
# grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=3)
#
# # 3. Get the min and max for normalization.
# vmin, vmax = grad_y.min(), grad_y.max()
#
# # Optional:
# # If you want to explicitly center on zero, you could set e.g.:
# # vmin, vmax = -100, 100
# # or any symmetric bounds that include zero.
# # For now, we'll just use the actual min and max from the image.
#
# # 4. Normalize gradient to [0, 1].
# norm_grad = (grad_y - vmin) / (vmax - vmin + 1e-8)
#
# # 5. Map the normalized gradient to the "coolwarm" colormap in matplotlib.
# cmap = matplotlib.cm.get_cmap('coolwarm')
# grad_rgba = cmap(norm_grad)  # shape: (height, width, 4) in RGBA
#
# # 6. Convert RGBA -> BGR for OpenCV, scale to 0..255, and cast to uint8.
# grad_bgr = (grad_rgba[..., :3] * 255).astype(np.uint8)  # take RGB only
# grad_bgr = grad_bgr[..., ::-1]  # swap R and B -> BGR
#
# # 7. Save or visualize the result.
# cv2.imwrite("gradient_y_coolwarm.png", grad_bgr)
# # cv2.imshow("Vertical Gradient (Top->Bottom) Coolwarm", grad_bgr)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
