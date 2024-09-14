import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "D:\leafdetection\sr4.jpeg"
image = cv2.imread(image_path)  

# Check if the image is loaded correctly
if image is None:
    print("Error: Unable to load image. Check the file path.")
else:
    # Step 1: Original Image
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Define the green color range and create a mask for green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_before_morph = cv2.inRange(hsv, lower_green, upper_green)

    # Step 4: Morphological operations to remove noise and fill gaps
    kernel = np.ones((10,10), np.uint8)
    mask_after_morph = cv2.morphologyEx(mask_before_morph, cv2.MORPH_CLOSE, kernel)
    mask_after_morph = cv2.morphologyEx(mask_after_morph, cv2.MORPH_OPEN, kernel)

    # Step 5: Find contours in the mask and draw them on the original image
    contours, _ = cv2.findContours(mask_after_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the leaf count
    leaf_count = 0
    output_image = image.copy()

    # Loop over the contours and filter based on size
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 60:  # Threshold to filter small objects (non-leaves)
            leaf_count += 1
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

    # Convert the output image for display
    final_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Step 6: Plot all stages in one figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Display Original Image
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Display HSV Image
    axs[0, 1].imshow(hsv)
    axs[0, 1].set_title('HSV Image')
    axs[0, 1].axis('off')

    # Display Mask before Morphological Operations
    axs[0, 2].imshow(mask_before_morph, cmap='gray')
    axs[0, 2].set_title('Green Mask (Before Morphological Operations)')
    axs[0, 2].axis('off')

    # Display Mask after Morphological Operations
    axs[1, 0].imshow(mask_after_morph, cmap='gray')
    axs[1, 0].set_title('Green Mask (After Morphological Operations)')
    axs[1, 0].axis('off')

    # Display Final Image with Counted Leaves
    axs[1, 1].imshow(final_output_image)
    axs[1, 1].set_title(f'Final Image (Leaves Counted: {leaf_count})')
    axs[1, 1].axis('off')

    # Display the Final Mask with Contours
    axs[1, 2].imshow(mask_after_morph, cmap='gray')
    axs[1, 2].set_title('Final Mask with Contours')
    axs[1, 2].axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the combined figure with all stages
    plt.show()

    # Print the final leaf count
    print(f"Number of leaves detected: {leaf_count}")
