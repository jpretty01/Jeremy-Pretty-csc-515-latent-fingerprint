# Jeremy Pretty
# CSC 515 Module 5 Crit Latent Fingerprint
# March 18, 2023
import cv2
import numpy as np
import os

def enhance_fingerprint(image_path):
    # Read the latent fingerprint image
    fingerprint = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding to highlight the fingerprint pattern
    fingerprint = cv2.adaptiveThreshold(fingerprint, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Define the structuring element for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion to remove noise
    eroded_fingerprint = cv2.erode(fingerprint, kernel, iterations=1)

    # Perform dilation to enhance the fingerprint ridges
    dilated_fingerprint = cv2.dilate(eroded_fingerprint, kernel, iterations=1)

    # Perform opening (erosion followed by dilation) to remove small artifacts
    opened_fingerprint = cv2.morphologyEx(dilated_fingerprint, cv2.MORPH_OPEN, kernel)

    # Perform closing (dilation followed by erosion) to close small gaps in the ridges
    enhanced_fingerprint = cv2.morphologyEx(opened_fingerprint, cv2.MORPH_CLOSE, kernel)

    return eroded_fingerprint, dilated_fingerprint, opened_fingerprint, enhanced_fingerprint

# Example usage
if __name__ == "__main__":
    #Filepath to the fingerprint
    latent_fingerprint = os.path.join(os.path.dirname(__file__), 'latent.jpg')
    eroded_image, dilated_image, opened_image, enhanced_image = enhance_fingerprint(latent_fingerprint)

    # Show the original and processed fingerprint images
    cv2.imshow("Original Fingerprint", cv2.imread(latent_fingerprint, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("Eroded Fingerprint", eroded_image)
    cv2.imshow("Dilated Fingerprint", dilated_image)
    cv2.imshow("Opened Fingerprint", opened_image)
    cv2.imshow("Enhanced Fingerprint", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
