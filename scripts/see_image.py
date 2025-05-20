import cv2

# Replace 'camera_name' with the actual camera name used in the script
image_path = "worm.jpg"  # Example image filename
image = cv2.imread(image_path)

if image is not None:
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)  # Press any key to close
    cv2.destroyAllWindows()
else:
    print("Image not found or failed to load.")

