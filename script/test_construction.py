import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup paths relative to this script
# Automatically find the script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Searching for files in: {base_dir}")

# 2. Search for the model file
model_path = os.path.join(base_dir, 'runs', 'detect', 'Construction_Safety_V13', 'weights', 'best.pt')

# 3. Search for images using a wildcard (recursive search)
#This looks for any 'test/images' folder inside the project
search_pattern = os.path.join(base_dir, "**", "test", "images", "*.jpg")
image_files = glob.glob(search_pattern, recursive=True)

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
elif not image_files:
    print("Error: No images found. Please ensure your dataset folder is inside the project.")
    print("Contents of current directory:", os.listdir(base_dir))
else:
    print(f"Success: Found model and {len(image_files)} images.")
    model = YOLO(model_path)

    # Create output folder
    output_folder = os.path.join(base_dir, 'test_results_batch')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 4. Process images (first 10 images)
    for img_path in image_files[:10]:
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Define Danger Zone (Geofence)
        danger_zone = np.array([
            [int(w * 0.1), h], [int(w * 0.4), int(h * 0.6)],
            [int(w * 0.6), int(h * 0.6)], [int(w * 0.9), h]
        ], np.int32)

        # Run Inference
        results = model(img, conf=0.25)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]

                # Geofencing Logic: Check if the object inside zone
                is_inside = cv2.pointPolygonTest(danger_zone, (int((x1 + x2) / 2), y2), False)
                color = (0, 0, 255) if is_inside >= 0 else (0, 255, 0)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Zone and Save
        cv2.polylines(img, [danger_zone], True, (255, 255, 0), 2)
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, filename), img)
        print(f"Processed and saved: {filename}")

    print(f"All results saved in: {output_folder} ")