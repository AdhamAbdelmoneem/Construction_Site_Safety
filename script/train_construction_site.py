from ultralytics import YOLO

# Project: Construction Site Safety with AI
# Dataset Source: Roboflow (CC BY 4.0)

def main():
    # 1. Load a pre-trained YOLOv8 model
    # It's lightweight and perfect for real-time safety monitoring
    model = YOLO('yolov8n.pt')

    # 2. Start Training
    model.train(
        data='data.yaml',
        epochs=50,         # Start with 50 to see initial performance
        imgsz=640,        # Standard resolution for safety cameras
        batch=4,         # Adjust based on GPU/CPU memory
        name='Construction_Safety_V1',
        device='cpu'      # Use 0 if you have an NVIDIA GPU
    )

if __name__ == '__main__':
    main()