from ultralytics import YOLO
 
def main():
    model = YOLO('ultralytics/cfg/models/v8/meter.yaml').load('yolov8n-obb.pt')  # build from YAML and transfer weights
    model.train(data="ultralytics/cfg/datasets/meter.yaml", epochs=100, imgsz=1024, batch=4, workers=0)
if __name__ == '__main__':
    main()