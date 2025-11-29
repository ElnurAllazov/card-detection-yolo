from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")
    data_yaml_file = 'dataset/data.yaml'
    project = 'project'
    experiment = 'Card-Model'
    batch_size = 32

    model.train(
        data=data_yaml_file,
        epochs=50,
        project=project,
        name=experiment,
        batch=batch_size,
        device=0,
        patience=5,
        imgsz=640,
        verbose=True,
        val=True
    )

if __name__ == "__main__":
    main()
