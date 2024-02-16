"""
Training and fine tuning
"""
from ultralytics import YOLO, settings

settings.update({"datasets_dir": "./data"})


def train_custom_model(epochs: int, verbose: bool = False, resume_training: str = None):
    """
    Training YOLO v8
    Fine tune the latest YOLO v8 or resume training of a given model.
    """

    if resume_training:
        # model = YOLO("./runs/detect/train13/weights/last.pt")
        model = YOLO(resume_training)
    else:
        model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    if verbose:
        model.info()

    # Train the model
    model.train(
        data="./YOLO_configs/openimage_shrimp.yaml",
        epochs=epochs,
        device="mps",
        fraction=0.5,
        # imgsz=(480, 848),
        verbose=verbose,
        weight_decay=0.01,
        lr0=0.1,
        dropout=0.1,
        resume=True if resume_training else False,
    )


train_custom_model(epochs=100)
