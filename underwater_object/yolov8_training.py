"""
Training and fine tuning
"""

from ultralytics import YOLO


def train_custom_model(
    config_yaml: str,
    epochs: int,
    verbose: bool = False,
    resume_training: str = None,
    default_model="yolov8n.pt",
):
    """
    Training YOLO v8
    Fine tune the latest YOLO v8 or resume training of a given model.
    """

    if resume_training:
        model = YOLO(resume_training)
    else:
        model = YOLO(default_model)

    if verbose:
        model.info()

    # Train the model
    model.train(
        data=config_yaml,
        epochs=epochs,
        device="mps",
        fraction=0.8,
        verbose=verbose,
        weight_decay=0.05,
        lr0=0.1,
        dropout=0.1,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        shear=0.5,
        resume=True if resume_training else False,
    )


train_custom_model(
    config_yaml="./YOLO_configs/shrimp_train_v2.yaml", epochs=10, verbose=True
)
