from ultralytics import YOLO
import click


@click.command()
@click.option("--input_video", "-v", type=click.Path(exists=True))
@click.option("--model_path", "-m", type=click.Path(exists=True))
@click.option(
    "--confidence-threshold",
    "-c",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    help="Minimum confidence score for detections to be kept.",
)
def process_video(input_video, model_path, confidence_threshold):
    """
    Processes a video using a provided model, filtering detections below a confidence threshold.

    Args:
        input_video (str): Path to the input video file.
        model_path (str): Path to the model file.
        confidence_threshold (float): Minimum confidence score for detections.
    """

    # Load the model
    model = YOLO(model=model_path)
    model.track(
        source=input_video, conf=confidence_threshold, show=True
    )  # Tracking with ByteTrack tracker


if __name__ == "__main__":
    process_video()
