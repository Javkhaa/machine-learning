import cv2
import click
from ultralytics import YOLO


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
@click.option("--show", is_flag=True, default=False, help="Show video")
@click.option("--save_video", type=str, default=None, help="Save the tracking video")
def process_video(
    input_video,
    model_path,
    confidence_threshold,
    show: bool = False,
    save_video: str | None = None,
):
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if save_video:
        output = cv2.VideoWriter(save_video)


    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=confidence_threshold, stream=True)

            # Visualize the results on the frame
            for result in results:
                if show:
                    annotated_frame = result.plot()
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
