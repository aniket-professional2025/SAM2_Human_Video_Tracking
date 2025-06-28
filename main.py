# Importing required libraries
from ultralytics import SAM
import cv2
import os # Import os for path manipulation
import numpy as np # Import numpy for potential array manipulations if needed, though not strictly used in this simplified plot method

# Define a function to completely segment a video
def segment_complete_video(input_video_path, output_video_path):

    print("Function for Segmenting an input video completely starts here....")

    # Load the SAM model
    sam_model = SAM(r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\sam2.1_b.pt")
    print("Successfully Loaded SAM model")

    # 1. Read the input video
    cap = cv2.VideoCapture(input_video_path)
    print("The video is loaded properly")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4. 'mp4v' is generally compatible.
    print("Video properties fetched successfully")

    # 2. Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print("The output video writer is set")

    # Start processing the frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video or error reading frame

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Convert BGR frame to RGB for SAM model input (Ultralytics models often prefer RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Perform segmentation on the current frame
        # The SAM model expects a list of images for batch processing,
        # but can also take a single image. `stream=True` is for generator output.
        # `verbose=False` suppresses detailed console output per frame.
        sam_results = sam_model(image_rgb, stream=True, verbose=False)

        annotated_frame = frame.copy() # Initialize annotated_frame with original frame

        # 4. Iterate through the SAM results for the current frame
        # (Even if it's a single frame, results might be iterable)
        for res in sam_results:
            # Plot the segmentation masks on the frame.
            # The .plot() method returns the annotated image (numpy array).
            # We set boxes=False to avoid drawing bounding boxes if SAM generates them,
            # focusing only on the masks.
            # Note: res.plot() returns a BGR image suitable for cv2.VideoWriter
            annotated_frame = res.plot(boxes=False)

        # 5. Write the annotated frame to the output video
        out.write(annotated_frame)

    # 6. Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("The process of segmenting an input video completely finished successfully")
