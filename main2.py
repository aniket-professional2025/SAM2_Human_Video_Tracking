# Importing required packages
from ultralytics import SAM
from ultralytics import YOLO # Import YOLO for object detection
import cv2
import os


# Define a function to create the segmentation masks only for the persons
def segment_persons_only(input_video_path, output_video_path):

    print("Function for Segmenting only Persons/Humans from input video start here.....")

    # Load the SAM model
    sam_model = SAM(r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\sam2.1_b.pt")

    # Load a YOLOv8 model for object detection
    yolo_model = YOLO(r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\yolov8n.pt")

    # Define the classes we are interested in (YOLOv8 COCO dataset classes)
    # Person class ID is 0, Ball might be a 'sports ball' which is 32 or similar.
    # You might need to check YOLOv8's class list for exact IDs.
    # For simplicity, let's assume 'person' and 'sports ball' (if ball is detected as such)
    target_classes = ['person'] # Or ['person', 'ball'] if it exists

    # 1. Read the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can try 'XVID', 'MJPG', 'mp4v' for different codecs

    # 2. Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Start processing the frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # 3. Perform object detection with YOLOv8
        yolo_results = yolo_model(frame, verbose=False) # verbose=False to suppress print output

        # Extract bounding boxes for target classes
        boxes_for_sam = []
        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = yolo_model.names[class_id] # Get class name from ID

                    if class_name in target_classes:
                        # Convert xyxy format to list [x1, y1, x2, y2]
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                        boxes_for_sam.append([x1, y1, x2, y2])

        annotated_frame = frame.copy() # Start with a copy of the original frame

        # 4. If target objects are found, perform segmentation with SAM using these boxes
        if boxes_for_sam:
            # Pass the detected boxes to SAM model
            # The SAM model expects a list of images and a list of boxes for each image.
            # Here, we have one image (frame) and potentially multiple boxes.
            sam_results = sam_model(frame, bboxes=boxes_for_sam, stream=True)

            # Plot only the segmented objects
            for res in sam_results:
                # `res.plot()` will draw the masks and boxes (if boxes are not explicitly false)
                # You might need to refine this to only draw masks if that's preferred.
                # However, `res.plot()` is the easiest way to get the visual output.
                annotated_frame = res.plot(boxes=False) # Plot masks, optionally hide boxes

        # 5. Write the annotated frame to the output video
        out.write(annotated_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # 6. Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("The process of segmenting Person/Human in an input video finished successfully")