# Importing functions from main, main2, main3, main4 and main5
from main import segment_complete_video
from main2 import segment_persons_only
from main3 import keypoints_draw_video_person_segmentation
from main4 import keypoints_video_person_segmentation
from main5 import person_diff_segmentation_mask

# Setting the input Video path
input_video_path = r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\Video_2_Examples\Org_Basketball_Video.mp4" 

# Inferenceing using the first function
output_video_path = r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\Video_2_Examples\Complete_Segmented_Video.mp4"
segment_complete_video(input_video_path, output_video_path)

# # Inferencing using the second function
# output_video_path_2 = r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\Video_2_Examples\Person_Only_Segmented_Video.mp4"
# segment_persons_only(input_video_path, output_video_path_2)

# # Inferencing with the third function
# output_video_path_3 = r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\Video_2_Examples\KeyPoints_Draw_Segmented_Video.mp4"
# keypoints_draw_video_person_segmentation(input_video_path, output_video_path_3)

# # Inferencing with the fourth function
# output_video_path_4 = r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\Video_2_Examples\KeyPoints_Segmented_Video.mp4"
# keypoints_video_person_segmentation(input_video_path, output_video_path_4)

# # Inferencing with the fifth function
# output_video_path_5 = r"C:\Users\Webbies\Jupyter_Notebooks\SAM2_VideoTracking\Video_2_Examples\Person_Different_Segmented_Video.mp4"
# person_diff_segmentation_mask(input_video_path, output_video_path_5)