from extract_metrics import Get_Metrics
from prompting import Get_Prompt

input_path = "data/driver/face_on/korda_driver_face.mp4"
YOLO_path = "models/yolo11l-pose.pt"
swingnet_path = "models/swingnet_1800.pth"

swing_angles, lean = Get_Metrics(input_path, YOLO_path, swingnet_path)
print("Swing angles: ", swing_angles)
print("Leans: ", lean)


# Get_Prompt(swing_angles, lean)