from extract_metrics import Get_Metrics

input_path = "data/driver/face_on/korda_driver_face.mp4"
YOLO_path = "models/yolo11l-pose.pt"
swingnet_path = "models/swingnet_1800.pth"

pose_estimation, swing_timings = Get_Metrics(input_path, YOLO_path, swingnet_path)

print(pose_estimation)