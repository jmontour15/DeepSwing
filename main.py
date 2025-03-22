from extract_metrics import Get_Metrics

input_path = "data/sample_video.mp4"
YOLO_path = "models/yolo11l-pose.pt"
swingnet_path = "models/swingnet_1800.pth"

Get_Metrics(input_path, YOLO_path, swingnet_path)