from preprocess import preprocess_swing_video
import cv2
from ultralytics import YOLO
from preprocess import preprocess_swing_video
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import EventDetector

'''
REFERENCE

Index	Body Part
------------------------
0	    Nose
1	    Left Eye
2	    Right Eye
3	    Left Ear
4	    Right Ear
5	    Left Shoulder
6	    Right Shoulder
7	    Left Elbow
8	    Right Elbow
9	    Left Wrist
10	    Right Wrist
11	    Left Hip
12	    Right Hip
13	    Left Knee
14	    Right Knee
15	    Left Ankle
16	    Right Ankle

'''

def Get_Metrics(input_path, YOLO_path, swingnet_path): # angle will be either down-the-line or face on
    # TODO: 
    # 1. Differentiate face on vs. down-the-line
    #   - Face on metrics
    #   - Down-the-line metrics
    # 2. Timings
    pose_estimation = Get_Pose_Estimation(input_path, YOLO_path)
    print("Pose Estimation Complete")
    print(len(pose_estimation))
    swing_timings = Get_Swing_Timings(input_path, swingnet_path)
    print("Swing Timings Collected")
    for timing in swing_timings.keys():
        print(f"{timing} - frame: {swing_timings[timing]['frame']}, time: {round(swing_timings[timing]['time'], 3)}s")
    
    keypoints = []
    for result in pose_estimation:
        xyn = pose_estimation[result].xyn  # normalized
        keypoints.append(xyn)
    
    swing_angles = get_swing_angles(keypoints, swing_timings)

def Get_Pose_Estimation(input_video_path, model_path):
    # Load model of correct size
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    results = {}
    current_frame = 0

    # Process Each Frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames

        # Run YOLO Pose Estimation on Frame
        outputs = model(frame, verbose=False)
        for output in outputs:
            results[f"frame{current_frame}"] = output.keypoints.cpu().numpy()
        current_frame +=1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return results

def Get_Swing_Timings(input_path, model_path, input_size=160, seq_length=64):
    """
    Predicts golf swing events from an MP4 video.
    
    Args:
        video_path (str): Path to the MP4 video file
        model_path (str): Path to the trained model weights
        input_size (int): Size to resize frames to (square)
        seq_length (int): Number of frames to process at once
        
    Returns:
        dict: Dictionary with swing event names as keys and timestamps (in seconds) as values
    """
    # Preprocess video
    preprocess_swing_video(input_path, output_path = "outputs/get_timings_output.mp4")

    # Event name mapping
    event_names = {
        0: 'Address',
        1: 'Toe-up',
        2: 'Mid-backswing',
        3: 'Top',
        4: 'Mid-downswing',
        5: 'Impact',
        6: 'Mid-follow-through',
        7: 'Finish'
    }
    
    # Load model
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Open video
    cap = cv2.VideoCapture("outputs/get_timings_output.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    
    # Calculate resize ratio and padding
    ratio = input_size / max(frame_size)
    new_size = tuple([int(x * ratio) for x in frame_size])
    delta_w = input_size - new_size[1]
    delta_h = input_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Normalization parameters (ImageNet)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Process video frames
    images = []
    for _ in range(frame_count):
        ret, img = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        resized = cv2.resize(img, (new_size[1], new_size[0]))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
        rgb_img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        images.append(rgb_img)
    
    cap.release()
    
    # Convert to tensor and normalize
    images = np.asarray(images)
    images = images.transpose((0, 3, 1, 2))  # [frames, channels, height, width]
    images = torch.from_numpy(images).float().div(255.)
    images = normalize(images)
    images = images.unsqueeze(0)  # Add batch dimension [1, frames, channels, height, width]
    
    # Run model inference in batches
    probs = None
    batch = 0
    with torch.no_grad():
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            
            logits = model(image_batch.to(device))
            
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            
            batch += 1
    
    # Get event frames
    event_frames = np.argmax(probs, axis=0)[:-1]  # Remove the last class (background)
    
    # Calculate confidence scores
    confidence = []
    for i, e in enumerate(event_frames):
        confidence.append(probs[e, i])
    
    # Convert frames to timestamps
    result = {}
    for i, frame_num in enumerate(event_frames):
        event_name = event_names[i]
        timestamp = frame_num / fps
        result[event_name] = {
            'frame': int(frame_num),
            'time': float(timestamp),
            'confidence': float(confidence[i])
        }
    # Remove edge case where address isn't recognized
    # Only happens if there isnt enough time before swing starts)
    if result["Address"]["time"] > result["Toe-up"]["time"]:
        result["Address"]["time"] = result["Toe-up"]["time"] - 0.7
        result["Address"]["frame"] = result["Toe-up"]["frame"] - 0.7*fps
        # If -0.7 makes address time negative
        if result["Address"]["time"] < 0:
            result["Address"]["time"] = 0
            result["Address"]["frame"] = 0

    return result

def get_swing_angles(keypoints, timings):

    # Metrics at address
    address = timings["Address"]["frame"]
    address_angles = calculate_event_angles(keypoints[address])

    # Metrics at toe-up
    toe_up = timings["Toe-up"]["frame"]
    toe_up_angles = calculate_event_angles(keypoints[toe_up])

    # Metrics at mid-backswing
    mid_backswing = timings["Mid-backswing"]["frame"]
    mid_backswing_angles = calculate_event_angles(keypoints[mid_backswing])

    # Metrics at top
    top = timings["Top"]["frame"]
    top_angles = calculate_event_angles(keypoints[top])

    # Metrics at mid-downswing
    mid_downswing = timings["Mid-backswing"]["frame"]
    mid_downswing_angles = calculate_event_angles(keypoints[mid_downswing])

    # Metrics at impact
    impact = timings["Impact"]["frame"]
    impact_angles = calculate_event_angles(keypoints[impact])

    angles = {
        "Address": address_angles,
        "Toe-up": toe_up_angles,
        "Mid-backswing": mid_backswing_angles,
        "Top": top_angles,
        "Mid-downswing": mid_downswing_angles,
        "Impact": impact_angles
    }
    return angles

def calculate_event_angles(keypoints):
    # calculates swing angles at a single time point
    #   Each arm to shoulders
    #   Shoulders to hips
    #   Upper arm to lower arm
    #   Center of hips horizontal value

    # Each arm to shoulders
    left_arm_shoulders = calculate_angle(p1 = keypoints[7], p2 = keypoints[5], p3 = keypoints[6])
    right_arm_shoulders = calculate_angle(p1 = keypoints[8], p2 = keypoints[6], p3 = keypoints[5])

    # Calculates shoulders angle to ground
    shoulders = calculate_angle(p1 = (keypoints[6][0]-1, keypoints[6][1]), p2 = keypoints[6], p3 = keypoints[5])

    # Calculates hips angle to ground
    hips = calculate_angle(p1 = (keypoints[12][0]-1, keypoints[12][1]), p2 = keypoints[12], p3 = keypoints[11])

    # Upper arm to lower arm
    arm_angle_left = calculate_angle(p1 = keypoints[9], p2 = keypoints[7], p3 = keypoints[5])
    arm_angle_right = calculate_angle(p1 = keypoints[10], p2 = keypoints[8], p3 = keypoints[6])

    # Mid point of hips to determine lean
    com = np.mean([keypoints[11][0], keypoints[12][0]])

    angles = [left_arm_shoulders, 
              right_arm_shoulders,
              shoulders,
              hips,
              arm_angle_left,
              arm_angle_right,
              com]
    
    return angles

def calculate_angle(p1, p2, p3):
    """
    Computes the angle at joint p2 given limb segments p1->p2 and p2->p3.
    
    Parameters:
    p1, p2, p3: Tuples representing (x, y) coordinates.
    
    Returns:
    Angle in degrees.
    """
    # Convert points to NumPy arrays
    v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
    v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

    # Compute dot product and magnitudes
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Compute angle in radians and convert to degrees
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

