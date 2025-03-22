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

    pass

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
    
    return result

