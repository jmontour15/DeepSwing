import os
import cv2
import numpy as np

def preprocess_swing_video(input_path, output_path, bbox=None, dim=160, start_frame=None, end_frame=None):
    """
    Preprocess a golf swing video by cropping to the region of interest,
    resizing while maintaining aspect ratio, and padding to a square format.
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path where the processed video will be saved
    bbox : list or tuple, optional
        Bounding box coordinates as [x, y, width, height] in normalized format (0-1)
        If None, the entire frame will be used
    dim : int, default=160
        Output dimension (both width and height) of the processed video
    start_frame : int, optional
        First frame to include (inclusive)
    end_frame : int, optional
        Last frame to include (inclusive)
    
    Returns:
    --------
    bool
        True if processing was successful, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up frame range
    start_frame = 0 if start_frame is None else max(0, start_frame)
    end_frame = total_frames - 1 if end_frame is None else min(total_frames - 1, end_frame)
    
    # Calculate the bounding box coordinates
    if bbox is not None:
        # Convert normalized bbox coordinates to pixel values
        x = int(frame_width * bbox[0])
        y = int(frame_height * bbox[1])
        w = int(frame_width * bbox[2])
        h = int(frame_height * bbox[3])
    else:
        # Use the entire frame
        x, y = 0, 0
        w, h = frame_width, frame_height
    
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (dim, dim))
    
    # ImageNet means for padding (BGR format)
    padding_color = [0.406*255, 0.456*255, 0.485*255]
    
    # Process the video
    frames_processed = 0
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames outside our desired range
        if frame_idx < start_frame:
            continue
        if frame_idx > end_frame:
            break
        
        # Step 1: Crop to the region of interest
        cropped = frame[y:y+h, x:x+w]
        
        # Step 2: Calculate resize ratio to maintain aspect ratio
        crop_height, crop_width = cropped.shape[:2]
        ratio = dim / max(crop_height, crop_width)
        new_size = (int(crop_width * ratio), int(crop_height * ratio))
        
        # Step 3: Resize the image while maintaining aspect ratio
        resized = cv2.resize(cropped, new_size)
        
        # Step 4: Calculate padding to make the image square
        delta_w = dim - new_size[0]
        delta_h = dim - new_size[1]
        # Padding is applied evenly on both sides
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        
        # Step 5: Add padding to create a square image with the desired dimensions
        padded = cv2.copyMakeBorder(
            resized, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT,
            value=padding_color
        )
        
        # Write the frame to the output video
        out.write(padded)
        frames_processed += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return True