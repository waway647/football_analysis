import cv2
import numpy as np
import pandas as pd
import pickle
import os
import logging
from mmocr.apis import MMOCRInferencer

log = logging.getLogger(__name__)

class JerseyNumberRecognizer:
    def __init__(self, det_model='DB_r_18', rec_model='SAR', device='cuda'):
        """
        Initialize MMOCR for jersey number recognition.
        
        Args:
            det_model (str): Text detection model (e.g., 'DB_r_18' for DBNet).
            rec_model (str): Text recognition model (e.g., 'SAR').
            device (str): Device for MMOCR ('cuda' or 'cpu').
        """
        try:
            self.mmocr = MMOCRInferencer(det=det_model, rec=rec_model, device=device if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
            log.info(f"MMOCR initialized with det={det_model}, rec={rec_model} on {device}")
        except Exception as e:
            log.error(f"Failed to initialize MMOCR: {e}")
            raise
        self.player_class = 'player'  # Matches Tracker's class name

    def recognize_jersey_number(self, frame, bbox):
        """
        Recognize jersey number in a player crop.
        
        Args:
            frame (np.ndarray): Video frame (BGR).
            bbox (list or tuple): Bounding box [x1, y1, x2, y2].
            
        Returns:
            int: Jersey number (e.g., 10) or -1 if not detected.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure valid crop bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                log.debug(f"Invalid bbox: {bbox}")
                return -1
            player_crop = frame[y1:y2, x1:x2]
            # Run MMOCR
            result = self.mmocr(player_crop, return_pred=True)
            if not result['det_polygons'] or not result['rec_texts']:
                log.debug(f"No text detected in bbox: {bbox}")
                return -1
            for text, score in zip(result['rec_texts'], result['rec_scores']):
                if text.isdigit() and len(text) <= 2 and score > 0.5:  # Jersey numbers: 1-2 digits
                    log.debug(f"Detected jersey number: {text}, score: {score}")
                    return int(text)
            log.debug(f"No valid jersey number in bbox: {bbox}")
            return -1
        except Exception as e:
            log.error(f"Error recognizing jersey number for bbox {bbox}: {e}")
            return -1

    def process_frame(self, frame, player_track):
        """
        Process player detections in a single frame to add jersey numbers.
        
        Args:
            frame (np.ndarray): Video frame (BGR).
            player_track (dict): Player tracks {track_id: {'bbox': [x1, y1, x2, y2], ...}}.
            
        Returns:
            dict: Updated player tracks with 'jersey_number'.
        """
        updated_track = player_track.copy()
        for track_id, track_info in updated_track.items():
            jersey_number = self.recognize_jersey_number(frame, track_info['bbox'])
            track_info['jersey_number'] = jersey_number
        log.debug(f"Processed {len(updated_track)} player detections, added jersey numbers")
        return updated_track

def add_jersey_numbers(tracks, video_path):
    """
    Add jersey numbers to player tracks in a video.
    
    Args:
        tracks (dict): Tracks dictionary with 'players', 'referees', 'ball'.
        video_path (str): Path to input video (e.g., 'input_videos/08fd33_4.mp4').
        
    Returns:
        dict: Updated tracks with 'jersey_number' in player tracks.
    """
    try:
        # Check if stub exists and read_from_stub is True
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                updated_tracks = pickle.load(f)
                log.info(f"Loaded tracks with jersey numbers from {stub_path}")
                return updated_tracks
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")
        
        recognizer = JerseyNumberRecognizer()
        updated_tracks = tracks.copy()
        
        for frame_num in range(len(updated_tracks['players'])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                log.warning(f"Failed to read frame {frame_num}")
                continue
            player_track = updated_tracks['players'][frame_num]
            updated_track = recognizer.process_frame(frame, player_track)
            updated_tracks['players'][frame_num] = updated_track
        
        cap.release()
        log.info(f"Added jersey numbers to {len(updated_tracks['players'])} frames")
        
        # Save updated tracks to stub if stub_path is provided
        if stub_path is not None:
            try:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)
                with open(stub_path, 'wb') as f:
                    pickle.dump(updated_tracks, f)
                log.info(f"Saved updated tracks to {stub_path}")
            except Exception as e:
                log.error(f"Failed to save tracks to {stub_path}: {e}")
        
        return updated_tracks
    except Exception as e:
        log.error(f"Error processing video for jersey numbers: {e}")
        raise