import cv2
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class HandTracker:
    """
    A class to track hand movement in real-time using a YOLO model optimized with OpenVINO.
    It runs the detection in a separate thread to ensure high performance in games.
    """

    def __init__(self, model_path='best_openvino_model', conf=0.2, imgsz=320):
        """
        Initializes the tracker, camera, and processing thread.
        
        :param model_path: Path to the exported OpenVINO model folder.
        :param conf: Confidence threshold for hand detection (0.0 to 1.0).
        :param imgsz: Input image size for the YOLO model.
        """
        # Load the YOLO model (OpenVINO format)
        try:
            self.model = YOLO(model_path, task='detect')
        except Exception as e:
            print(f"ERROR: Could not load the model from {model_path}. {e}")
            raise

        self.conf = conf
        self.imgsz = imgsz
        
        # Initialize camera using DirectShow (CAP_DSHOW) for better stability on Windows
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Get camera frame dimensions
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Fallback to standard resolution if camera doesn't report size
        if self.cam_width == 0: self.cam_width = 640
        if self.cam_height == 0: self.cam_height = 480

        # Hand position coordinates and detection status
        self.hand_x = 0
        self.hand_y = 0
        self.is_detected = False
        
        # Deque objects for position smoothing (Moving Average Filter)
        self.smooth_x = deque(maxlen=5)
        self.smooth_y = deque(maxlen=5)
        
        # Background thread initialization
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """
        Internal method that runs in a separate thread. 
        Continuously captures frames, runs detection, and updates coordinates.
        """
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue
            
            # Flip frame horizontally for a natural 'mirror' effect
            frame = cv2.flip(frame, 1)
            
            # Perform inference (stream=True for better memory management)
            results = self.model.predict(
                frame, 
                conf=self.conf, 
                imgsz=self.imgsz, 
                stream=True, 
                verbose=False
            )
            
            found_this_frame = False
            for r in results:
                if r.boxes and len(r.boxes) > 0:
                    # Select the bounding box with the highest confidence
                    box = r.boxes[0].xyxy[0].cpu().numpy()
                    
                    # Calculate the center point of the hand
                    raw_x = int((box[0] + box[2]) // 2)
                    raw_y = int((box[1] + box[3]) // 2)
                    
                    # Append coordinates for smoothing
                    self.smooth_x.append(raw_x)
                    self.smooth_y.append(raw_y)
                    
                    # Calculate the smoothed (average) position
                    self.hand_x = int(np.mean(self.smooth_x))
                    self.hand_y = int(np.mean(self.smooth_y))
                    found_this_frame = True
            
            self.is_detected = found_this_frame
            
            # Prevent the thread from hogging CPU resources
            time.sleep(0.001)

    def get_position(self, target_width=1280, target_height=720):
        """
        Calculates and returns the hand position scaled to the game window size.
        
        :param target_width: Width of the game window.
        :param target_height: Height of the game window.
        :return: A tuple (x, y) representing coordinates in the game window.
        """
        if not self.is_detected:
            # Return last known position if hand is lost
            return (self.hand_x, self.hand_y) 
        
        # Proportional scaling formula: (hand_pos / cam_max) * target_max
        game_x = (self.hand_x / self.cam_width) * target_width
        game_y = (self.hand_y / self.cam_height) * target_height
        
        return (int(game_x), int(game_y))

    def get_hand_visible(self):
        """
        Returns the current detection status.
        :return: True if a hand is currently visible, False otherwise.
        """
        return self.is_detected

    def stop(self):
        """
        Gracefully stops the background thread and releases camera resources.
        """
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    # Diagnostic mode: run this file directly to test the tracker
    print("Initializing HandTracker Diagnostic Mode...")
    tracker = HandTracker()
    
    try:
        while True:
            if tracker.get_hand_visible():
                pos = tracker.get_position(1280, 720)
                print(f"Hand Detected! Game Position: {pos}      ", end="\r")
            else:
                print("Searching for hand...                         ", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        tracker.stop()