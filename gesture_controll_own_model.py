# import cv2
# import threading
# import numpy as np
# from ultralytics import YOLO
# from collections import deque
# import time

# class HandTracker:
#     def __init__(self, model_path='best_openvino_model', conf=0.25, imgsz=320):
#         try:
#             self.model = YOLO(model_path, task='detect')
#         except Exception as e:
#             print(f"ERROR: {e}")
#             raise

#         self.conf = conf
#         self.imgsz = imgsz
        
#         self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.cap.set(cv2.CAP_PROP_FPS, 60)
        
#         self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
#         self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

#         self.margin = 0.15 

#         self.hand_x = self.cam_width / 2
#         self.hand_y = self.cam_height / 2
#         self.target_x = self.hand_x
#         self.target_y = self.hand_y
        
#         self.is_detected = False
        
#         self.smoothing_factor = 0.4 
        
#         self.running = True
#         self.thread = threading.Thread(target=self._update, daemon=True)
#         self.thread.start()

#     def _update(self):
#         while self.running:
#             for _ in range(3):
#                 self.cap.grab()
            
#             success, frame = self.cap.read()
#             if not success:
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             results = self.model.predict(
#                 frame, 
#                 conf=self.conf, 
#                 imgsz=self.imgsz, 
#                 stream=True, 
#                 verbose=False
#             )
            
#             for r in results:
#                 if r.boxes and len(r.boxes) > 0:
#                     box = r.boxes[0].xyxy[0].cpu().numpy()
#                     self.target_x = (box[0] + box[2]) / 2
#                     self.target_y = (box[1] + box[3]) / 2
#                     self.is_detected = True
#                     break
#                 else:
#                     self.is_detected = False

#             self.hand_x += (self.target_x - self.hand_x) * self.smoothing_factor
#             self.hand_y += (self.target_y - self.hand_y) * self.smoothing_factor
            
#             time.sleep(0.001)

#     def get_position(self, target_width=1280, target_height=720):
#         m_x = self.cam_width * self.margin
#         m_y = self.cam_height * self.margin
        
#         active_w = self.cam_width - (2 * m_x)
#         active_h = self.cam_height - (2 * m_y)
        
#         norm_x = (self.hand_x - m_x) / active_w
#         norm_y = (self.hand_y - m_y) / active_h
        
#         norm_x = max(0, min(1, norm_x))
#         norm_y = max(0, min(1, norm_y))
        
#         return (int(norm_x * target_width), int(norm_y * target_height))

#     def get_hand_visible(self):
#         return self.is_detected

#     def stop(self):
#         self.running = False
#         if self.cap.isOpened():
#             self.cap.release()

# if __name__ == "__main__":
#     tracker = HandTracker()
#     try:
#         while True:
#             pos = tracker.get_position(1280, 720)
#             status = "OK" if tracker.get_hand_visible() else "LOST"
#             print(f"[{status}] Pozycja gry: {pos}      ", end="\r")
#             time.sleep(0.01)
#     except KeyboardInterrupt:
#         tracker.stop()