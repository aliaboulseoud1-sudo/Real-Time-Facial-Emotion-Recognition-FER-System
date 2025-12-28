import cv2
import torch
import numpy as np
import time
from collections import deque

from model import ModelBuilder
from config import Config


class LiveEmotionDetector:

    
    def __init__(
        self,
        checkpoint_path: str = None,
        config: Config = None,
        camera_id: int = 0,
        enable_face_detection: bool = False,
        smoothing_window: int = 5
    ):

        self.config = config if config else Config
        self.device = self.config.DEVICE
        self.camera_id = camera_id
        self.enable_face_detection = enable_face_detection
        self.smoothing_window = smoothing_window
        
        self.prediction_history = deque(maxlen=smoothing_window)
        
        print("\n" + "="*70)
        print("LIVE EMOTION DETECTOR INITIALIZATION".center(70))
        print("="*70 + "\n")
        
        self._load_model(checkpoint_path)
        
        if self.enable_face_detection:
            self._load_face_detector()
        
        self._setup_camera()
        
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        self._setup_preprocessing()
        
        print("="*70 + "\n")
    
    def _load_model(self, checkpoint_path: str):
        print("📦 Loading emotion recognition model...")
        
        self.model = ModelBuilder(
            num_classes=self.config.NUM_CLASSES,
            model_name=self.config.MODEL_NAME,
            pretrained=False,
            freeze_backbone=False,
            dropout_p=self.config.DROPOUT_P,
            print_summary=False
        ).to(self.device)
        
        checkpoint_path = checkpoint_path if checkpoint_path else self.config.BEST_MODEL_PATH
        
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        torch.set_grad_enabled(False)
        
        print("✓ Model loaded successfully!")
    
    def _load_face_detector(self):
        print("👤 Loading face detector...")
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("⚠ Warning: Face detector not loaded. Disabling face detection.")
            self.enable_face_detection = False
        else:
            print("✓ Face detector loaded!")
    
    def _setup_camera(self):
        print(f"📹 Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Error: Cannot open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera initialized successfully!")
    
    def _setup_preprocessing(self):
        self.input_size = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def preprocess_frame(self, frame):

        resized = cv2.resize(frame, self.input_size)
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        img = rgb.astype(np.float32) / 255.0
        
        img = np.transpose(img, (2, 0, 1)).copy()
        
        img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
        
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor.unsqueeze(0)
    
    def predict_emotion(self, frame):
        input_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)
            
            emotion = self.config.EMOTION_LABELS[pred_idx.item()]
            conf_value = confidence.item()
        
        return emotion, conf_value
    
    def smooth_prediction(self, emotion: str, confidence: float):

        self.prediction_history.append((emotion, confidence))
        
        if len(self.prediction_history) < 2:
            return emotion, confidence
        
        emotion_counts = {}
        for pred_emotion, pred_conf in self.prediction_history:
            if pred_emotion not in emotion_counts:
                emotion_counts[pred_emotion] = []
            emotion_counts[pred_emotion].append(pred_conf)
        
        most_common = max(emotion_counts.items(), key=lambda x: len(x[1]))
        smoothed_emotion = most_common[0]
        smoothed_confidence = np.mean(most_common[1])
        
        return smoothed_emotion, smoothed_confidence
    
    def detect_faces(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        return faces
    
    def draw_info(self, frame, emotion: str, confidence: float, fps: float, face_rect=None):

        height, width = frame.shape[:2]
        
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        emotion_text = f"Emotion: {emotion}"
        cv2.putText(
            frame, emotion_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 2
        )
        
        conf_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(
            frame, conf_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 2
        )
        
        emotion_colors = {
            "Angry": (0, 0, 255),
            "Disgust": (0, 255, 0),
            "Fear": (128, 0, 128),
            "Happy": (0, 255, 255),
            "Sad": (255, 0, 0),
            "Surprise": (255, 165, 0),
            "Neutral": (128, 128, 128)
        }
        
        color = emotion_colors.get(emotion, (255, 255, 255))
        cv2.circle(frame, (width - 50, 50), 30, color, -1)
        cv2.circle(frame, (width - 50, 50), 30, (255, 255, 255), 2)
    
    def run(self, record_output: str = None, show_fps: bool = True):

        print("\n" + "="*70)
        print("STARTING LIVE EMOTION DETECTION".center(70))
        print("="*70 + "\n")
        
        if self.enable_face_detection:
            print("👤 Face detection: ENABLED")
        else:
            print("👤 Face detection: DISABLED (processing full frame)")
        
        print("\nControls:")
        print("  'q' or 'ESC' - Quit")
        print("  'r' - Toggle recording")
        print("  's' - Save screenshot")
        print("  'f' - Toggle FPS display")
        print("\n" + "="*70 + "\n")
        
        video_writer = None
        is_recording = False
        
        if record_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (640, 480)
            video_writer = cv2.VideoWriter(record_output, fourcc, fps, frame_size)
            is_recording = True
            print(f"🎥 Recording to: {record_output}")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Failed to capture frame!")
                    break
                
                if self.enable_face_detection:
                    faces = self.detect_faces(frame)
                    
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        face_frame = frame[y:y+h, x:x+w]
                        
                        emotion, confidence = self.predict_emotion(face_frame)
                        emotion, confidence = self.smooth_prediction(emotion, confidence)
                        
                        self.draw_info(frame, emotion, confidence, 
                                     np.mean(self.fps_history) if self.fps_history else 0,
                                     (x, y, w, h))
                    else:
                        cv2.putText(frame, "No face detected", (20, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    emotion, confidence = self.predict_emotion(frame)
                    emotion, confidence = self.smooth_prediction(emotion, confidence)
                    
                    self.draw_info(frame, emotion, confidence,
                                 np.mean(self.fps_history) if self.fps_history else 0)
                
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                
                if is_recording and video_writer is not None:
                    video_writer.write(frame)
                
                cv2.imshow("Live Emotion Detection - Press 'q' to quit", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    if video_writer is None:
                        output_path = f"recording_{int(time.time())}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
                        is_recording = True
                        print(f"🎥 Recording started: {output_path}")
                    else:
                        is_recording = not is_recording
                        status = "resumed" if is_recording else "paused"
                        print(f"🎥 Recording {status}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
        finally:
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """Release resources"""
        print("\n" + "="*70)
        print("SHUTTING DOWN".center(70))
        print("="*70 + "\n")
        
        if video_writer is not None:
            video_writer.release()
            print("✓ Video saved")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Total frames processed: {self.frame_count}")
        if self.fps_history:
            print(f"✓ Average FPS: {np.mean(self.fps_history):.2f}")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-Time Emotion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (full frame processing)
  python live_detection.py
  
  # With face detection
  python live_detection.py --face-detection
  
  # Record output
  python live_detection.py --record output.mp4
  
  # Use specific checkpoint
  python live_detection.py --checkpoint checkpoints/best_model.pth
  
  # Use different camera
  python live_detection.py --camera 1
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: best_model.pth)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--face-detection",
        action="store_true",
        help="Enable face detection before emotion recognition"
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Record output to video file"
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=5,
        help="Smoothing window size (default: 5 frames)"
    )
    
    args = parser.parse_args()
    
    try:
        detector = LiveEmotionDetector(
            checkpoint_path=args.checkpoint,
            config=Config,
            camera_id=args.camera,
            enable_face_detection=args.face_detection,
            smoothing_window=args.smoothing
        )
        
        detector.run(record_output=args.record)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Program ended successfully!")