# coding: utf-8
import time
import queue
import threading
import cv2
import numpy as np

# Try importing pyvirtualcam safely to prevent app crashes if dependencies are missing
PYVIRTUALCAM_AVAILABLE = False
try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    pass

class VirtualCameraStreamer:
    """
    A thread-safe, high-performance wrapper for pyvirtualcam.
    Uses a background worker thread and a queue to stream frames smoothly
    without causing interface lag or locking the main Gradio execution thread.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=2)  # Cap size at 2 to avoid latency accumulation
        self.cam = None
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.last_error = ""

    def is_available(self):
        """Returns True if the pyvirtualcam package is installed."""
        return PYVIRTUALCAM_AVAILABLE

    def start(self):
        """Starts the virtual camera backend and background sender thread."""
        with self.lock:
            if self.running:
                return True
            
            if not PYVIRTUALCAM_AVAILABLE:
                self.last_error = "pyvirtualcam library is not installed. Run: pip install pyvirtualcam"
                print(f"[VirtualCamera] {self.last_error}")
                return False

            try:
                # Initialize pyvirtualcam (will search for installed virtual webcams like Unity Capture or OBS)
                self.cam = pyvirtualcam.Camera(
                    width=self.width, 
                    height=self.height, 
                    fps=self.fps
                )
                print(f"[VirtualCamera] Successfully opened device: {self.cam.device}")
                self.last_error = ""
            except Exception as e:
                self.last_error = (
                    f"No virtual camera device detected (e.g., Unity Video Capture is not active/installed). "
                    f"Details: {str(e)}"
                )
                print(f"[VirtualCamera] {self.last_error}")
                self.cam = None
                return False

            # Start background thread to consume frames from the queue
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._stream_loop, 
                daemon=True,
                name="VirtualCameraStreamerWorker"
            )
            self.worker_thread.start()
            return True

    def stop(self):
        """Gracefully stops the streaming thread and closes the virtual camera."""
        with self.lock:
            self.running = False
            
        if self.worker_thread:
            # Drain queue to unblock worker if it's waiting on get()
            try:
                self.frame_queue.put_nowait(None)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        with self.lock:
            if self.cam:
                try:
                    self.cam.close()
                except Exception as e:
                    print(f"[VirtualCamera] Error closing camera: {e}")
                self.cam = None
            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            print("[VirtualCamera] Stopped streaming.")

    def send_frame(self, frame):
        """
        Pushes a new frame into the queue to be sent to the virtual webcam.
        
        Args:
            frame (numpy.ndarray): An RGB frame (shape: H x W x 3).
        """
        if not self.running or self.cam is None:
            return

        try:
            h, w = frame.shape[:2]
            
            # 1. Handle resizing if dimensions do not match the virtual camera's configured dimensions
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
                
            # 2. Add to queue (non-blocking). If queue is full, drop the oldest frame
            # to maintain real-time low latency.
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)
        except Exception as e:
            print(f"[VirtualCamera] Error queueing frame: {e}")

    def _stream_loop(self):
        """Background loop that reads frames from the queue and sends them to the driver."""
        print("[VirtualCamera] Background loop started.")
        while True:
            with self.lock:
                if not self.running:
                    break
            
            frame = self.frame_queue.get()
            if frame is None:
                break
                
            try:
                # pyvirtualcam expects RGB frames by default
                self.cam.send(frame)
                
                # Enforce pacing based on target FPS to prevent overloading drivers
                self.cam.sleep_until_next_frame()
            except Exception as e:
                print(f"[VirtualCamera] Loop error: {e}")
                time.sleep(0.01)
                
        print("[VirtualCamera] Background loop exited.")
