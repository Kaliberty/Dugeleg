import cv2
import threading
import numpy as np
import jetson.inference
import jetson.utils

class CSI_Camera:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(gstreamer_pipeline_string, cv2.CAP_GSTREAMER)
            self.grabbed, self.frame = self.video_capture.read()
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                if grabbed:
                    cuda_frame = jetson.utils.cudaFromNumpy(frame)
                    detections = self.net.Detect(cuda_frame, overlay="box,labels,conf")
                    frame = jetson.utils.cudaToNumpy(cuda_frame)
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError as e:
                print("Could not read image from camera:", e)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def camera_thread(camera):
    camera.start()
    cv2.namedWindow(f"Camera {camera.camera_id}", cv2.WINDOW_AUTOSIZE)
    while camera.running:
        grabbed, frame = camera.read()
        if grabbed:
            cv2.imshow(f"Camera {camera.camera_id}", frame)
        if cv2.waitKey(1) == 27: # ESC key
            break
    camera.stop()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    left_camera = CSI_Camera(camera_id=0)
    right_camera = CSI_Camera(camera_id=1)

    left_camera.open(gstreamer_pipeline(sensor_id=0))
    right_camera.open(gstreamer_pipeline(sensor_id=1))

    left_thread = threading.Thread(target=camera_thread, args=(left_camera,))
    right_thread = threading.Thread(target=camera_thread, args=(right_camera,))

    left_thread.start()
    right_thread.start()

    left_thread.join()
    right_thread.join()

    left_camera.stop()
    right_camera.stop()
    left_camera.release()
    right_camera.release()
