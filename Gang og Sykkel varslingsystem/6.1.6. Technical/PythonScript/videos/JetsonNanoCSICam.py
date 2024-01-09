import cv2
import threading
import numpy as np
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import paho.mqtt.client as mqtt

# Global MQTT Configuration
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = 'your/mqtt/topic'

# Object Detection Function
def detect_objects(frame, net):
    detections = net.Detect(frame)
    # Process detections
    # Add your code here to handle each detection, e.g., drawing bounding boxes

# MQTT Client Setup
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

def setup_mqtt_client():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    return client

# CSI Camera Class
class CSI_Camera:
    def __init__(self):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

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
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread != None:
            self.read_thread.join()

# GStreamer Pipeline
def gstreamer_pipeline(sensor_id=0, capture_width=1920, capture_height=1080, display_width=1920, display_height=1080, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# Main Function
def run_cameras():
    net = detectNet("ssd-mobilenet-v2", threshold=0.5)
    mqtt_client = setup_mqtt_client()

    left_camera = CSI_Camera()
    left_camera.open(gstreamer_pipeline(sensor_id=0, flip_method=0, display_width=960, display_height=540))

    right_camera = CSI_Camera()
    right_camera.open(gstreamer_pipeline(sensor_id=1, flip_method=0, display_width=960, display_height=540))

    left_camera.start()
    right_camera.start()

    try:
        while True:
            _, left_frame = left_camera.read()
            _, right_frame = right_camera.read()

            detect_objects(left_frame, net)
            detect_objects(right_frame, net)

            combined_frame = np.hstack((left_frame, right_frame))

            # MQTT Publish (define what data to send)
            # mqtt_client.publish(MQTT_TOPIC, payload)

            cv2.imshow("Dual CSI Cameras", combined_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
                break
    finally:
        left_camera.stop()
        right_camera.stop()
        cv2.destroyAllWindows()
        mqtt_client.disconnect()

if __name__ == "__main__":
    run_cameras()
