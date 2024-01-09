import cv2
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from flask import Flask
from flask_mqtt import Mqtt
from multiprocessing import Process, Queue, Value
import time

# Function to process each frame

# Paths for model files
protopath = "model/object_detection/MobileNetSSD_deploy.prototxt"
modelpath = "model/object_detection/MobileNetSSD_deploy.caffemodel"


# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Shared variables for sensor counts
Sensor2_People_Count = Value('i', 0)
Sensor2_Bike_Count = Value('i', 0)

class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
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
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

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



def process_frame(frame, detector):
    frame = imutils.resize(frame, width=720)
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    detections = detector.forward()

    people_cnt = 0
    bike_cnt = 0

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["person", "bicycle"]:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                color = (0, 0, 255) if CLASSES[idx] == "person" else (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                if CLASSES[idx] == "person":
                    people_cnt += 1
                else:
                    bike_cnt += 1

    return frame, people_cnt, bike_cnt

def Camera(camera_index, output_queue):
    # Initialize the CSI Camera
    camera = CSI_Camera()
    camera.open(gstreamer_pipeline(sensor_id=camera_index, ...))
    camera.start()

    # Load the model inside the process
    detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

    while True:
        ret, frame = camera.read()
        if ret:
            processed_frame, people_count, bike_count = process_frame(frame, detector)
            output_queue.put((camera_index, people_count, bike_count))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    camera.release()

def MQTT(input_queue, Sensor2_People_Count, Sensor2_Bike_Count):
    # Initialize camera counts
    camera_counts = {0: {'people': 0, 'bikes': 0}, 2: {'people': 0, 'bikes': 0}}

    # MQTT setup
    app = Flask(__name__)
    app.config['MQTT_BROKER_URL'] = 'broker.hivemq.com'
    app.config['MQTT_BROKER_PORT'] = 1883
    app.config['MQTT_USERNAME'] = ''
    app.config['MQTT_PASSWORD'] = ''
    app.config['MQTT_KEEPALIVE'] = 5
    app.config['MQTT_TLS_ENABLED'] = False
    mqtt = Mqtt(app)

    @mqtt.on_connect()
    def handle_connect(client, userdata, flags, rc):
        mqtt.subscribe('Dugeleg/VisionTest/Sensor2/Personer')
        mqtt.subscribe('Dugeleg/VisionTest/Sensor2/Sykkler')
        print("Subscribed to Dugeleg/VisionTest/Sensor2 topics")

    @mqtt.on_message()
    def handle_mqtt_message(client, userdata, message):
        if message.topic == 'Dugeleg/VisionTest/Sensor2/Personer':
            try:
                with Sensor2_People_Count.get_lock():
                    Sensor2_People_Count.value = int(message.payload.decode())
                print(f"Sensor 2 People Count: {Sensor2_People_Count.value}")
            except ValueError:
                print("Payload for People Count is not an integer")

        elif message.topic == 'Dugeleg/VisionTest/Sensor2/Sykkler':
            try:
                with Sensor2_Bike_Count.get_lock():
                    Sensor2_Bike_Count.value = int(message.payload.decode())
                print(f"Sensor 2 Bike Count: {Sensor2_Bike_Count.value}")
            except ValueError:
                print("Payload for Bike Count is not an integer")

    while True:
        while not input_queue.empty():
            camera_index, people_count, bike_count = input_queue.get()

            # Update counts for the respective camera
            camera_counts[camera_index]['people'] = people_count
            camera_counts[camera_index]['bikes'] = bike_count

            # Calculate total counts
            total_people_count = sum([counts['people'] for counts in camera_counts.values()])
            total_bike_count = sum([counts['bikes'] for counts in camera_counts.values()])

            # Publish total counts to MQTT
            mqtt.publish("Dugeleg/VisionTest/Sensor1/Personer", str(total_people_count))
            mqtt.publish("Dugeleg/VisionTest/Sensor1/Sykkler", str(total_bike_count))

        time.sleep(0.1)

def print_sensor_values(camera_counts, Sensor2_People_Count, Sensor2_Bike_Count):
    while True:
        print("\nCurrent Sensor Values:")
        print("Camera Counts:", camera_counts)
        print("Sensor 2 People Count:", Sensor2_People_Count.value)
        print("Sensor 2 Bike Count:", Sensor2_Bike_Count.value)
        time.sleep(5)  # Adjust the sleep time as needed


# Main process
if __name__ == '__main__':
    output_queue = Queue()
    camera_counts = {0: {'people': 0, 'bikes': 0}, 2: {'people': 0, 'bikes': 0}}  # Define camera counts

    # Create and start camera processes
    cam1 = Process(target=Camera, args=(0, output_queue))
    cam2 = Process(target=Camera, args=(1, output_queue))  # Assuming camera index 1 for the second camera
    cam1.start()
    cam2.start()

    # Create and start MQTT process
    mqtt_process = Process(target=MQTT, args=(output_queue, Sensor2_People_Count, Sensor2_Bike_Count))
    mqtt_process.start()

    # Create and start the print values process
    print_values_process = Process(target=print_sensor_values, args=(camera_counts, Sensor2_People_Count, Sensor2_Bike_Count))
    print_values_process.start()

    # Join the processes
    cam1.join()
    cam2.join()
    mqtt_process.join()
    print_values_process.join()

    cv2.destroyAllWindows()
