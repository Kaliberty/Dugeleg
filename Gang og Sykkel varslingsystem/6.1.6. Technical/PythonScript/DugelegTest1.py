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
CONFIDENCE_THRESHOLD = 0.7 #70%

# Shared variables for sensor counts
Sensor2_People_Count = Value('i', 0)
Sensor2_Bike_Count = Value('i', 0)

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
    # Load the model inside the process
    detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if ret:
            processed_frame, people_count, bike_count = process_frame(frame, detector)
            cv2.imshow(f"Camera {camera_index}", processed_frame)
            output_queue.put((camera_index, people_count, bike_count))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

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
            mqtt.publish("Dugeleg/VisionTest/Sensor1/Personer", str(Sensor2_People_Count))
            mqtt.publish("Dugeleg/VisionTest/Sensor1/Sykkler", str(Sensor2_Bike_Count))

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

    cam1 = Process(target=Camera, args=(0, output_queue))
    cam2 = Process(target=Camera, args=(1, output_queue))
    mqtt = Process(target=MQTT, args=(output_queue, Sensor2_People_Count, Sensor2_Bike_Count))

    cam1.start()
    cam2.start()
    mqtt.start()

    # Initialize and start the print values process
    print_values_process = Process(target=print_sensor_values, args=(camera_counts, Sensor2_People_Count, Sensor2_Bike_Count))
    print_values_process.start()

    cam1.join()
    cam2.join()
    mqtt.join()
    print_values_process.join()

    cv2.destroyAllWindows()
