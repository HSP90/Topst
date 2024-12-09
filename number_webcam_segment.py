import cv2
import numpy as np
import pickle
import os
import time
import sys

# GPIO 관련 설정
GPIO_EXPORT_PATH = "/sys/class/gpio/export"
GPIO_UNEXPORT_PATH = '/sys/class/gpio/unexport'
GPIO_DIRECTION_PATH_TEMPLATE = '/sys/class/gpio/gpio{}/direction'
GPIO_VALUE_PATH_TEMPLATE = '/sys/class/gpio/gpio{}/value'
GPIO_BASE_PATH_TEMPLATE = '/sys/class/gpio/gpio{}'

# GPIO PIN 번호 설정
a, b, c, d, e, f, g, dp = 81, 82, 89, 88, 87, 83, 84, 90
gpio = [a, b, c, d, e, f, g, dp]
gpio_array = [
    [a, b, c, d, e, f],       # 0
    [b, c],                   # 1
    [a, b, d, e, g],          # 2
    [a, b, c, d, g],          # 3
    [b, c, f, g],             # 4
    [a, c, d, f, g],          # 5
    [a, c, d, e, f, g],       # 6
    [a, b, c],                # 7
    [a, b, c, d, e, f, g],    # 8
    [a, b, c, d, f, g]        # 9
]

def is_gpio_exported(gpio_number):
    gpio_base_path = GPIO_BASE_PATH_TEMPLATE.format(gpio_number)
    return os.path.exists(gpio_base_path)

def export_gpio(gpio_number):
    if not is_gpio_exported(gpio_number):
        try:
            with open(GPIO_EXPORT_PATH, 'w') as export_file:
                export_file.write(str(gpio_number))
        except IOError as e:
            print(f"Error exporting GPIO: {e}")
            sys.exit(1)

def unexport_gpio(gpio_number):
    try:
        with open(GPIO_UNEXPORT_PATH, 'w') as unexport_file:
            unexport_file.write(str(gpio_number))
    except IOError as e:
        print(f"Error unexporting GPIO: {e}")
        sys.exit(1)

def set_gpio_direction(gpio_number, direction):
    gpio_direction_path = GPIO_DIRECTION_PATH_TEMPLATE.format(gpio_number)
    try:
        with open(gpio_direction_path, 'w') as direction_file:
            direction_file.write(direction)
    except IOError as e:
        print(f"Error setting GPIO direction: {e}")
        sys.exit(1)

def set_gpio_value(gpio_number, value):
    gpio_value_path = GPIO_VALUE_PATH_TEMPLATE.format(gpio_number)
    try:
        with open(gpio_value_path, 'w') as value_file:
            value_file.write(str(value))
    except IOError as e:
        print(f"Error setting GPIO value: {e}")
        sys.exit(1)

def gpio_num_on(gpio_num):
    for pin in gpio_num:
        set_gpio_value(pin, 1)

def gpio_num_off(gpio_num):
    for pin in gpio_num:
        set_gpio_value(pin, 0)

# 모든 GPIO 초기화
def reset_all_gpio():
    for pin in gpio:
        set_gpio_value(pin, 0)

# 숫자 인식 및 세그먼트 출력
def detect_and_recognize_digits(image, model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # MNIST 크기
    normalized = resized / 255.0
    flattened = normalized.flatten().reshape(1, -1)

    pred = model.predict(flattened)[0]  # 예측 숫자

    # 7세그먼트에 출력
    reset_all_gpio()
    if 0 <= pred <= 9:
        gpio_num_on(gpio_array[pred])
        print(f"Detected number: {pred}")

# 모델 로드
with open('random_forest_mnist.pkl', 'rb') as f:
    clf = pickle.load(f)

# GPIO 초기화
for pin in gpio:
    export_gpio(pin)
    set_gpio_direction(pin, "out")
reset_all_gpio()

# 카메라 설정
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

print("Camera ready. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_and_recognize_digits(frame, clf)
        cv2.imshow("Digit Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    reset_all_gpio()
    for pin in gpio:
        unexport_gpio(pin)
