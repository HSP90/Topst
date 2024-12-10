import numpy as np
import cv2
import pickle

# 학습된 모델 로드
with open('random_forest_mnist.pkl', 'rb') as f:
    clf = pickle.load(f)

def preprocess_digit(roi):
    """
    입력 영역(ROI)을 MNIST 데이터셋에 맞게 전처리.
    28x28 크기로 변환하고, 이진화 처리 후 flatten.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY_INV)
    flattened = binary.flatten() / 255.0
    return flattened

def detect_and_recognize_digits(image):
    """
    이미지에서 숫자를 탐지하고 분류.
    """
    # 그레이스케일 변환 및 대비 조정
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 대비 개선
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # 윤곽선 탐지
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 윤곽선으로부터 영역 추출
        x, y, w, h = cv2.boundingRect(contour)

        # 크기 조건으로 필터링
        if 20 < w < 100 and 20 < h < 100 and 0.5 < w / h < 2:
            roi = image[y:y+h, x:x+w]
            processed_roi = preprocess_digit(roi)

            try:
                # 숫자 예측
                pred = clf.predict([processed_roi])[0]

                # 사각형 그리기 및 예측값 출력
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f'{int(pred)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                # 예외 처리: 잘못된 데이터 스킵
                print(f"Error processing contour: {e}")

    return image

print("모델 로드 완료")
cap = cv2.VideoCapture(1)  # 0은 기본 웹캠, 1은 외부 웹캠
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("카메라 연결 완료")

while True:
    ret, img = cap.read()
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    # 숫자 탐지 및 분류
    result = detect_and_recognize_digits(img)

    # 결과 화면 출력
    cv2.imshow('Digit Recognition', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
