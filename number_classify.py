import cv2
import numpy as np
import pickle

# 학습된 모델 로드
with open('random_forest_mnist.pkl', 'rb') as f:
    clf = pickle.load(f)

# 전처리 함수
def preprocess_digit(roi):
    """
    입력 영역(ROI)을 MNIST 데이터셋에 맞게 전처리.
    28x28 크기로 변환하고, 이진화 처리 후 flatten.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
    flattened = binary.flatten() / 255.0
    return flattened

# 마우스 클릭을 이용하여 숫자 그리기
drawing = False
points = []
predicted_number = None

def draw(event, x, y, flags, param):
    global drawing, points, predicted_number
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            cv2.line(img, points[-2], points[-1], (255, 255, 255), 10)  # 흰색으로 그리기
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 그린 숫자 예측
        roi = img.copy()
        # 그린 숫자 영역만 추출 (너무 작은 숫자들을 무시하기 위해 크기 조정 필요)
        x, y, w, h = cv2.boundingRect(np.array(points))
        if w > 10 and h > 10:  # 너무 작은 영역은 무시
            roi = img[y:y+h, x:x+w]
            processed_roi = preprocess_digit(roi)
            predicted_number = clf.predict([processed_roi])[0]

# 화면 초기화 (검은 배경)
img = np.zeros((500, 500, 3), dtype=np.uint8)

# 마우스 이벤트 처리
cv2.namedWindow('Draw Digits')
cv2.setMouseCallback('Draw Digits', draw)

while True:
    # 배경을 새로 고침 (매 프레임마다 초기화)
    display_img = img.copy()

    # 예측된 숫자 출력 (숫자가 예측되었을 때만)
    if predicted_number is not None:
        cv2.putText(display_img, f'Number is {int(predicted_number)}', (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 이미지 표시
    cv2.imshow('Draw Digits', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
