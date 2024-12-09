import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# MNIST 데이터셋 로드
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist.data)
y = np.array(mnist.target, dtype=int)

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 훈련
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# 모델 평가
y_pred = clf.predict(X_test)
print(f"숫자 분류 정확도: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 모델을 파일로 저장
with open('random_forest_mnist.pkl', 'wb') as f:
    pickle.dump(clf, f)