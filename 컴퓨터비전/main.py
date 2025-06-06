import cv2
import numpy as np

# Haar Cascade 얼굴 검출기 로드
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# DNN 얼굴 검출기 로드
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# 웹캠을 통한 실시간 영상 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환 (Haar Cascade 사용 시 필요)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar Cascade를 이용한 얼굴 검출
    haar_faces = haar_face_cascade.detectMultiScale(gray, 1.1, 4)

    # DNN을 이용한 얼굴 검출 (DNN 모델은 이미지를 300x300 크기로 변환해야 함)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(blob)
    dnn_faces = net.forward()

    # Haar Cascade 얼굴 영역에 대해 모자이크 처리
    for (x, y, w, h) in haar_faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (10, 10))  # 축소
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)  # 확대 (모자이크 효과)
        frame[y:y+h, x:x+w] = face

    # DNN 모델로 감지된 얼굴에 대해서도 모자이크 처리
    for i in range(dnn_faces.shape[2]):
        confidence = dnn_faces[0, 0, i, 2]
        if confidence > 0.5:  # confidence threshold (50%)
            box = dnn_faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (10, 10))  # 축소
            face = cv2.resize(face, (endX - startX, endY - startY), interpolation=cv2.INTER_NEAREST)  # 확대
            frame[startY:endY, startX:endX] = face

    # 결과 출력 (실시간 영상 표시)
    cv2.imshow('Mosaic Face Detection', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 종료
cap.release()
cv2.destroyAllWindows()
