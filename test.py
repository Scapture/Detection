import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8 모델 로드 (모델 경로를 사용자가 저장한 모델로 변경)
model_path = "/Users/taebeom/Documents/GitHub/Scapture/Detection/scapture_train4_best.pt"  # 사용자 모델 경로로 변경
model = YOLO(model_path)

# 비디오 파일 로드 (분석할 비디오 경로로 변경)
video_path = "/Users/taebeom/Documents/GitHub/Scapture/Detection/left.mp4"  # 사용자 비디오 경로로 변경
output_path = "/Users/taebeom/Documents/GitHub/Scapture/Detection/model_test_result.mp4"  # 결과 비디오 저장 경로

# 비디오 캡처 초기화
cap = cv2.VideoCapture(video_path)

# 비디오 속성 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 비디오 라이터 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 축구공 등장 횟수 초기화
football_count = 0

# 정확도 임계값 설정
confidence_threshold = 0.60

# 채도 조절 함수
def increase_saturation(frame, saturation_scale=1.5):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, saturation_scale)
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

# 비디오의 프레임을 하나씩 읽어서 객체 탐지 수행
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 채도 증가
    frame = increase_saturation(frame, 1.5)

    # 객체 탐지 수행
    results = model(frame)

    # 프레임에서 축구공 객체 카운트
    for result in results:
        boxes = result.boxes  # 탐지된 객체의 박스들
        for box in boxes:
            # box.cls는 클래스 인덱스를, box.conf는 확률(confidence)을 의미
            if box.cls == 0 and box.conf >= confidence_threshold:  # 'Football' 클래스의 인덱스가 0이라고 가정
                football_count += 1

        # 탐지된 객체를 포함한 프레임을 가져오기
        annotated_frame = result.plot()

        # 결과 프레임을 비디오 파일로 저장
        out.write(annotated_frame)

    # 결과 프레임을 화면에 표시 (원하는 경우)
    # cv2.imshow('YOLOv8 Object Detection', annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total number of football appearances: {football_count}")
