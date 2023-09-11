import cv2

# 원본 동영상 파일 경로
original_video_path = '하이라이트 좌측 영상.mp4'

# 결과 동영상 파일 저장 디렉토리
output_directory = '출력영상/'

# 입장프레임.txt에서 프레임 번호를 읽어옵니다.
with open('입장프레임.txt', 'r') as file:
    frame_numbers = [int(line.strip()) for line in file]

# 원본 비디오 파일 열기
cap = cv2.VideoCapture(original_video_path)

# 비디오에서 프레임을 하나씩 읽어서 처리
for frame_number in frame_numbers:
    # 프레임 번호 주위 3초를 계산
    start_frame = max(1, frame_number - 90)  # 3초 전
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_number + 90)  # 3초 후

    # 결과 동영상 파일 생성
    output_video_name = f'{frame_number}.mp4'
    output_video_path = output_directory + output_video_name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (1020, 500))

    # 원본 비디오를 프레임 단위로 읽어서 저장
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)  # 시작 프레임으로 이동
    frame_count = start_frame

    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))
        out.write(frame)
        frame_count += 1

    # 파일을 닫습니다.
    out.release()

# 원본 동영상 파일 닫기
cap.release()

print(f"{len(frame_numbers)} 개의 영상 파일이 {output_directory}에 생성되었습니다.")
