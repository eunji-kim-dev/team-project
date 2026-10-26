import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

##import 설치용
# pip install opencv-python numpy pyrealsense2

## 실행 시 카메라 작동
## 's'키 입력시 해당 폴더에 rgb 카메라 이미지, depth 카메라 측정 data 색상 보정 이미지
##                  depth 측정 데이터, 실제 거리 데이터(depth 측정값*scale) 저장
##            덮어씌우질 가능성 있음 주의요망
##'q'키 입력 시 카메라 종료

# 폴더명 지정
folderdir_img='./image'
folderdir_depth_img='./depth_img'
folderdir_depth_raw='./depth_raw'
folderdir_depth_meter='./depth_meter'
folders = [folderdir_img,folderdir_depth_img,folderdir_depth_raw,folderdir_depth_meter]
# 폴더 생성
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 1. 프로그램이 시작할 때 image 폴더에 있는 파일 개수를 측정
if os.path.exists(folderdir_img):
    save_count = len(os.listdir(folderdir_img))
else:
    save_count = 0

# 기존 파일이 있는지 체크하고 번호 올리기
color_filename = f"{folderdir_img}/image_bgr8_{save_count}.jpg"
while os.path.exists(color_filename):
    save_count += 1
    color_filename = f"{folderdir_img}/image_bgr8_{save_count}.jpg"

print(f"시작 번호 설정 완료: {save_count}번부터 이어서 저장합니다.")

# 파이프라인 및 설정
pipeline = rs.pipeline()
config = rs.config()

# 1. Depth와 Color 둘 다 활성화
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작 및 프로필 확보
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# 센서에서 실제 Depth Scale(보통 0.001) 가져오기
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"현재 카메라 Depth Scale: {depth_scale}")

print("--------------------------------------------------")
print("촬영을 시작합니다.")
print(" - 's' 키를 누르면 현재 화면과 깊이 데이터가 저장됩니다.")
print(" - 'q' 키를 누르면 프로그램이 종료됩니다.")
print("--------------------------------------------------")

try:
    latest = None
    last_frame_time = time.monotonic()

    while True:
        # 1. 타임아웃(200ms)을 적용하여 프레임 대기 (무한정 멈춤 방지)
        ok, frames = pipeline.try_wait_for_frames(200)
        if ok:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                # 원본 데이터를 복사하여 안전하게 보관 (latest 갱신)
                raw_depth = np.asanyarray(depth_frame.get_data()).copy()
                color_image = np.asanyarray(color_frame.get_data()).copy()
                
                # 정렬 수행
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                if aligned_depth_frame:
                    aligned_depth = np.asanyarray(aligned_depth_frame.get_data()).copy()
                    
                    # 최신 프레임 묶음(latest) 업데이트
                    latest = {
                        'color': color_image,
                        'raw_depth': raw_depth,
                        'aligned_depth': aligned_depth
                    }
                    last_frame_time = time.monotonic()

        if latest is None:
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
                continue

        
        # aligned_frames = align.process(frames)
        # depth_frame = aligned_frames.get_depth_frame()
        # color_frame = aligned_frames.get_color_frame()

        # 데이터 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Depth 이미지 컬러맵 처리 (시각화용)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW
        )
        # depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        # RGB와 Depth 영상을 가로로 나란히 합쳐서 출력
        # combined_image = np.hstack((color_image, depth_colormap))
        # cv2.imshow("Color & Depth Simultaneous View", combined_image)
        # 프래임 드랍으로인해 각각출력
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)

        # 키보드 입력 처리
        key = cv2.waitKey(10) & 0xFF

        # 1. 'q'를 누르면 종료
        if key == ord("q"):
            break

        # 2. 's'를 누르면 이미지 및 딥스 데이터 저장
        elif key == ord("s"):


            # Raw Depth 데이터에 scale을 곱해서 미터(m) 단위의 실수 배열로 변환
            depth_meters = depth_image.astype(np.float32) * depth_scale

            # 파일 이름 생성
            color_filename = f"{folderdir_img}/image_bgr8_{save_count}.jpg"
            depth_img_filename = f"{folderdir_depth_img}/depth_img_{save_count}.jpg"
            depth_raw_filename = f"{folderdir_depth_raw}/depth_raw_{save_count}.npy"
            depth_meter_filename = f"{folderdir_depth_meter}/depth_meter_{save_count}.npy"


            # 저장 실행
            cv2.imwrite(color_filename, color_image)          # RGB 이미지
            cv2.imwrite(depth_img_filename, depth_colormap)   # 시각화 딥스
            np.save(depth_raw_filename, depth_image)          # 16비트 Raw 딥스 (mm 단위)
            np.save(depth_meter_filename, depth_meters)       # 실제 거리 배열 (m 단위)

            print(f"\n[성공] {save_count}번째 데이터가 저장되었습니다!")
            print(f" - RGB: {color_filename}")
            print(f" - 시각화 딥스: {depth_img_filename}")
            print(f" - Raw 딥스(mm): {depth_raw_filename}")
            print(f" - 실제 거리(m): {depth_meter_filename}")
            #파일 num 증가
            save_count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()