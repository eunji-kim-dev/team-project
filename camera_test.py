<<<<<<< HEAD
print('hi')
=======
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

# 파이프라인 및 설정
pipeline = rs.pipeline()
config = rs.config()

# 1. Depth와 Color 둘 다 활성화
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작 및 프로필 확보
profile = pipeline.start(config)

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
    save_count = 0  # 저장 파일 이름을 구분하기 위한 카운트

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 데이터 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Depth 이미지 컬러맵 처리 (시각화용)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # RGB와 Depth 영상을 가로로 나란히 합쳐서 출력
        combined_image = np.hstack((color_image, depth_colormap))
        cv2.imshow("Color & Depth Simultaneous View", combined_image)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF

        # 1. 'q'를 누르면 종료
        if key == ord("q"):
            break

        # 2. 's'를 누르면 이미지 및 딥스 데이터 저장
        elif key == ord("s"):
            save_count += 1

            # Raw Depth 데이터에 scale을 곱해서 미터(m) 단위의 실수 배열로 변환
            depth_meters = depth_image.astype(np.float32) * depth_scale

            # 파일 이름 생성
            color_filename = f"capture_{save_count}_color.jpg"
            depth_vis_filename = f"capture_{save_count}_depth_vis.jpg"
            depth_raw_filename = f"capture_{save_count}_depth_raw.npy"
            depth_meter_filename = f"capture_{save_count}_depth_meters.npy"

            # 저장 실행
            cv2.imwrite(color_filename, color_image)          # RGB 이미지
            cv2.imwrite(depth_vis_filename, depth_colormap)   # 시각화 딥스
            np.save(depth_raw_filename, depth_image)          # 16비트 Raw 딥스 (mm 단위)
            np.save(depth_meter_filename, depth_meters)       # 실제 거리 배열 (m 단위)

            print(f"\n[성공] {save_count}번째 데이터가 저장되었습니다!")
            print(f" - RGB: {color_filename}")
            print(f" - 시각화 딥스: {depth_vis_filename}")
            print(f" - Raw 딥스(mm): {depth_raw_filename}")
            print(f" - 실제 거리(m): {depth_meter_filename}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
>>>>>>> 2ba43e0cd0fd82a2b49df004ced69767b903a5e4
