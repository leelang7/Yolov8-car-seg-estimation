from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# YOLOv8 모델 로드
model = YOLO("240730_IS.pt")  # 여기에 올바른 YOLOv8 모델 파일 경로를 지정합니다.

# 이미지 저장 경로
SAVED_IMAGE_PATH = "./seg_results/"
if not os.path.exists(SAVED_IMAGE_PATH):
    os.makedirs(SAVED_IMAGE_PATH)

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    # YOLOv8 세그멘테이션 수행
    results = model(img, task="segment")[0]  # 세그멘테이션 작업 수행

    # 결과 처리
    response = []
    total_est_cost = 0  # 총 픽셀 수 초기화
    
    if results.masks is not None:  # 세그멘테이션 마스크 체크
        for i in range(len(results.masks.xy)):  # 각 마스크에 대해
            # 마스크에 대한 정보 접근
            segmentation_mask = results.masks.xy[i]  # NumPy 배열 형태
            
            # 이진 마스크 생성
            mask_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            
            # Polygon Fill: 세그멘테이션 마스크를 이진 마스크로 변환
            cv2.fillPoly(mask_image, [segmentation_mask.astype(np.int32)], 1)  # 객체에 해당하는 부분을 1로 설정
            
            # 픽셀 수 계산
            pixel_count = np.sum(mask_image)  # 마스크에서 1의 개수를 세서 픽셀 수를 계산
            
            total_est_cost += pixel_count  # 총 픽셀 수 누적
            
            label = int(results.boxes.cls[i].item()) if results.boxes.cls is not None else None
            confidence = float(results.boxes.conf[i].item()) if results.boxes.conf is not None else None
            
            response.append({
                "label": label,
                "confidence": confidence,
                "pixel_count": int(pixel_count),  # 개별 픽셀 수 추가
            })

    # 고유 ID 생성
    unique_id = str(uuid.uuid4())
    
    # 합쳐진 이미지 저장 경로 설정
    combined_image_path = os.path.join(SAVED_IMAGE_PATH, f"{unique_id}_combined.jpg")

    # 결과 시각화 이미지 생성
    annotated_img = results.plot()  # YOLOv8의 결과를 시각화한 이미지를 생성

    # 원본 이미지와 시각화 이미지를 나란히 결합
    combined_img = np.hstack((img, annotated_img))

    # 결합된 이미지 저장
    if cv2.imwrite(combined_image_path, combined_img):
        print(f"Combined image saved at {combined_image_path}")
    else:
        print("Failed to save combined image")

    return jsonify({
        "results": response,
        "combined_image_path": combined_image_path,
        "total_est_cost": int(total_est_cost)  # 총 픽셀 수를 int로 변환하여 반환
    })

if __name__ == '__main__':
    app.run(debug=True)
