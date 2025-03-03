import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from openvino.runtime import Core
from torchvision import models
import torch.nn.functional as F

# OpenVINO 모델 로드
ie = Core()
model_path = "facedetection_모델 경로/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
compiled_model = ie.compile_model(model_path, "CPU")
infer_request = compiled_model.create_infer_request()
input_layer = compiled_model.input(0)

# 사용자 정의 ResNet50 모델 로드
model_path = "학습된 가중치 모델 경로"
resnet50 = models.resnet50()
num_classes = 6  # 사용자 모델의 클래스 수
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)  # 출력 레이어 변경
resnet50.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
resnet50.eval()

transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    input_blob = cv2.resize(frame, (672, 384))  # 모델 입력 크기에 맞추기
    input_blob = input_blob.transpose((2, 0, 1))  # 채널 변환
    input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32)
    
    # 얼굴 검출 수행
    infer_request.infer(inputs={input_layer.any_name: input_blob})
    results = infer_request.get_output_tensor().data[0][0]
    
    for result in results:
        conf = result[2]  # 신뢰도
        if conf > 0.5:  # 신뢰도 임계값
            xmin = int(result[3] * w)
            ymin = int(result[4] * h)
            xmax = int(result[5] * w)
            ymax = int(result[6] * h)
            face = frame[ymin:ymax, xmin:xmax]
            
            # ResNet50을 사용하여 얼굴 분류
            if face.size > 0:
                face = transform(face).unsqueeze(0)
                with torch.no_grad():
                    output = resnet50(face)
                    probabilities = F.softmax(output, dim=1)  # 확률 계산
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[0, predicted_class].item() * 100  # 확률(%)
                    
                # 얼굴 주변에 박스 및 클래스 출력
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'Class: {predicted_class} ({confidence:.2f}%)', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Face Detection and Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
