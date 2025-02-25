## 서론
* 전체 코드 X
* CNN,Resnet 모델 학습 기본코드
* face detection에 대해서는 [face_detection] 참조
<br>

## 가상환경
```bush
pip install openvino
pip install torch torchvision torchaudio (CPU)
Pip install numpy
Pip install opencv-python
Pip install tqdm
```
✔GPU 사용을 원할 경우 [Pytorch] 그래픽 카드에 맞는 버전 설치
<br>
<br>
<br>


## face recoginition 과정
1. openVINO face_detection(ads0001) 얼굴 인식
2. 인식한 부분에 대하여 CNN모델 학습
3. 얼굴인식된 부분에 대하여 CNN 추론 진행
<br>


## face_detection
![face_detection](https://github.com/user-attachments/assets/d6c79df9-28c1-44d4-ae6a-b360f29238af)
<br>

## face_recoginition
![face_recoginition](https://github.com/user-attachments/assets/bf40a958-90d0-4861-8b91-13e1b6946855)<br>
✔예시 이미지로 누구인지 분류는 하지 못함.

[face_detection]: https://github.com/yangjoon03/openVINO
[Pytorch]: https://pytorch.kr/get-started/previous-versions/


