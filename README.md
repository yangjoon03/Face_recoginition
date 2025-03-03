## 서론
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

## 코드
* 해당 코드는 테스트를 위한 코드로 동적으로 세팅 되어 있지 않음.
* train 80% val 20%



## 디렉토리 구조
📂 프로젝트 폴더<br>
 ├── 📂 train/   <br>
 │    ├── 📂 c0/<br>
 │    │    ├── img_1.jpg<br>
 │    │    ├── img_2.jpg<br>
 │    │<br>
 │    ├── 📂 c1/<br>
 │    │    ├── img_3.jpg<br>
 │    │    ├── img_4.jpg<br>

<br>

## face recognition 과정
1. openVINO face_detection(ads0001) 얼굴 인식
2. 인식한 부분에 대하여 CNN모델 학습
3. 얼굴인식된 부분에 대하여 CNN 추론 진행
<br>


## 테스트 환경(CPU only)
* CPU : i5 12400
* face detection 실시간 가능
* face recognition 5명까지 실시간 가능

## BTS 훈련
* 데이터 출처
* 해당데이터에서 추가적인 가공 진행 후 학습 진행하였음.
* 개인적인 학습용도로 활용. 학습된 모델에 대해서는 공유 X
* 라벨 : ['jhope', 'jimin', 'jin', 'jungkook', 'rm', 'suga', 'v']
* ipynb를 보게되면 val 정확도가 낮은 것을 볼 수 있다. 파라미터를 조절하면 더 좋은 결과를 얻을 수 있을 것이다.
* 학습 데이터의 해상도가 높지 않아 높은 정확도를 보이진 않는다.


## face_detection
![face_detection](https://github.com/user-attachments/assets/d6c79df9-28c1-44d4-ae6a-b360f29238af)
<br>

## face_recognition
![image](https://github.com/user-attachments/assets/26f2947f-f0ce-43d4-8d9b-515276bf96f4)
* ✔ 제이홉,뷔,정국,RM,지민,진,슈가


해당 프로젝트를 하고있는 하드웨어 라떼판다3에서는 추가적인 작업을 하기에는 제약이 있어 yolo5모델을 사용할 예정이므로 후의 코드는 yolo5에서 진행.

[face_detection]: https://github.com/yangjoon03/openVINO
[Pytorch]: https://pytorch.kr/get-started/previous-versions/


