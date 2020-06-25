```bash
├── cosFace
│   ├── casia_train.py
│   ├── casia_train_summary_CosFace_20200625-115034
│   │   └── events.out.tfevents.1593111034.joshuakang-Alienware
│   ├── checkpoint
│   │   ├── accuracy.npy
│   │   ├── casia_train.py
│   │   ├── dataLoader.py
│   │   ├── faceNet.py
│   │   ├── images.png
│   │   ├── inference.py
│   │   ├── lfw_eval.py
│   │   ├── loss.npy
│   │   ├── matlab_cp2tform.py
│   │   ├── net_1.pth
│   │   ├── net_2.pth
│   │   ├── net_3.pth
│   │   ├── net_4.pth
│   │   ├── net_5.pth
│   │   ├── net_6.pth
│   │   ├── net_7.pth
│   │   ├── netFinal_8.pth
│   │   ├── trainingLog_0.txt
│   │   ├── trainingLog_1.txt
│   │   ├── trainingLog_2.txt
│   │   ├── trainingLog_3.txt
│   │   ├── trainingLog_4.txt
│   │   ├── trainingLog_5.txt
│   │   ├── trainingLog_6.txt
│   │   └── trainingLog_7.txt
│   ├── data
│   │   ├── casia_landmarkMTCNN.txt
│   │   ├── casia_landmark.txt
│   │   ├── josh_landmarksMTCNN.txt
│   │   ├── lfw_landmark.txt
│   │   └── pairs.txt
│   ├── dataLoader.py
│   ├── faceNet.py
│   ├── inference.py
│   ├── lfw_eval.py
│   ├── matlab_cp2tform.py
│   ├── plot_graphs.ipynb
│   └── __pycache__
│       ├── dataLoader.cpython-37.pyc
│       ├── faceNet.cpython-37.pyc
│       └── matlab_cp2tform.cpython-37.pyc
├── mtcnn
│   ├── example.jpg
│   ├── get_landmarks.py
│   ├── LICENSE
│   └── src
│       ├── box_utils.py
│       ├── detector.py
│       ├── first_stage.py
│       ├── get_nets.py
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── box_utils.cpython-37.pyc
│       │   ├── detector.cpython-37.pyc
│       │   ├── first_stage.cpython-37.pyc
│       │   ├── get_nets.cpython-37.pyc
│       │   ├── __init__.cpython-37.pyc
│       │   └── visualization_utils.cpython-37.pyc
│       ├── visualization_utils.py
│       └── weights
│           ├── onet.npy
│           ├── pnet.npy
│           └── rnet.npy
└── README.md
```
# MTCNN for facial landmark detection
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

# 20 residual Layer'ed (batch-norm) CNN using cosFace loss function
[CosFace: Large Margin Cosine Loss for Deep Face Recognition] (https://arxiv.org/pdf/1801.09414.pdf)
Fully trained model on CASIA + 215 images of Me (could have done transfer learning, but meh this is way cooler). Can then finetune with more images of me.   
Model : (https://drive.google.com/file/d/1UJW8chHcD8KEl28yGSy3vwD2KzOMb1em/view?usp=sharing) ~0.98 training accuracy, ~1.92 Training Loss, 10575(CASIA) + 1(me) classes
