import sys
import cv2
import faceNet
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--marginFactor', type=float, default=0.35, help='margin factor')
opt = parser.parse_args()

########## MODEL STUFF ########## 
if not torch.cuda.is_available():
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    raise RuntimeError('no CUDA. Unable to inference without a cuda device')
# Initialize network
print('Initalizing the network')
net = faceNet.faceNet_BN(classnum=10576, m = opt.marginFactor, feature = True )
state_dict = torch.load('./checkpoint/netFinal_8.pth')
net.load_state_dict(state_dict)
net = net.cuda(0)


video_capture = cv2.VideoCapture(0)

while True : 
    _, frame = video_capture.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # canvas, faceCount = detect(gray, frame, faceCount, rightNow)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
