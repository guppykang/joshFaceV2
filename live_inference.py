import sys
import code
from mtcnn.src import detect_faces
import cv2
import cosFace.faceNet as faceNet
import torch
from torch.autograd import Variable
import torch.functional as F
import argparse
from PIL import Image
import numpy as np
from cosFace.matlab_cp2tform import get_similarity_transform_for_PIL

parser = argparse.ArgumentParser()
parser.add_argument('--marginFactor', type=float, default=0.35, help='margin factor')
opt = parser.parse_args()


def alignment(img, landmark, cropSize=(96,112), refLandmark=[ 
    [30.2946, 51.6963],[65.5318, 51.5014],[48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]):

    #reshape and retype
    landmark = np.array(landmark, dtype=np.float32).reshape(5,2)
    refLandmark = np.array(refLandmark, dtype=np.float32).reshape(5,2)
    img = Image.fromarray(np.uint8(img)) #since the entire library was written to work with PIL images (sorry)


    tfm = get_similarity_transform_for_PIL(landmark, refLandmark )
    img = img.transform(cropSize, Image.AFFINE,
            tfm.reshape(6), resample=Image.BILINEAR)
    img = np.asarray(img )
    if len(img.shape ) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2) #if the image is a greyscale, then we triplicate (is that a word)
    else:
        img = img[:, :, ::-1] #since we are doing inference on BGR

    img = np.transpose(img, [2, 0, 1] )
    return img

########## MODEL STUFF ########## 
if not torch.cuda.is_available():
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    raise RuntimeError('no CUDA. Unable to inference without a cuda device')
# Initialize network
print('Initalizing the network')
net = faceNet.faceNet_BN(classnum=10576, m = opt.marginFactor)
state_dict = torch.load('./cosFace/checkpoint/netFinal_8.pth')
net.load_state_dict(state_dict)
net = net.cuda(0)


video_capture = cv2.VideoCapture(0)

while True: 
    _, frame = video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the model was indeed trained on BGR but all my alignment functions reverse it. So deal with it
    frame = Image.fromarray(np.uint8(frame)) #since the entire library was written to work with PIL images (sorry)
    bounding_boxes, landmarks = detect_faces(frame, live_inference=True)
    frame = np.ascontiguousarray(frame) #since we changed it to a PIL image, so change back to [H, W, 3]

    for box_idx, box in enumerate(bounding_boxes):
        cropped_face = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2]), :] #maybe add a +/- 10 pixels here in case the bounding boxes are too strict
        aligned_face = alignment(cropped_face, landmarks[box_idx]) #crop and align the face to the preset landmark locations
        aligned_face = aligned_face.reshape((1,3,112,96)) 

        aligned_face = (aligned_face.astype(np.float32 ) - 127.5) / 128 #not actually 100% sure why I do this lol
        aligned_face = Variable(torch.from_numpy(aligned_face)).float().cuda()

        pred = net(aligned_face)[0].cpu().data.numpy().squeeze() #aint this line just satisfying
        pred_class = np.argmax(pred) 
        print(pred_class)
        if pred_class == 10575: #aka me
            print('me!')

        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (36,255,12), 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #Change back lol poggers pepe head
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
