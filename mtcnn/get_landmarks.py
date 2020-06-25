from PIL import Image
from src import detect_faces
import glob
import argparse
import os.path as osp
import numpy as np
import code


def computeArea(points ):
    assert(points.shape[0] == 2 and points.shape[1] == 5)
    x1 = (points[:, 0] - points[:, 4] )
    x2 = (points[:, 1] - points[:, 3] )
    area = np.abs(x1[1] *x2[0] - x1[0] * x2[1] )
    return area

parser = argparse.ArgumentParser(description='Mostly to get the facial landmarks for the casia dataset')
parser.add_argument('--dataset', default='/home/joshuakang/datasets/CASIA-WebFace', type=str)
parser.add_argument('--output', type=str, default = '../cosFace/data/josh_landmarksMTCNN.txt')
parser.add_argument('--label', type=bool, default=True)
args = parser.parse_args()

#get the names of the directories (aka the subject ids)
names = glob.glob(osp.join(args.dataset, '*') )
names = [x for x in names if osp.isdir(x) ]
names = sorted(names )

#get all the paths to the images for each subject id
imgs = []
total_num_faces = 0
for x in names:
    current_id_images = []
    current_id_images = sorted(glob.glob(osp.join(x, '*.jpg') ) ) #change this based on the image types that you are processing
    imgs.append(current_id_images)
    total_num_faces += len(current_id_images)

with open(args.output, 'w') as fOut:
    cnt = 0
    for subject_idx, subject in enumerate(imgs):
        for image_name in subject:
            cnt += 1
            print(f'{cnt}/{total_num_faces}: {image_name}')
            img = Image.open(image_name)

            #for the greyscale images
            if np.asarray(img).ndim == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)

            _, landmarks = detect_faces(img )
            if len(landmarks) == 0: 
                print(f'Warning: SKIPPING because there are no detected faces in {image_name}')
                continue #Sorry the detection is not perfect (this will trigger cause some false negatives)

            faceNum = landmarks.shape[0]
            if faceNum > 1:
                print('Warning: %s faces have been detected!' % faceNum )
                        
            #get the largest face in the image if there are more than one
            largestMark = np.array(landmarks[0, :] ).reshape([2, 5] )
            largestArea = computeArea(largestMark )
            for n in range(1, faceNum):
                mark = np.array(landmarks[n, :] ).reshape([2, 5] )
                area =  computeArea(mark )
                if area > largestArea:
                    largestMark = mark
                    largestArea = area
    
            #use the largest image
            landmarks = np.transpose(largestMark, [1, 0] ).reshape(10 )

            fOut.write('%s\t' % '/'.join(image_name.split('/')[-2:] ) )
            fOut.write(f'{subject_idx}\t') #include the label of the face here
            for f in landmarks:
                fOut.write('%.3f\t' % f )
            fOut.write('\n')
    
    