import os

root_dir = '/mnt/data/liyihui/face_data/DFDet/'
datasets = ['origin', 'NeuralTexture', 'FaceSwap', 'F2F', 'Deepfakes']
mode = 'train'

with open(root_dir + 'dataset_list/' + mode + '_list.txt','w') as f:
    for dataset in datasets:
        path = root_dir + dataset + '/face_' + mode
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split('.')[-1] == 'jpg':
                    f.write(os.path.join(root, file) + ' ')
                    if dataset == 'origin':
                        f.write('0\n')
                    else:
                        f.write('1\n')
