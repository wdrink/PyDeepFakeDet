import os

# path = '/mnt/data/liyihui/face_data/DFDet/origin/origin_video/'
#
# with open(path+'video_list.txt', mode='w') as f:
#     lists = os.listdir(path)
#     for i in lists:
#         f.write(i+'\n')

path = '/mnt/data/liyihui/face_data/DFDet/F2F/F2F_face_train/image_list.txt'
mode = 'train'
save_dir = '/mnt/data/liyihui/face_data/DFDet/F2F/face_all/image_list_train.txt'

# with open(path, 'r') as f:
#     for line in f:
#         line = line.strip('\n')
#         print(line)

f = open(path, 'r')
lines = f.readlines()
with open(save_dir, 'w') as ff:
    for line in lines:
        line = line.strip('\n')
        ff.write(line + ' 1\n')
