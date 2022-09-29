img ='/mnt/data/liyihui/face_data/DFDet/ana/ana_list/cnn_all_wrong_test_list.txt'
paths = open(img, 'r').readlines()
for path in paths:
    path = path.strip('\n').split()[0]
    save_dir = 'block12'+path.replace('/', '_')
    print(save_dir)