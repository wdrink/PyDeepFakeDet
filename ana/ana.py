root_dir = '/mnt/data/liyihui/face_data/DFDet/Output/'
save_dir = '/mnt/data/liyihui/face_data/DFDet/ana/ana_list/'

cnns = ['efficientnet', 'f3net', 'mat', 'resnet50', 'xception']
vits = ['vit', 'm2tr']


def from_root_all_output_to_wrong(dataset='vit'):
    file = open(root_dir + dataset + '.txt', 'r')
    dict = {'a': 0}
    lines = file.readlines()
    i = 0
    with open(save_dir + dataset + '.txt', 'w') as f:
        for line in lines:
            # i+=1
            s = line.strip('\n').split()
            # if s[0] in dict.keys():
            #     print(i, s, dict[s[0]])
            # else:
            #     dict[s[0]] = i
            if len(s) > 2 and float(s[1]) - float(s[2]) != 0:
                f.write(s[0] + '\n')


def check_all_wrong(models=cnns):
    path_dict = {}
    for model in models:
        lines = open(save_dir + model + '.txt', 'r').readlines()
        for line in lines:
            s = line.strip('\n').split()
            if not s[0] in path_dict.keys():
                path_dict[s[0]] = 1
            else:
                path_dict[s[0]] += 1
    reference = {}
    with open('/mnt/data/liyihui/face_data/DFDet/dataset_list/test_list.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            s = line.strip('\n').split()
            reference[s[0]] = s[1]
    with open(save_dir + 'vits_all_wrong_test_list.txt', 'w') as f:
        for k in path_dict:
            cnt = path_dict[k]
            v = reference[k]
            if cnt == 2:
                f.write(k + ' ' + v + ' ' + str(cnt) + '\n')


vname = lambda v, nms: [vn for vn in nms if id(v) == id(nms[vn])][0]


def wrong_in_A_but_not_B(A='cnns', B='vits'):
    fp = save_dir + A + '_all_wrong_test_list.txt'
    a_lines = open(fp, 'r').readlines()
    b_models = globals()[B]
    ref = {}
    for line in a_lines:
        s = line.strip('\n').split()
        ref[s[0]] = s[1]
    for b_model in b_models:
        b_lines = open(save_dir + b_model + '.txt', 'r').readlines()
        for b_line in b_lines:
            s = b_line.strip('\n').split()
            if b_line[0] in ref.keys():
                ref.pop(b_line[0])
    with open(save_dir + f'wrong_in_{A}_but_no_{B}.txt', 'w') as f:
        for k in ref:
            v = ref[k]
            f.write(k + ' ' + v + '\n')


def move_jpg():
    import os
    dir = '/mnt/data/liyihui/face_data/DFDet/ana/ana_visual/xception/'
    lines = open(save_dir + 'wrong_in_cnns_but_no_vits.txt', 'r').readlines()
    for line in lines:
        s = line.strip('\n').split()
        name = 'block12' + s[0].replace('/', '_')
        os.system(f'cp {dir}{name} /mnt/data/liyihui/face_data/DFDet/ana/ana_visual/wrong_in_cnns_but_no_vits/')


if __name__ == '__main__':
    pass
