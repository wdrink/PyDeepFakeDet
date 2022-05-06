import argparse
import os
import pickle
import random
from multiprocessing import Process, Queue
from time import time

import cv2
import torch
from retinaface.pre_trained_models import get_model

parser = argparse.ArgumentParser('script', add_help=False)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--process', default=8, type=int)
args = parser.parse_args()


def read_list(path):
    ls = []
    with open(path, 'r') as f:
        for line in f:
            ls.append(line.strip().split())
    return ls


def write_list(path, ls):
    with open(path, 'w') as w:
        for line in ls:
            print(' '.join([str(ele) for ele in line]), file=w)


def gen_dirs(raw, new):
    if not os.path.exists(new):
        os.mkdir(new)
    for root, dirs, files in os.walk(raw):
        for dir in dirs:
            whole_path = os.path.join(root, dir)
            rel_path = os.path.relpath(whole_path, raw)
            new_path = os.path.join(new, rel_path)
            if not os.path.exists(new_path):
                os.mkdir(new_path)


def can_seg(img_path, save_path, model=None, scale=1.3):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotation = model.predict_jsons(
        img, confidence_threshold=0.3
    ) 
    if len(annotation[0]['bbox']) == 0:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./tmp2/%d.jpg' % (random.randint(1, 100)), img)
        return False
    x1, y1, x2, y2 = annotation[0]['bbox']
    x1, y1, x2, y2 = list(
        map(
            int,
            [
                x1 - (x2 - x1) * (scale - 1) / 2,
                y1 - (y2 - y1) * (scale - 1) / 2,
                x2 + (x2 - x1) * (scale - 1) / 2,
                y2 + (y2 - y1) * (scale - 1) / 2,
            ],
        )
    )
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    img_face = img[y1:y2, x1:x2, :]
    img_face = cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_face)
    return True


def solve(process_id, raw_list):
    device = 'cuda:%d' % process_id
    model = get_model("resnet50_2020-07-20", max_size=1024, device=device)
    model.eval()
    new_list = []
    check = 20
    start_time = time()
    for i, line in enumerate(raw_list):
        raw_path = os.path.join(args.root_dir, line[0])
        save_path = os.path.join(
            args.save_dir, os.path.relpath(raw_path, args.root_dir)
        )
        try:
            if 'tmp' not in raw_path:
                if os.path.exists(save_path) or can_seg(
                    raw_path, save_path, model
                ):
                    new_list.append(line)
        except Exception as e:
            print(e, raw_path)
            exit(0)
        if i % check == check - 1:
            ET = time() - start_time
            start_time = time()
            ETA = ET / check * (len(raw_list) - i - 1)
            print(
                "process:%d %d/%d ET:%.2fmin ETA:%.2fmin "
                % (process_id, i + 1, len(raw_list), ET / 60, ETA / 60)
            )
    with open('tmp/%d.pkl' % process_id, 'wb') as w:
        pickle.dump(new_list, w)


if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    torch.multiprocessing.set_start_method('spawn')
    raw_list_path = os.path.join(args.root_dir, 'image_list.txt')
    new_list_path = os.path.join(args.save_dir, 'image_list.txt')
    raw_list = read_list(raw_list_path)
    new_list = []
    gen_dirs(args.root_dir, args.save_dir)

    # multi-process
    num_process = args.process
    sub_raw_list = []
    n = len(raw_list)
    step = n // num_process
    j = 0
    random.shuffle(raw_list)
    for i in range(0, n, step):
        j += 1
        if j == num_process:
            sub_raw_list.append(raw_list[i:n])
            break
        else:
            sub_raw_list.append(raw_list[i : i + step])
    process_list = []
    Q = Queue()
    for i, item in enumerate(sub_raw_list):
        cur_process = Process(target=solve, args=(i, item))
        process_list.append(cur_process)
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    # merge
    print("merge")
    sub_image_list = []
    for i in range(num_process):
        with open("tmp/%d.pkl" % i, 'rb') as f:
            sub_image_list.append(pickle.load(f))
    image_list = []
    # print(sub_image_list)
    for ele in sub_image_list:
        for line in ele:
            image_list.append(line)
    write_list(new_list_path, image_list)
