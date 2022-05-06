import argparse
import os
import pickle
import random
from multiprocessing import Process, Queue
from time import time

import cv2

parser = argparse.ArgumentParser('script', add_help=False)
if not os.path.exists('tmp'):
    os.mkdir('tmp')
parser.add_argument('--root_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--samples', type=int)
parser.add_argument('--process', default=1, type=int)
args = parser.parse_args()
video_list_path = os.path.join(args.root_dir, 'video_list.txt')
image_list_path = os.path.join(args.save_dir, 'image_list.txt')


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


def video2img(video_path, save_path_pattern, sample=8):
    flag = True
    for i in range(sample):
        if not os.path.exists(save_path_pattern % i):
            flag = False
            break
    if flag:
        return
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames == 0:
        print("err frame=0", video_path)
        return False, -2, []
    stride = num_frames // sample + 1
    order = [j for i in range(stride) for j in range(i, num_frames, stride)]
    if sample > len(order):
        sample = len(order)
    order = order[:sample]
    order = sorted(order)
    for i, num in enumerate(order):
        cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        img_path = save_path_pattern % i
        if not os.path.exists(img_path):
            flag, frame = cap.read()
            cv2.imwrite(img_path, frame)

    cap.release()


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


def solve(process_id, video_list, samples):
    image_list = []
    check = 1
    start_time = time()
    for i, line in enumerate(video_list):
        video_path = os.path.join(args.root_dir, line[0])
        save_path_pattern = os.path.join(
            args.save_dir, line[0][: line[0].find('.')] + '_%03d.jpg'
        )
        video2img(video_path, save_path_pattern, samples)
        for j in range(samples):
            save_path = save_path_pattern % j
            rel_path = os.path.relpath(save_path, args.save_dir)
            if os.path.exists(save_path):
                image_list.append([rel_path, *line[1:]])
        if i % check == check - 1:
            ET = time() - start_time
            ETA = ET / (i + 1) * (len(video_list) - i - 1)
            print(
                "process:%d %d/%d ET: %.2fmin ETA:%.2fmin "
                % (process_id, i + 1, len(video_list), ET / 60, ETA / 60)
            )
    with open('tmp/%d.pkl' % process_id, 'wb') as w:
        pickle.dump(image_list, w)


if __name__ == '__main__':
    video_list = read_list(video_list_path)
    image_list = []
    gen_dirs(args.root_dir, args.save_dir)
    # multi-process
    num_process = args.process
    sub_video_list = []
    n = len(video_list)
    step = n // num_process
    j = 0
    random.shuffle(video_list)
    for i in range(0, n, step):
        j += 1
        if j == num_process:
            sub_video_list.append(video_list[i:n])
            break
        else:
            sub_video_list.append(video_list[i : i + step])
    process_list = []
    Q = Queue()
    for i, item in enumerate(sub_video_list):
        cur_process = Process(target=solve, args=(i, item, args.samples))
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
    for ele in sub_image_list:
        for line in ele:
            image_list.append(line)
    write_list(image_list_path, image_list)
