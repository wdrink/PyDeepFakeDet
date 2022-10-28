import os

from tqdm import tqdm

from utils import (
    gen_dirs,
    get_files_from_path,
    parse,
    parse_video,
    static_shuffle,
)


def get_all_videos(real_videos):
    fake_videos = []
    for v in real_videos:
        fake_videos.append(v[:4] + '_fake.mp4')
    return fake_videos + real_videos


def get_splits(videos):
    static_shuffle(videos)
    return (
        get_all_videos(videos[:30]),
        get_all_videos(videos[30:39]),
        get_all_videos(videos[39:]),
    )  # 30*2, 9*2, 10*2


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    train_txt = os.path.join(faces_path, 'train.txt')
    val_txt = os.path.join(faces_path, 'val.txt')
    test_txt = os.path.join(faces_path, 'test.txt')
    fake_path = os.path.join(path, 'fake')
    real_path = os.path.join(path, 'real')
    fake_videos = get_files_from_path(fake_path)
    real_videos = get_files_from_path(real_path)
    assert len(fake_videos) == 49
    assert len(real_videos) == 49
    train_split, val_split, test_split = get_splits(real_videos)
    f_train = open(train_txt, 'w')
    f_val = open(val_txt, 'w')
    f_test = open(test_txt, 'w')

    all_videos = fake_videos + real_videos
    for v in tqdm(all_videos):
        if v in train_split:
            f = f_train
        elif v in val_split:
            f = f_val
        elif v in test_split:
            f = f_test
        else:
            assert False, 'Unknow file'
        if v[4] == '_':
            k = 'fake'
            label = '1'
        else:
            k = 'real'
            label = '0'
        parse_video(path, k, v, label, f, samples, face_scale)

    f_train.close()
    f_val.close()
    f_test.close()


# 0 is real
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale)
