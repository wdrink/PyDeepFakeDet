import os

import cv2
from tqdm import tqdm

from utils import (
    crop_img,
    gen_dirs,
    get_face_location,
    get_files_from_path,
    parse,
    static_shuffle,
    video2frames,
)


def parse_split(
    train_subjects,
    faces_path,
    hq_path,
    lq_path,
    f_hq,
    f_lq,
    label,
    samples,
    face_scale,
):
    for subject in tqdm(train_subjects):
        videos = [
            f
            for f in os.listdir(os.path.join(hq_path, subject))
            if f.endswith('.avi')
        ]
        assert len(videos) == 10
        for video in videos:
            gen_dirs(os.path.join(faces_path, 'higher_quality', subject))
            gen_dirs(os.path.join(faces_path, 'lower_quality', subject))
            # hq
            file_names, frames = video2frames(
                os.path.join(hq_path, subject, video), samples
            )
            crop_datas = [
                *map(get_face_location, frames, [face_scale] * len(frames))
            ]
            faces = [*map(crop_img, frames, crop_datas)]
            [
                *map(
                    cv2.imwrite,
                    [
                        os.path.join(
                            faces_path, 'higher_quality', subject, img_name
                        )
                        for img_name in file_names
                    ],
                    faces,
                    [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(faces),
                )
            ]
            f_hq.writelines(
                [
                    os.path.join('faces', 'higher_quality', subject, img_name)
                    + ' '
                    + label
                    + '\n'
                    for img_name in file_names
                ]
            )

            # lq
            _, frames = video2frames(
                os.path.join(lq_path, subject, video), samples
            )
            faces = [*map(crop_img, frames, crop_datas)]
            [
                *map(
                    cv2.imwrite,
                    [
                        os.path.join(
                            faces_path, 'lower_quality', subject, img_name
                        )
                        for img_name in file_names
                    ],
                    faces,
                    [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(faces),
                )
            ]
            f_lq.writelines(
                [
                    os.path.join('faces', 'lower_quality', subject, img_name)
                    + ' '
                    + label
                    + '\n'
                    for img_name in file_names
                ]
            )


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    train_hq_txt = os.path.join(faces_path, 'train_hq.txt')
    test_hq_txt = os.path.join(faces_path, 'test_hq.txt')
    train_lq_txt = os.path.join(faces_path, 'train_lq.txt')
    test_lq_txt = os.path.join(faces_path, 'test_lq.txt')
    f_train_hq = open(train_hq_txt, 'w')
    f_test_hq = open(test_hq_txt, 'w')
    f_train_lq = open(train_lq_txt, 'w')
    f_test_lq = open(test_lq_txt, 'w')

    hq_path = os.path.join(path, 'higher_quality')
    lq_path = os.path.join(path, 'lower_quality')
    subjects = get_files_from_path(hq_path)
    assert len(subjects) == 32
    static_shuffle(subjects)
    train_subjects = subjects[:22]
    test_subjects = subjects[22:]
    label = '1'

    print('Parsing train split...')
    parse_split(
        train_subjects,
        faces_path,
        hq_path,
        lq_path,
        f_train_hq,
        f_train_lq,
        label,
        samples,
        face_scale,
    )

    print('Parsing test split...')
    parse_split(
        test_subjects,
        faces_path,
        hq_path,
        lq_path,
        f_test_hq,
        f_test_lq,
        label,
        samples,
        face_scale,
    )

    f_train_hq.close()
    f_test_hq.close()
    f_train_lq.close()
    f_test_lq.close()


# 1 is fake
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale)
