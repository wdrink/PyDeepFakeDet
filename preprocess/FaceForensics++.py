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


def get_source_videos(videos):
    res = []
    for video in videos:
        res.append(video[:3] + '.mp4')
        res.append(video[4:])
    return res


def double_videos(videos):
    res = []
    for video in videos:
        res.append(video[4:7] + '_' + video[:3] + '.mp4')
    return res + videos


def get_ff_splits(path):
    videos = [
        video
        for video in get_files_from_path(path)
        if int(video[:3]) < int(video[4:7])
    ]
    assert len(videos) == 500, f'Missing some videos(expect 1000): {path}'
    static_shuffle(videos)
    train_split_m = videos[:360]
    val_split_m = videos[360:430]
    test_split_m = videos[430:]

    train_split_o = get_source_videos(train_split_m)
    val_split_o = get_source_videos(val_split_m)
    test_split_o = get_source_videos(test_split_m)

    train_split_m = double_videos(train_split_m)
    val_split_m = double_videos(val_split_m)
    test_split_m = double_videos(test_split_m)

    return (
        train_split_m + train_split_o,
        val_split_m + val_split_o,
        test_split_m + test_split_o,
    )


def get_DFD_splits(original_path, manipulated_path):
    train_list = [
        '01', '02', '03', '04', '06', '07',
        '09', '11', '12', '13', '14', '15',
        '18', '20', '21', '25', '26', '27',
    ]
    val_list = ['05', '08', '16', '17', '28']
    test_list = ['10', '19', '22', '23', '24']
    videos = get_files_from_path(original_path) + get_files_from_path(
        manipulated_path
    )
    train_split = []
    val_split = []
    test_split = []
    for video in videos:
        if video[:2] in train_list:
            train_split.append(video)
        elif video[:2] in val_list:
            val_split.append(video)
        elif video[:2] in test_list:
            test_split.append(video)
        else:
            assert False
    return train_split, val_split, test_split


def solve(
    dataset_path,
    faces_path,
    video_path,
    video_name,
    f,
    label,
    samples,
    face_scale,
    file_names=None,
    crop_datas=None,
):
    new_file_names, frames = video2frames(
        os.path.join(dataset_path, video_path, video_name), samples
    )
    if file_names is None:
        file_names = new_file_names
    file_names = [
        os.path.join(faces_path, video_path, file_name)
        for file_name in file_names
    ]
    if crop_datas is None:
        crop_datas = [
            *map(get_face_location, frames, [face_scale] * len(frames))
        ]
    assert len(frames) == len(file_names), f'{len(frames) != {len(file_names)}}'
    assert len(crop_datas) == len(
        file_names
    ), f'{len(crop_datas) != {len(file_names)}}'
    faces = [*map(crop_img, frames, crop_datas)]
    [
        *map(
            cv2.imwrite,
            file_names,
            faces,
            [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(faces),
        )
    ]
    if f is not None:
        f.writelines(
            [
                os.path.join('faces', video_path, os.path.basename(img_name))
                + ' '
                + label
                + '\n'
                for img_name in file_names
            ]
        )
    return new_file_names, crop_datas


def main(path, samples, face_scale, subset):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)

    f_train_raw = open(os.path.join(faces_path, 'train_raw.txt'), 'w')
    f_val_raw = open(os.path.join(faces_path, 'val_raw.txt'), 'w')
    f_test_raw = open(os.path.join(faces_path, 'test_raw.txt'), 'w')
    f_train_c23 = open(os.path.join(faces_path, 'train_c23.txt'), 'w')
    f_val_c23 = open(os.path.join(faces_path, 'val_c23.txt'), 'w')
    f_test_c23 = open(os.path.join(faces_path, 'test_c23.txt'), 'w')
    f_train_c40 = open(os.path.join(faces_path, 'train_c40.txt'), 'w')
    f_val_c40 = open(os.path.join(faces_path, 'val_c40.txt'), 'w')
    f_test_c40 = open(os.path.join(faces_path, 'test_c40.txt'), 'w')

    # FaceShifter don't have masks
    if subset == 'FF':
        datasets = [
            'original',
            'Deepfakes',
            'Face2Face',
            'FaceSwap',
            'NeuralTextures',
        ]
        train_split, val_split, test_split = get_ff_splits(
            os.path.join(
                path, 'manipulated_sequences', 'Deepfakes', 'raw', 'videos'
            )
        )
    else:
        datasets = ['DeepFakeDetection_original', 'DeepFakeDetection']
        train_split, val_split, test_split = get_DFD_splits(
            os.path.join(
                path,
                'manipulated_sequences',
                'DeepFakeDetection',
                'raw',
                'videos',
            )
        )
    for i, dataset in enumerate(datasets):
        print(f'Now parsing {dataset}...')
        label = '0 '
        if 'original' == dataset:
            raw_path = os.path.join(
                'original_sequences', 'youtube', 'raw', 'videos'
            )
        elif 'DeepFakeDetection_original' == dataset:
            raw_path = os.path.join(
                'original_sequences', 'actors', 'raw', 'videos'
            )
        else:
            raw_path = 'manipulated_sequences'
            raw_path = os.path.join(raw_path, dataset, 'raw', 'videos')
            label = '1 '
        label = label + str(i)

        gen_dirs(os.path.join(faces_path, raw_path))
        c23_path = raw_path.replace('raw', 'c23')
        gen_dirs(os.path.join(faces_path, c23_path))
        c40_path = raw_path.replace('raw', 'c40')
        gen_dirs(os.path.join(faces_path, c40_path))
        if 'original' not in dataset:
            masks_path = raw_path.replace('raw', 'masks')
            gen_dirs(os.path.join(faces_path, masks_path))

        raw_videos = get_files_from_path(os.path.join(path, raw_path))
        for video in tqdm(raw_videos):
            if video in train_split:
                f_raw = f_train_raw
                f_c23 = f_train_c23
                f_c40 = f_train_c40
            elif video in val_split:
                f_raw = f_val_raw
                f_c23 = f_val_c23
                f_c40 = f_val_c40
            else:
                f_raw = f_test_raw
                f_c23 = f_test_c23
                f_c40 = f_test_c40

            file_names, crop_datas = solve(
                path,
                faces_path,
                raw_path,
                video,
                f_raw,
                label,
                samples,
                face_scale,
            )
            solve(
                path,
                faces_path,
                c23_path,
                video,
                f_c23,
                label,
                samples,
                face_scale,
                file_names,
                crop_datas,
            )
            solve(
                path,
                faces_path,
                c40_path,
                video,
                f_c40,
                label,
                samples,
                face_scale,
                file_names,
                crop_datas,
            )
            if 'original' not in dataset:
                solve(
                    path,
                    faces_path,
                    masks_path,
                    video,
                    None,
                    None,
                    samples,
                    face_scale,
                    file_names,
                    crop_datas,
                )

    f_train_raw.close()
    f_val_raw.close()
    f_test_raw.close()
    f_train_c23.close()
    f_val_c23.close()
    f_test_c23.close()
    f_train_c40.close()
    f_val_c40.close()
    f_test_c40.close()


'''
change code 'return' to 'continue' to download the full datasets in download_ffdf.py
'''

if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale, args.subset)
