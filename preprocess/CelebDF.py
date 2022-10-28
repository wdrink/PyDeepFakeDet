import os

from tqdm import tqdm

from utils import get_files_from_path, parse, parse_video, static_shuffle


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    if not os.path.exists(faces_path):
        os.mkdir(faces_path)
    train_info_txt = os.path.join(faces_path, 'train_info.txt')
    val_info_txt = os.path.join(faces_path, 'val_info.txt')
    test_info_txt = os.path.join(faces_path, 'test_info.txt')

    test_txt = os.path.join(path, 'List_of_testing_videos.txt')
    with open(test_txt, 'r') as f:
        test_lines = f.read().splitlines()
    test_videos = []
    with open(test_info_txt, 'w') as f:
        print('Generating test split...')
        for l in tqdm(test_lines):
            label, video = l.split(' ')
            test_videos.append(video)
            parse_video(
                path,
                os.path.dirname(video),
                os.path.basename(video),
                label,
                f,
                samples,
                face_scale,
            )

    celeb_real_path = os.path.join(path, 'Celeb-real')
    celeb_synthesis_path = os.path.join(path, 'Celeb-synthesis')
    youtube_real_path = os.path.join(path, 'YouTube-real')
    celeb_real_videos = [
        os.path.join('Celeb-real', video)
        for video in get_files_from_path(celeb_real_path)
    ]
    celeb_synthesis_videos = [
        os.path.join('Celeb-synthesis', video)
        for video in get_files_from_path(celeb_synthesis_path)
    ]
    youtube_real_videos = [
        os.path.join('YouTube-real', video)
        for video in get_files_from_path(youtube_real_path)
    ]

    celeb_real_videos = [
        *filter(lambda v: v not in test_videos, celeb_real_videos)
    ]
    celeb_synthesis_videos = [
        *filter(lambda v: v not in test_videos, celeb_synthesis_videos)
    ]
    youtube_real_videos = [
        *filter(lambda v: v not in test_videos, youtube_real_videos)
    ]

    all_videos = [('1', v) for v in celeb_real_videos + youtube_real_videos] + [
        ('0', v) for v in celeb_synthesis_videos
    ]

    static_shuffle(all_videos)
    val_size = 500
    train_split = all_videos[: len(all_videos) - val_size]
    with open(train_info_txt, 'w') as f:
        print('Generating train split...')
        for l in tqdm(train_split):
            label, video = l
            parse_video(
                path,
                os.path.dirname(video),
                os.path.basename(video),
                label,
                f,
                samples,
                face_scale,
            )
    val_split = all_videos[len(all_videos) - val_size :]
    with open(val_info_txt, 'w') as f:
        print('Generating val split...')
        for l in tqdm(val_split):
            label, video = l
            parse_video(
                path,
                os.path.dirname(video),
                os.path.basename(video),
                label,
                f,
                samples,
                face_scale,
            )


# real is 1
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale)
