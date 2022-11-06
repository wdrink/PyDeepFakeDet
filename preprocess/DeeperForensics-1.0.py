import os

from tqdm import tqdm

from utils import gen_dirs, parse, parse_video, read_txt

ms = [
    'end_to_end',
    'end_to_end_level_1',
    'end_to_end_level_2',
    'end_to_end_level_3',
    'end_to_end_level_4',
    'end_to_end_level_5',
    'end_to_end_mix_2_distortions',
    'end_to_end_mix_3_distortions',
    'end_to_end_mix_4_distortions',
    'end_to_end_random_level',
    'reenact_postprocess',
]
modes = ['train', 'val', 'test']


def parse_source_videos(source_videos):
    d = {}
    for video in source_videos:
        name = video[14:18]
        if name in d:
            d[name].append(video)
        else:
            d[name] = [video]
    return d


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)

    source_videos = read_txt(
        os.path.join(
            path, 'lists', 'source_videos_lists', 'source_videos_list.txt'
        )
    )
    source_videos += read_txt(
        os.path.join(
            path, 'lists', 'source_videos_lists', 'source_videos_extra_list.txt'
        )
    )
    source_videos = parse_source_videos(source_videos)
    for mode in modes:
        print(f'Parsing {mode} split...')
        with open(os.path.join(faces_path, mode + '.txt'), 'w') as f:
            videos = read_txt(
                os.path.join(path, 'lists', 'splits', mode + '.txt')
            )
            for video in tqdm(videos):
                # manipulated
                for m in ms:
                    parse_video(
                        path,
                        os.path.join('manipulated_videos', m),
                        video,
                        '1',
                        f,
                        samples,
                        face_scale,
                    )
                name = video[4:8]
                # original
                if name in source_videos:
                    for source_video in source_videos[name]:
                        parse_video(
                            path,
                            os.path.dirname(source_video),
                            os.path.basename(video),
                            '0',
                            f,
                            samples,
                            face_scale,
                        )
                    del source_videos[name]


# 1 is fake
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale)
