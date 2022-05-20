## Introduction

We use the retinaface to preprocess the original dataset. 

## Requirements

* retinaface_pytorch([https://pypi.org/project/retinaface-pytorch/](https://pypi.org/project/retinaface-pytorch/))
* opencv-python
* numpy

## Run 

There are two steps for data preparation:

- Extract the frames from videos (since most datasets only provide video data).

- Extract the face region in each frame.

### Extract frames

Multi threads are used to extract video frames. You need to prepare the data set in advance.

```python
python preprocess/extract_frames.py --root_dir ROOT_PATH --save_dir SAVE_DIR --samples 128

optional arguments:
  -h, --help,    Show this help message and exit
  --root_dir,    DATASET root path
  --save_dir,    Frames data save path
  --samples,     Number of frames you want to extract for each video
  --process,     Number of processes processed, default=1
```

Examples: `python --root_dir video_dir --save_dir frame_dir --samples 128 --process 1`

### Exrtact faces

We use retinaface to crop the face region and also use multi threads with maximum numbers of gpu. 

```python
python preprocess/extract_faces.py --root_dir ROOT_PATH --save_dir SAVE_DIR

optional arguments:
  -h, --help,    Show this help message and exit
  --root_dir,    Frame data root path
  --save_dir,    Face data save path
  --process,     Number of processes processed, default=1
```
