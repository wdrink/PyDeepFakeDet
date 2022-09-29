import json

type = 'val'

with open(type + '.json', "r") as f:
    row_data = json.load(f)
# 读取每一条json数据
a = 0
with open('/mnt/data/liyihui/face_data/DFDet/origin/videos/video_list_' + type + '.txt', 'w') as ff:
    for d in row_data:
        a = a + 1
        ff.write(d[0] + '.mp4\n')
        ff.write(d[1] + '.mp4\n')

print(a)
