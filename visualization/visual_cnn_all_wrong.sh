#!bin/bash
cat /mnt/data/liyihui/face_data/DFDet/ana/ana_list/cnn_all_wrong_test_list.txt | while read line;
do
  path=($line)
#  OLD_IFS="$IFS"
#  IFS="/"
#  array=($path)
#  IFS="$OLD_IFS"
  name=${path////_}
  CUDA_VISIBLE_DEVICES=4 python gradcam.py --model Xception --pth /mnt/data/liyihui/face_data/DFDet/model/Xception_FFDF_c23.pyth --layer block12 --img $path --save_path /mnt/data/liyihui/face_data/DFDet/ana/ana_visual/xception/block12$name
done

CUDA_VISIBLE_DEVICES=4 python gradcam.py --model VisionTransformer --pth /mnt/data/liyihui/face_data/DFDet/model/VisionTransformer_FFDF_c23.pyth --layer norm --img /mnt/data/liyihui/face_data/DFDet/ana/ana_list/cnn_all_wrong_test_list.txt --save_path /mnt/data/liyihui/face_data/DFDet/ana/ana_visual/vit/