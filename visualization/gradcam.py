import sys

sys.path.append('/mnt/data/liyihui/job/PyDeepFakeDet')
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet18
from PyDeepFakeDet import models
from PyDeepFakeDet.utils.checkpoint import load_checkpoint
import argparse


class GradCAM:
    def __init__(
            self,
            model: nn.Module,
            target_layer: str,
            size=(224, 224),
            num_cls=1000,
            mean=None,
            std=None,
    ) -> None:
        self.model = model
        self.model.eval()

        # blocks = getattr(self.model, target_layer)
        # block11 = getattr(blocks, '11')
        # att = getattr(block11,'attn')
        # qkv = getattr(att,'qkv')
        # qkv.register_forward_hook(self.__forward_hook)
        # qkv.register_backward_hook(self.__backward_hook)

        getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.model, target_layer).register_forward_hook(self.__backward_hook)

        self.size = size
        self.origin_size = None
        self.num_cls = num_cls

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if mean and std:
            self.mean, self.std = mean, std

        self.grads = []
        self.fmaps = []

    def forward(self, img_arr: np.ndarray, label=None, show=True, write=False, path=''):
        img_input = self.__img_preprocess(img_arr.copy())

        output = self.model({"img": img_input})
        output = output["logits"]
        idx = np.argmax(output.cpu().data.numpy())

        self.model.zero_grad()
        loss = self.__compute_loss(output, label)

        loss.backward()

        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        cam_show = cv2.resize(cam, self.origin_size)
        img_show = img_arr.astype(np.float32) / 255
        self.__show_cam_on_image(img_show, cam_show, if_show=show, if_write=write, path=path)

        self.fmaps.clear()
        self.grads.clear()

    def __img_transform(
            self, img_arr: np.ndarray, transform: torchvision.transforms
    ) -> torch.Tensor:
        img = img_arr.copy()
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)
        return img

    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        self.origin_size = (img_in.shape[1], img_in.shape[0])
        img = img_in.copy()
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)

    def __compute_loss(self, logit, index=None):
        if not index:
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(index)

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros((1, self.num_cls)).scatter_(1, index, 1)
        one_hot.requires_grad = True
        loss = torch.sum(one_hot * logit)
        return loss

    def __compute_cam(self, feature_map, grads):
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2))
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.size)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam

    def __show_cam_on_image(
            self, img: np.ndarray, mask: np.ndarray, if_show=True, if_write=False, path=""
    ):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        if if_write:
            cv2.imwrite(path, cam)
        if if_show:
            plt.imshow(cam[:, :, ::-1])
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('gradcam script', add_help=False)
    parser.add_argument('--model', default='Xception', type=str)
    parser.add_argument('--pth', default='/mnt/data/liyihui/face_data/DFDet/model/Xception_FFDF_c23.pyth', type=str)
    parser.add_argument('--layer', default='block12', type=str)
    parser.add_argument('--img', default='/mnt/data/liyihui/face_data/DFDet/ana/ana_list/cnn_all_wrong_test_list.txt',
                        type=str)
    parser.add_argument('--save_path',
                        default='/mnt/data/liyihui/face_data/DFDet/ana/ana_visual/xception/', type=str)

    # parser = argparse.ArgumentParser('gradcam script', add_help=False)
    # parser.add_argument('--model', default='VisionTransformer', type=str)
    # parser.add_argument('--pth', default='/mnt/data/liyihui/face_data/DFDet/model/VisionTransformer_FFDF_c23.pyth',
    #                     type=str)
    # parser.add_argument('--layer', default='blocks', type=str)
    # parser.add_argument('--img', default='/mnt/data/liyihui/face_data/DFDet/ana/ana_list/cnn_all_wrong_test_list.txt',
    #                     type=str)
    # parser.add_argument('--save_path',
    #                     default='/mnt/data/liyihui/face_data/DFDet/ana/ana_visual/vit/', type=str)

    args = parser.parse_args()
    cfg = {"PRETRAINED": False, "ESCAPE": ""}
    net = getattr(models, args.model)(cfg)
    state_dic = torch.load(args.pth)
    print("keys in your model:", state_dic["model_state"].keys())
    load_checkpoint(args.pth, net, False)

    grad_cam = GradCAM(
        net,
        num_cls=2,
        target_layer=args.layer,
        size=(384, 384),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    paths = open(args.img, 'r').readlines()
    for path in paths:
        path = path.strip('\n').split()[0]
        img = cv2.imread(path, 1)
        save_dir = args.save_path + args.layer + path.replace('/', '_')
        grad_cam.forward(img, show=False, write=True, path=save_dir)
