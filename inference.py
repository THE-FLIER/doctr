import collections
import os
from PIL import Image
import cv2
import numpy as np
import torch

from torchvision import transforms
import torch.nn.functional as F

import cv2
from GeoTr import GeoTr
import hdf5storage as h5
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
def reload_model(model, path):
    if not bool(path):
        return model
    else:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        return model

def inference(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_list = os.listdir(opt.distorrted_path)
    if not os.path.exists(opt.gsave_path):
        os.mkdir(opt.gsave_path)
    model = GeoTr().to(device)
    model = reload_model(model, opt.GeoTr_path)
    model.eval()
    for img_path in img_list:
        name = img_path.split(".")[-2]
        img_path = opt.distorrted_path + img_path

        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.0
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.Tensor(im).unsqueeze(0).float().to(device)
        with torch.no_grad():
            bm = model(im)
            # h_, w_ = (h, w)  # 目标高宽
            # w_scale = w_ / 288
            # h_scale = h_ / 288
            # bm_ = F.interpolate(bm, scale_factor=(h_scale, w_scale), mode='bilinear', align_corners=False)
            # bm_ = bm_.permute(0, 2, 3, 1)
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            bm_ = np.stack([bm0, bm1], axis=2)
            # bm_ = np.reshape(bm, (1, h, w, 2))
            bm_ = torch.Tensor(bm_).float().unsqueeze(0)

            img_ = torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float()
            # img_ = np.expand_dims(img_, 0)

            uw = F.grid_sample(img_, bm_, align_corners=True)
            img_geo = ((uw[0] * 255).permute(1, 2, 0).numpy())[:, :, ::-1].astype(np.uint8)
            # uw_ = uw.cpu().detach().numpy()
            # uw_ = np.array(uw_[0]).transpose((1, 2, 0))
            # gt_uw = cv2.cvtColor(uw_, cv2.COLOR_RGB2BGR)
            # gt_uw = cv2.normalize(
            #     gt_uw,
            #     dst=None,
            #     alpha=0,
            #     beta=255,
            #     norm_type=cv2.NORM_MINMAX,
            #     dtype=cv2.CV_8U,
            # )
            cv2.imwrite(f"{opt.gsave_path}{name}.png", img_geo)
        print("Done: ", name+'.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distorrted_path", default="dataset/train/crop/")
    parser.add_argument("--gsave_path", default="./inference_results/1/")
    parser.add_argument("--GeoTr_path", default="./outputs/national-1/best_model.pth")
    opt = parser.parse_args()
    inference(opt)

if __name__=="__main__":
    main()
