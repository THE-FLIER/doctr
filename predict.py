import argparse

import cv2
import torch
import torch.nn.functional as F
import os
from GeoTr import GeoTr

def reload_model(model, path):
    if not bool(path):
        return model
    else:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        return model

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = args.image
    model_path = args.model
    output_path = args.output
    img_list = os.listdir(image_path)

    model = GeoTr().to(device)
    model = reload_model(model, model_path)
    model.eval()
    for img_path in img_list:
        name = img_path.split(".")[-2]
        img_path = image_path + img_path
        img = cv2.imread(img_path)
        img = cv2.resize(img, (288, 288))
        img = img[:, :, ::-1]
        img = img.astype("float32") / 255.0
        img = img.transpose(2, 0, 1)

        x = torch.from_numpy(img).unsqueeze(0).float().to(device)

        with torch.no_grad():
            bm = model(x)
            bm = bm.permute(0, 2, 3, 1)
            # bm = bm.transpose([0, 2, 1, 3]).transpose([0, 1, 3, 2])
            out = F.grid_sample(x, bm)

            out = out.cpu().detach()
            out = out[0].numpy().transpose((1, 2, 0))
            out = out[:, :, ::-1]
            out = out * 255.0
            out = out.astype("uint8")

            cv2.imwrite(f"{output_path}{name}.png", out)
        print("Done: ", name + '.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")

    parser.add_argument(
        "--image",
        "-i",
        nargs="?",
        type=str,
        default="dataset/train/crop/",
        help="The path of image",
    )

    parser.add_argument(
        "--model",
        "-m",
        nargs="?",
        type=str,
        default="./outputs/national-1/best_model.pth",
        help="The path of model",
    )

    parser.add_argument(
        "--output",
        "-o",
        nargs="?",
        type=str,
        default="./inference_results/1/",
        help="The path of output",
    )

    args = parser.parse_args()

    print(args)

    run(args)
