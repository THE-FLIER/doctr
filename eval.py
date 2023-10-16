import cv2
import numpy as np
from skimage.metrics import structural_similarity
from test_pre import mask_get
def eval_unwarp(pred, gt, gt_mask):

    pred = cv2.GaussianBlur(pred, (7, 7), 1)
    gt = cv2.GaussianBlur(gt, (7, 7), 1)
    gt_mask = cv2.GaussianBlur(gt_mask, (7, 7), 1)

    pred = cv2.resize(pred, (0, 0), fx=0.5, fy=0.5)
    gt = cv2.resize(gt, (0, 0), fx=0.5, fy=0.5)
    gt_mask = cv2.resize(gt_mask, (0, 0), fx=0.5, fy=0.5)

    pred = pred.astype(np.float32) / 255.
    gt = gt.astype(np.float32) / 255.
    gt_mask = gt_mask.astype(np.float32) / 255.

    pred =cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gt =cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

    # 计算光流场
    flow = cv2.calcOpticalFlowFarneback(gt, pred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # 掩码拼接
    mask = (gt_mask == 0)
    ld = np.mean(magnitude[~mask])

    # MSSIM
    levels = 5
    mssim = 0
    for _ in range(levels):
        ssim = structural_similarity(pred, gt_mask, data_range=1)
        mssim += (0.0448 * (0.1333) ** levels) * ssim
        pred = cv2.pyrDown(pred)
        gt_mask = cv2.pyrDown(gt_mask)
    print(mssim, ld)
    return mssim, ld

a,b,c=mask_get(img1="dataset/INV3D/warped_albedo.png",img2="dataset/INV3D/warped_albedo.png")
eval_unwarp(a,b,c)