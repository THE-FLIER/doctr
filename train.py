import math
from skimage.metrics import peak_signal_noise_ratio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import  torch.optim as optim
from GeoTr import GeoTr
from dataloader import Doc3dDataset
import argparse
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
def train(args):
    #定义空间
    experiment_name = os.path.join(args.project, args.name)
    os.makedirs(experiment_name, exist_ok=True)

    #定义日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器，将日志写入文件
    file_handler = logging.FileHandler(f'{experiment_name}/training.log')
    file_handler.setLevel(logging.INFO)
    # 创建一个控制台处理器，将日志打印到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建一个格式化器，定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 将格式化器添加到处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    #可视化初始化
    train_writer = SummaryWriter(f'{experiment_name}/logs/train')
    val_writer = SummaryWriter(f'{experiment_name}/logs/validation')

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据集
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    #训练集
    train_dataset = Doc3dDataset(
        train_data_path,
        img_size=(args.img_rows, args.img_cols),
        split="train",
        is_augment=True
    )
    train_data_size = len(train_dataset)
    logger.info(f"The number of training samples = {train_data_size}")

    #验证集
    val_dataset = Doc3dDataset(
        val_data_path,
        split="val",
        img_size=(args.img_rows, args.img_cols),
        is_augment=False
    )
    val_data_size = len(val_dataset)
    logger.info("The number of validation samples = %d" % val_data_size)

    #数据加载
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, num_workers=1, shuffle=True
    )

    # 定义模型
    model = GeoTr().to(device)

    #训练步数
    toal_steps = math.ceil(len(train_dataloader) * args.epochs)

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                           max_lr=args.lr,
                           epochs=65,
                           steps_per_epoch=len(train_dataloader),
                           pct_start=0.1,
                           )

    epoch_start = 0
    best_fitness = 0
    #加载checkpoints
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(f"Loading model and optimizer from checkpoint '{args.resume}'")

            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            best_fitness = checkpoint["best_fitness"]
            epoch_start = checkpoint["epoch"]

        else:
            logger.warning(f"No checkpoint found at '{args.resume}'")

    #loss
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    global_step = 0

    scaler = amp.GradScaler()
    # 训练参数
    for epoch in range(epoch_start, args.epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                images = images.to(device)
                labels = labels.to(device)

                target = model(images)
                x = target
                target_nhwc = x.permute(0, 2, 3, 1)
                loss = criterion_l1(target_nhwc, labels)
                mse_loss = criterion_mse(target_nhwc, labels)


            #backward
            # loss.backward()
            # optimizer.step()
            # scheduler.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 记录日志
            global_step += 1
            train_writer.add_scalar('loss', loss, global_step)
            train_writer.add_scalar('mse', mse_loss, global_step)
            train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            if (i+1) % 2 ==0:
                logger.info(
                f"epochs {epoch + 1}/{args.epochs}, global_step {global_step}/{toal_steps}, l1_loss: {loss:.6f},mse_loss: {mse_loss:.6f}, lr: {optimizer.param_groups[0]['lr']:.6f}")


        #20步验证一次
            if (i+1) % 20 == 0:
                #验证
                model.eval()
                with torch.no_grad():
                    fitness = torch.zeros([]).to(device)
                    avg_mssim = torch.zeros([]).to(device)
                    avg_val_loss = 0.0
                    avg_mse_loss = 0.0
                    for a, (images_val, labels_val) in enumerate(val_dataloader):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        target = model(images_val)
                        pred = target
                        pred_nhwc = pred.permute(0, 2, 3, 1)

                        # predict image
                        out = F.grid_sample(images_val, pred_nhwc)
                        out_gt = F.grid_sample(images_val, labels_val)

                        # calculate ms_ssim
                        ssim_val = ssim(out, out_gt, data_range=1.0)
                        ms_ssim_val = ms_ssim(out, out_gt, data_range=1.0,
                                              weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

                        # loss
                        val_loss = criterion_l1(pred_nhwc, labels_val)
                        val_mse_loss = criterion_mse(pred_nhwc, labels_val)

                        # fitness
                        fitness += ssim_val
                        avg_mssim += ms_ssim_val
                        # loss
                        avg_val_loss += val_loss
                        avg_mse_loss += val_mse_loss

                        if a == len(val_dataloader) - 1:
                            # img
                            out_ = out.cpu().detach()
                            out_ = out_[0].numpy().transpose((1, 2, 0))
                            out_ = out_[:, :, ::-1]
                            out_ = out_ * 255.0
                            out_ = out_.astype("uint8")

                            out_gt_ = out_gt.cpu().detach()
                            out_gt_ = out_gt_[0].numpy().transpose((1, 2, 0))
                            out_gt_ = out_gt_[:, :, ::-1]
                            out_gt_ = out_gt_ * 255.0
                            out_gt_ = out_gt_.astype("uint8")

                            val_writer.add_image('image-predict', out_, global_step, dataformats='HWC')
                            val_writer.add_image('image-gt', out_gt_, global_step, dataformats='HWC')

                    fitness /= len(val_dataloader)
                    avg_mssim /= len(val_dataloader)
                    avg_val_loss /= len(val_dataloader)
                    avg_mse_loss /= len(val_dataloader)

                    logger.info(f"[VAL MODE] Epoch: {epoch+1}, VAL Iter: {i+1}, "
                                f"l1_loss: {float(avg_val_loss)}, mse_loss: {float(avg_mse_loss)}, "
                                f"ms-ssim: {float(avg_mssim)}, ssim: {float(fitness)}")

                    val_writer.add_scalar('loss', avg_val_loss, global_step)
                    val_writer.add_scalar('mse', avg_mse_loss, global_step)
                    val_writer.add_scalar('metric-ms_ssim', avg_mssim, global_step)
                    val_writer.add_scalar('metric-ssim', fitness, global_step)


                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_fitness": best_fitness,
                }

                model.train()
                #
                if best_fitness < fitness:
                    best_fitness = fitness
                    base_name = "best_model.pth"
                    full_name = os.path.join(experiment_name, base_name)
                    # 保存为.pth
                    torch.save(state, full_name)

                # 每5epoch保存一次
                if (epoch + 1) % 5 == 0:
                    base_name = f"{epoch + 1}_epoch.pth"
                    full_name = os.path.join(experiment_name, base_name)
                    # 保存为.pth
                    torch.save(state, full_name)

    #可视化
    train_writer.close()
    val_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-params")
    parser.add_argument(
        "--train_data_path", nargs="?", type=str, default="dataset/doc3d/", help="Data path to load data"
    )#--data_path="dataset/doc3d/"
    parser.add_argument(
        "--val_data_path", nargs="?", type=str, default="dataset/doc3d/", help="Data path to load data"
    )  # --data_path="dataset/doc3d/"
    parser.add_argument(
        "--img_rows", nargs="?", type=int, default=288, help="Height of the input image"
    )
    parser.add_argument(
        "--img_cols", nargs="?", type=int, default=288, help="Width of the input image"
    )
    parser.add_argument(
        "--epochs", nargs="?", type=int, default=65, help="# of the epochs"
    )
    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=8, help="Batch Size"
    )
    parser.add_argument(
        "--lr", nargs="?", type=float, default=1e-04, help="Learning Rate"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default="outputs/109_1/best_model.pth",
        help="Path to previous saved model to restart from",
    )
    parser.add_argument(
        "--name", type=str, default="109_1", help="Name of the experiment"
    )
    parser.add_argument(
        "--project", type=str, default="outputs", help="Name of the project"
    )
    args = parser.parse_args()

    train(args)
