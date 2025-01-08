# -*- coding:utf-8 -*-
# @Author  :   Multimodal Perception and Intelligent Analysis
# @Contact :   rzw@njust.edu.cn
# Train fusion models
import os
import torch
import time
from tools import utils
import random
from torch.optim import Adam
from torch.autograd import Variable
from network.net_AMLCA import FuseNet
from network.loss import Gradient_loss, Order_loss, Patch_loss
from Args import Args as args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -------------------------------------------------------
custom_config = {
    "en_out_channels1": 32,
    "out_channels": 1,
    "part_out": 64,
    "train_flag": True,
    "img_size": 32,
    "patch_size": 2,
    "depth_self": 1,
    "depth_cross": 1,
    "n_heads": 16,
    "qkv_bias": True,
    "mlp_ratio": 4,
    "p": 0.,
    "attn_p": 0.,
}
# -------------------------------------------------------
def load_data(path, train_num):
    imgs_path, _ = utils.list_images_datasets(path, train_num)
    random.shuffle(imgs_path)
    return imgs_path

def train(data, img_flag):
    batch_size = args.batch
    step = args.step
    model = FuseNet(**custom_config)
    shift_flag = False
    if args.resume_model_trans is not None:
        print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_trans))
        model.load_state_dict(torch.load(args.resume_model_trans))
    trainable_params = [{'params': filter(lambda x: x.requires_grad, model.parameters()), 'lr': args.lr}]
    optimizer = Adam(trainable_params, args.lr, weight_decay=0.9)
    gra_loss = Gradient_loss(custom_config['out_channels'])
    order_loss = Order_loss(custom_config['out_channels'])
    if args.cuda:
        model.cuda()
        gra_loss.cuda()
        order_loss.cuda()

    print('Start training.....')
    temp_path_model1 = os.path.join(args.save_fusion_model)
    if os.path.exists(temp_path_model1) is False:
        os.mkdir(temp_path_model1)
    temp_path_model = os.path.join(temp_path_model1)
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    temp_path_loss = os.path.join(temp_path_model, 'loss')
    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)
    loss_p4 = 0.
    loss_p8 = 0.
    loss_p9 = 0.
    loss_all = 0.

    loss_mat = []
    model.train()
    count = 0
    lr_cur = args.lr

    for e in range(args.epochs):
        lr_cur = utils.adjust_learning_rate(optimizer, e, lr_cur)
        img_paths, batch_num = utils.load_dataset(data, batch_size)

        for idx in range(batch_num):

            image_paths_ir = img_paths[idx * batch_size:(idx * batch_size + batch_size)]
            img_ir = utils.get_train_images(image_paths_ir, height=args.Height, width=args.Width, flag=img_flag)

            image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
            img_vi = utils.get_train_images(image_paths_vi, height=args.Height, width=args.Width, flag=img_flag)

            count += 1
            optimizer.zero_grad()
            batch_ir = Variable(img_ir, requires_grad=False)
            batch_vi = Variable(img_vi, requires_grad=False)

            if args.cuda:
                batch_ir = batch_ir.cuda()
                batch_vi = batch_vi.cuda()

            outputs = model.train_module(batch_ir, batch_vi, shift_flag, gra_loss,
                                         order_loss)

            total_loss = outputs['total_loss']
            loss_mat.append(total_loss.item())
            total_loss.backward()
            optimizer.step()
            loss_p4 += outputs['pix_loss']
            loss_p8 += outputs['vgg_loss']
            loss_p9 += outputs['gra_loss']
            loss_all += total_loss

            if count % step == 0:
                loss_p4 /= step
                loss_p8 /= step
                loss_p9 /= step
                loss_all /= step

                mesg = "{} - Epoch {}/{} - Batch {}/{} - lr:{:.6f} - pix loss: {:.6f} - vgg loss: {:.6f} - gra loss: {:.6f} - total loss: {:.6f} \n". \
                    format(time.ctime(), e + 1, args.epochs, idx + 1, batch_num, lr_cur,
                           loss_p4, loss_p8, loss_p9, loss_all)

                print(mesg)
                loss_p4 = 0.
                loss_p8 = 0.
                loss_p9 = 0.
                loss_all = 0.
        if (e + 1) % 1 == 0:
            model.eval()
            model.cpu()
            save_model_filename = "epoch_" + str(e + 1) + ".model"
            save_model_path = os.path.join(temp_path_model, save_model_filename)
            torch.save(model.state_dict(), save_model_path)
            print("\nCheckpoint, trained model saved at: " + save_model_path)
        model.train()
        model.cuda()
    print("\nDone, TransFuse training phase.")


if __name__ == "__main__":
    # True - RGB, False - gray
    if args.channel == 1:
        img_flag = False
    else:
        img_flag = True
    path = args.path_ir
    train_num = args.train_num
    data = load_data(path, train_num)
    train(data, img_flag)
