import os
import torch
from torch.autograd import Variable
from network.net_AMLCA import FuseNet
from tools import utils
from Args import Args as args

k = 10
def load_model(custom_config_trans,  model_path_trans):
    model_trans = FuseNet(**custom_config_trans)
    model_trans.load_state_dict(torch.load(model_path_trans))
    model_trans.cuda()
    model_trans.eval()
    return model_trans


def test( model_trans, shift_flag, ir_path, vi_path, ir_name, output_path):
    ir_img = utils.get_train_images(ir_path, None, None, flag=False)
    vi_img, vi_cb, vi_cr = utils.get_test_images_color(vi_path, None, None, flag=img_flag)
    ir_img = Variable(ir_img, requires_grad=False)
    vi_img = Variable(vi_img, requires_grad=False)
    if args.cuda:
        ir_img = ir_img.cuda()
        vi_img = vi_img.cuda()

    outputs = model_trans(ir_img, vi_img, shift_flag)
    img_out = outputs['out']
    # ---------------------------------------------
    path_out = output_path + '/'
    utils.save_image_color(img_out, vi_cb, vi_cr, path_out + ir_name)
    print('Done. ', ir_name)


if __name__ == "__main__":
    custom_config_trans = {
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
        "p": 0.1,
        "attn_p": 0.,
    }
    model_path_trans = "./models/AMLCA.model"
    # ----------------------------------------------------
    img_flag = True
    data_type = ['M3FD', 'TNO','LLVIP']
    data_type = data_type[0]
    test_path_ir = './images/' + data_type + '/ir'
    test_path_vi = './images/' + data_type + '/vi'

    ir_pathes, ir_names = utils.list_images_test(test_path_ir)
    # ---------------------------------------------------
    output_path1 = './output/'
    if os.path.exists(output_path1) is False:
        os.mkdir(output_path1)
    output_path = output_path1 + data_type
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    # ---------------------------------------------------
    count = 0
    shift_flag = True
    with torch.no_grad():
        model_trans = load_model(custom_config_trans, model_path_trans)
        for ir_name in ir_names:
            vi_name = ir_name.replace('IR', 'VIS')
            ir_path = os.path.join(test_path_ir, ir_name)
            vi_path = os.path.join(test_path_vi, vi_name)
            # ---------------------------------------------------
            test( model_trans, shift_flag, ir_path, vi_path, ir_name, output_path)




