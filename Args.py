# -*- coding:utf-8 -*-
# @Author  :   Multimodal Perception and Intelligent Analysis
# @Contact :   rzw@njust.edu.cn
class Args():
	# For training
	path_ir = ['datasets/LLVIP/lwir']
	cuda = True
	lr = 0.01
	epochs = 1
	batch = 24
	train_num = 500
	step = 10
	channel = 1
	Height = 256
	Width = 256

	resume_model_trans = None
	save_fusion_model = "./models"
	save_loss_dir = "./models/loss"