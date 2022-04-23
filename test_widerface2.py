from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

# 需要注意修改的地方：
# 1. 是否保存图片； 2.使用那个测试网络，在下面参数中修改，不是在前面的参数中； 3.将测试文件保存在那个文件夹中

parser = argparse.ArgumentParser(description='Retinaface')
# parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('-m', '--trained_model', default='./weights/model_250_params.pth',
#                     type=str, help='Trained state_dict file path to open')

parser.add_argument('-m', '--trained_model', default='/home/kpl/RetinaFace-pytorch/models/M-0513/model_250_params.pth',
                    type=str, help='Trained state_dict file path to open')

# parser.add_argument('-m', '--trained_model', default='/home/kpl/RetinaFace-pytorch/models/M-0520/model_250_params.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
#                     type=str, help='Trained state_dict file path to open')

# parser.add_argument('-m', '--trained_model', default='./weights/new/model_250_params.pth',
#                     type=str, help='Trained state_dict file path to open')



# parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
# parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/test_wider', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/test_wider/txt2_wider', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/Pytorch_Retinaface-master/txt_result0519/txt2_wider', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0519- 3', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0519-4', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0520-1', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0520-4', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0528-1', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0528-2', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0530-1', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0603-1', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0603-2', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0605-2', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0605-3', type=str, help='Dir to save txt results') # 0606
# parser.add_argument('--save_folder', default='/home/kpl/result/0605-3-1', type=str, help='Dir to save txt results') # 0606 no pretrain
# parser.add_argument('--save_folder', default='/home/kpl/result/0605', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0605-1', type=str, help='Dir to save txt results')
# parser.add_argument('--save_folder', default='/home/kpl/result/0605-2-1', type=str, help='Dir to save txt results') # 有pretrain。
# parser.add_argument('--save_folder', default='/home/kpl/result/0605-2-2', type=str, help='Dir to save txt results')  # no pratrain
# parser.add_argument('--save_folder', default='/home/kpl/result/0608', type=str, help='Dir to save txt results')  # pretrain，学习率没改
# parser.add_argument('--save_folder', default='/home/kpl/result/0608-1', type=str, help='Dir to save txt results')  # pretrain，更改之后

# parser.add_argument('--save_folder', default='/home/kpl/result/0614', type=str, help='Dir to save txt results')  # 衰减率设置为i0.001

# parser.add_argument('--save_folder', default='/home/kpl/result/0617', type=str, help='Dir to save txt results')  # 测试训练集的精确度看是否过拟合，对比文件0614
parser.add_argument('--save_folder', default='./image_test/0306', type=str, help='Dir to save txt results')  # 测试训练集的精确度看是否过拟合，对比文件0614






# parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
# parser.add_argument('--dataset_folder', default='./data/widerface/train/images/', type=str, help='dataset path')

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold') # 不是0.5吗
# parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')


parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')  # if False.即保存图片
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results') # true就是不保存图片
parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
#     net = load_model(net, args.trained_model, args.cpu)
#     path='./weights/new/model_250_params.pth'
#     path='/home/kpl/RetinaFace-pytorch/models/M-0513/model_250_params.pth'
#     path='/home/kpl/RetinaFace-pytorch/models/M-0520/model_250_params.pth'
#     path='/home/kpl/RetinaFace-pytorch/models/M-0528/model_250_params.pth'  # 使用官网数据集
#     path='/home/kpl/RetinaFace-pytorch/models/M-0529/model_250_params.pth' # 修改anchor尺寸之后，将16修改为26。32修改为42。（应该没关系？因为训练网络的过程就是在调整anchor的过程，如果差距大，就往大的方向调，如果差距小，就调整的幅度小，应该是没关系的。）
    
#     path='/home/kpl/RetinaFace-pytorch/models/M-0602/model_250_params.pth'
#     path='/home/kpl/RetinaFace-pytorch/models/M-0603/model_250_params.pth'  # 0605-1  #0606:result 0603-2. no pretrian when test.
#     path='/home/kpl/Model/0605-1/model_250_params.pth' 
#     path='/home/kpl/RetinaFace-pytorch/models/M-0605/model_250_params.pth'  # have rorate,no initial
#     path='/home/kpl/RetinaFace-pytorch/models/M-0605-3/model_250_params.pth'
#     path='/home/kpl/RetinaFace-pytorch/models/M-0605-2/model_250_params.pth'  # 没有初始化，添加了一个140.目前，M-0605-3模型最好。
#     path='/home/kpl/RetinaFace-pytorch/models/M-0608/model_235_params.pth'  # 没有初始化，官网数据集
#     path='/home/kpl/RetinaFace-pytorch/models/M-0608/model_250_params.pth'
#     path='/home/kpl/RetinaFace-pytorch/models/M-0614/model_250_params.pth'
#     path = './weights/mobilenet0.25_Final.pth'
    path = './model/0305mobilenet0.25_Final.pth'
    
    net_paras=net.load_state_dict(torch.load(path))
    print(net_paras.state_dict())
    # print(net_paras.keys())
    
#     net.load_state_dict(torch.load(path),strict=False)
# #     net.load_state_dict(torch.load(path)['model'],strict=False)  # 咱们自己的这个是需要加上参数model的
#     net.eval()
#     print('Finished loading model!')
#     print(net)
#     cudnn.benchmark = True
#     device = torch.device("cpu" if args.cpu else "cuda")
#     net = net.to(device)
#
#     # testing dataset
#     testset_folder = args.dataset_folder
#     testset_list = args.dataset_folder[:-7] + "wider_val.txt"
# #     testset_list = args.dataset_folder[:-7] + "image_label.txt"
#
#
#     with open(testset_list, 'r') as fr:
#         test_dataset = fr.read().split()
#     num_images = len(test_dataset)
#
#     _t = {'forward_pass': Timer(), 'misc': Timer()}
#
#     # testing begin
# #     stop=0
#     for i, img_name in enumerate(test_dataset):
# #         if stop==4:
# #             break
# #         stop+=1
#         image_path = testset_folder + img_name
#         img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         img = np.float32(img_raw)
#
#         # testing scale
#         target_size = 1600
#         max_size = 2150
#         im_shape = img.shape
#         im_size_min = np.min(im_shape[0:2])
#         im_size_max = np.max(im_shape[0:2])
#         resize = float(target_size) / float(im_size_min)
#         # prevent bigger axis from being more than max_size:
#         if np.round(resize * im_size_max) > max_size:
#             resize = float(max_size) / float(im_size_max)
#         if args.origin_size:
#             resize = 1
#
#         if resize != 1:
#             img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
#         im_height, im_width, _ = img.shape
#         scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
#         img -= (104, 117, 123)
#         img = img.transpose(2, 0, 1)
#         img = torch.from_numpy(img).unsqueeze(0)
#         img = img.to(device)
#         scale = scale.to(device)
#
#         _t['forward_pass'].tic()
#         loc, conf, landms = net(img)  # forward pass
#         _t['forward_pass'].toc()
#         _t['misc'].tic()
# #         print(loc.data)
#         priorbox = PriorBox(cfg, image_size=(im_height, im_width))
#         priors = priorbox.forward()
#         priors = priors.to(device)
#         prior_data = priors.data
# #         print(prior_data)
#         boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#         boxes = boxes * scale / resize
# #         print(boxes)
#         boxes = boxes.cpu().numpy()
#         scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
# #         print(scores)
#         landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#         scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                                img.shape[3], img.shape[2]])
#         scale1 = scale1.to(device)
#         landms = landms * scale1 / resize
#
#         landms = landms.cpu().numpy()
#
#         # ignore low scores
#         inds = np.where(scores > args.confidence_threshold)[0]
#         boxes = boxes[inds]
#         landms = landms[inds]
#         scores = scores[inds]
# #         print(scores)
#
#         # keep top-K before NMS
#         order = scores.argsort()[::-1]
#         # order = scores.argsort()[::-1][:args.top_k]
# #         print(order)
#         boxes = boxes[order]
#         landms = landms[order]
#         scores = scores[order]
#
#         # do NMS
#         dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#         keep = py_cpu_nms(dets, args.nms_threshold)
# #         print(keep)
#         # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
#         dets = dets[keep, :]
#         landms = landms[keep]
#
#         # keep top-K faster NMS
#         # dets = dets[:args.keep_top_k, :]
#         # landms = landms[:args.keep_top_k, :]
#
#         dets = np.concatenate((dets, landms), axis=1)
#         _t['misc'].toc()
#
#         # --------------------------------------------------------------------
#         save_name = args.save_folder + img_name[:-4] + ".txt"
#         dirname = os.path.dirname(save_name)
#         if not os.path.isdir(dirname):
#             os.makedirs(dirname)
#         with open(save_name, "w") as fd:
#             bboxs = dets
#             file_name = os.path.basename(save_name)[:-4] + "\n"
#             bboxs_num = str(len(bboxs)) + "\n"
#             fd.write(file_name)
#             fd.write(bboxs_num)
#             for box in bboxs:
#                 x = int(box[0])
#                 y = int(box[1])
#                 w = int(box[2]) - int(box[0])
#                 h = int(box[3]) - int(box[1])
#                 confidence = str(box[4])
#                 line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
#                 fd.write(line)
#
#         print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
# #         print(args.save_image)
#
#         # save image
#         if not args.save_image:
#             for b in dets:
#                 if b[4] < args.vis_thres:
#                     continue
#                 text = "{:.4f}".format(b[4])
#                 b = list(map(int, b))
#                 cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
#                 cx = b[0]
#                 cy = b[1] + 12
#                 cv2.putText(img_raw, text, (cx, cy),
#                             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
#
#                 # landms
#                 cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
#                 cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
#                 cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
#                 cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
#                 cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
#             # save image
# #             if not os.path.exists("./results/"):
# #                 os.makedirs("./results/")
# #             name = "./results/" + str(i) + ".jpg"
# #             cv2.imwrite(name, img_raw)
#
#             if not os.path.exists("/home/kpl/result/ImageResult/0519-3"):
#                 os.makedirs("/home/kpl/result/ImageResult/0519-3")
#             name = "/home/kpl/result/ImageResult/0519-3" + str(i) + ".jpg"
#             cv2.imwrite(name, img_raw)
#

