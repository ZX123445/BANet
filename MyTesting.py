import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, cv2

from lib.AM_EAM import Network

# from lib.backbone import Network
# from lib.backbone_HFAM import Network
# from lib.backbone_HFAM_HBFM import Network
# from lib.backbone_HFAM_HBFM_BEF import Network

from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='/hy-tmp/TPRNet-main/snapshot/SINet_V2/Net_epoch_best.pth')
opt = parser.parse_args()
dataset_path = opt.test_path

model = Network()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

test_datasets = ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']
for dataset in test_datasets:
    save_path = './test_map/' + dataset + '/'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'edge/', exist_ok=True)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()


        _, _, _, _, res, e = model(image)



        # GT预测
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)

        # 边缘预测
        e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
        e = e.data.cpu().numpy().squeeze()
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        cv2.imwrite(save_path + 'edge/' + name, (e * 255).astype(np.uint8))

