import os
import torch
import torch.nn.functional as F
import sys
import eval.metrics as Measure

sys.path.append('./models')
import numpy as np
from datetime import datetime

from lib.AM_EAM import Network


from torchvision.utils import make_grid
from data_cod import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options_cod import opt


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

cudnn.benchmark = True
save_path = opt.save_path

model = Network().cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')


train_loader = get_loader(image_root=opt.rgb_root + 'Imgs/',
                              gt_root=opt.rgb_root + 'GT/',
                              edge_root=opt.rgb_root +'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)


val_loader = test_dataset(image_root=opt.val_rgb_root + 'Imgs/',
                              gt_root=opt.val_rgb_root + 'GT/',
                              testsize=opt.trainsize)

total_step = len(train_loader)



#set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.cuda()
    model.train()

    sal_loss_all = 0#总损失
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()



            s0, s1, s2, s3,s4, edge = model(images)





            # ---loss function-----------
            loss4 = structure_loss(s4, gts)
            loss3 = structure_loss(s3, gts)
            loss2 = structure_loss(s2, gts)
            loss1 = structure_loss(s1, gts)
            loss0 = structure_loss(s0, gts)
            losse = dice_loss(edge, edges)



            loss = loss4 +loss3 + loss2 + loss1 + loss0 + 3 * losse




            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)#内存使用量
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, memory_used))

                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = s1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'newNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'newNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path):
    """
    validation function
    """

    global best_score, best_epoch, best_metric_dict
    FM = Measure.Fmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():#评估时不需要梯度计算
        for i in range(test_loader.size):
            image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            #将预测结果和真实标签作为输入，逐步计算这些指标
            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)

        #从评估指标对象中获取结果，并更新 metrics_dict 字典
        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
        metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))

        cur_score = metrics_dict['Sm'] + metrics_dict['mxFm'] + metrics_dict['mxEm']

        if epoch == 1:
            best_score = cur_score
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print(
                '[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                    epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                    best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))
            logging.info(
                '[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch:{}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                    epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                    best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))

if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        val(val_loader, model, epoch, save_path)

