import sys
sys.path.append("..")
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
# from model_DBnet.pred_single import *
import torch.nn.functional as F
import AllConfig.GConfig as GConfig
from Tools.ImageIO import img_read, img_tensortocv2
from Tools.Baseimagetool import *
import random
from UAUP.Auxiliary import *
from Tools.Log import logger_config
import datetime
from Tools.EvalTool import DetectionIoUEvaluator,read_txt
from Tools.PIAttack import project_noise
from mmocr.utils.ocr import MMOCR
from PIL import Image
import numpy as np
from torchvision import transforms
import torchattacks

to_tensor = transforms.ToTensor()


class RepeatAdvPatch_Attack():
    def __init__(self,
                 data_root, savedir, log_name,
                 eps=100 / 255, alpha=1 / 255, decay=1.0,
                 T=200, batch_size=8,
                 adv_patch_size=(1, 3, 100, 100), gap=20, lambdaw=1.0,
                 feamapLoss=False):

        # load DBnet
        # self.midLayer = ["neck.lateral_convs.0", "neck.smooth_convs.0",
        #                  "backbone.layer2.convs.2", "backbone.layer4.convs.2","bbox_head.binarize.7"]
        self.midLayer = ["bbox_head.binarize.7"]
        # hyper-parameters
        self.eps, self.alpha, self.decay = eps, alpha, decay

        # train settings
        self.T = T
        self.batch_size = batch_size

        # Loss
        self.feamapLoss = feamapLoss
        self.Mseloss = nn.MSELoss()
        self.lambdaw= lambdaw

        # path process
        self.data_root = data_root
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.train_dataset = init_train_dataset(data_root)
        self.test_dataset = init_test_dataset(data_root)

        # all gap
        self.shufflegap = len(self.train_dataset) // self.batch_size
        self.gap = gap

        # initiation
        self.adv_patch = torch.ones(list(adv_patch_size))
        self.start_epoch = 1
        self.t=1
        # recover adv patch
        recover_adv_path, recover_t = self.recover_adv_patch()
        if recover_adv_path != None:
            self.adv_patch = recover_adv_path
            self.t = recover_t


        self.logger = logger_config(log_filename=log_name)
        self.evaluator=DetectionIoUEvaluator()

    def recover_adv_patch(self):
        temp_save_path = os.path.join(self.savedir, "advtorch")
        if os.path.exists(temp_save_path):
            files = os.listdir(temp_save_path)
            if len(files) == 0:
                return None, None
            files = sorted(files, key=lambda x: int(x.split('.')[0].split("_")[-1]))
            t = int(files[-1].split("_")[-1])
            keyfile = os.path.join(temp_save_path, files[-1])
            return torch.load(keyfile), t
        return None, None

    #self.CallLoss(res, res_du, x_d2 - DU_d2, it_adv_patch, m_d2)
    def CallLoss(self,text_feamap,uau_feamap,diff,adv_patch,mask):
        mmloss = 0
        for f1, f2 in zip(text_feamap[:-1], uau_feamap[:-1]):
            mmloss -= self.Mseloss(f1.mean([2, 3]), f2.mean([2, 3]))
            # mmloss += (1-F.cosine_similarity(f1.mean([2,3]).flatten(), f2.mean([2,3]).flatten(), dim=0))
        mmloss /= torch.norm(diff, p=1)
        log_mmloss=mmloss.detach().cpu().item()
        # log_mmloss=0

        pred=text_feamap[-1]
        target = torch.zeros_like(pred)
        target = target.to(GConfig.DB_device)
        mask=transforms.Resize(target.shape[2:])(mask)
        #dloss = self.Mseloss(mask * pred, target)
        dloss = self.Mseloss(pred, target)
        log_dloss=dloss.detach().cpu().item()
        grad = torch.autograd.grad(-dloss-mmloss*self.lambdaw,
                                   adv_patch, retain_graph=False, create_graph=False, allow_unused=True)[0]
        # grad = torch.autograd.grad(-dloss,
        #                            adv_patch, retain_graph=False, create_graph=False, allow_unused=True)[0]
        return grad.detach().cpu(),log_mmloss,log_dloss

    # 快捷初始化
    def inner_init_adv_patch_image(self, mask, image, hw, device):
        adv_patch = self.adv_patch.clone().detach()
        adv_patch = adv_patch.to(device)
        adv_patch.requires_grad = True
        image = image.to(device)
        adv_image = self.get_merge_image(adv_patch, mask=mask,
                                         image=image, hw=hw, device=device)
        return adv_patch, adv_image

    def train(self):
        momentum = 0

        eps = self.eps
        alpha_beta =  10/255
        gamma = alpha_beta
        amplification = 0.0

        print("start training-====================")
        shuff_ti = 0  # train_dataset_iter
        for t in range(self.t + 1, self.T):
            print("iter: ", t)
            if t % self.shufflegap == 0:
                random.shuffle(self.train_dataset)
                shuff_ti = 0
            batch_dataset = self.train_dataset[shuff_ti * self.batch_size: (shuff_ti + 1) * self.batch_size]
            shuff_ti += 1

            batch_mmLoss = 0
            batch_dLoss = 0
            sum_grad = torch.zeros_like(self.adv_patch)
            for [x, m_gt, gtpath] in batch_dataset:
                it_adv_patch=self.adv_patch.clone().detach().to(GConfig.DB_device)
                it_adv_patch.requires_grad=True
                x = x.to(GConfig.DB_device)
                m_gt = m_gt.to(GConfig.DB_device)
                x_d1, m_gt_d1 = Diverse_module_1(x, m_gt, t, self.gap)
                m = extract_background(x_d1)  # character region
                h, w = x_d1.shape[2:]
                DU = repeat_4D(patch=it_adv_patch, h_real=h, w_real=w)
                merge_x = DU * m + x_d1 * (1 - m)
                x_d2, m_d2, DU_d2 = Diverse_module_2(image=merge_x, mask=m_gt_d1, UAU=DU,
                                                             now_ti=t, gap=self.gap)

                boxes, res = MMOCR(det='DB_r18', recog=None).single_grad_inference(x_d2, feaName=self.midLayer)
                _, res_du = MMOCR(det='DB_r18', recog=None).single_grad_inference(DU_d2,self.midLayer)

                res = [v.fea.clone() for v in res]
                res_du = [v.fea.clone() for v in res_du]

                grad, temp_mmloss, temp_dloss = self.CallLoss(res, res_du, x_d2 - DU_d2, it_adv_patch, m_d2)
                sum_grad += grad
                batch_dLoss += temp_dloss
                batch_mmLoss += temp_mmloss
                torch.cuda.empty_cache()

                # torch.cuda.empty_cache()

            # update grad
            # MIM 这个不删吗？
            grad = sum_grad / torch.mean(torch.abs(sum_grad), dim=(1), keepdim=True)  # 有待考证
            grad = grad + momentum * self.decay
            momentum = grad

            #PI
            # amplification += alpha_beta * torch.sign(grad)
            # cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
            # projection = gamma * torch.sign(project_noise(cut_noise)).cpu()
            # amplification += projection


            # update self.adv_patch
            temp_patch = self.adv_patch.clone().detach().cpu() + self.alpha * grad.sign() #+ projection.clone().detach().cpu()
            temp_patch = torch.clamp(temp_patch, min=1-self.eps, max=1)
            self.adv_patch = temp_patch

            # update logger
            e = "iter:{}, batch_loss==mmLoss:{},dloss:{}===".format(t, batch_mmLoss / self.batch_size,
                                                                    batch_dLoss / self.batch_size)
            self.logger.info(e)

            # save adv_patch with
            temp_save_path = os.path.join(self.savedir, "advpatch")
            if os.path.exists(temp_save_path) == False:
                os.makedirs(temp_save_path)
            save_adv_patch_img(self.adv_patch, os.path.join(temp_save_path, "advpatch_{}.png".format(t)))
            temp_torch_save_path = os.path.join(self.savedir, "advtorch")
            if os.path.exists(temp_torch_save_path) == False:
                os.makedirs(temp_torch_save_path)
            torch.save(self.adv_patch, os.path.join(temp_torch_save_path, "advpatch_{}".format(t)))

            #保存epoch结果
            if t != 0 and t % 20 == 0:
                self.evauate_test_path(t)

    def evaluate_and_draw(self, adv_patch,image_root,gt_root,save_path,resize_ratio=0,is_resize=False):
        image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
        images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
        test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]
        results=[]#PRF
        for img,name,gt in zip(images,image_names,test_gts):
            h,w=img.shape[2:]
            UAU=repeat_4D(adv_patch.clone().detach(),h,w)
            mask_t=extract_background(img)
            merge_image=img*(1-mask_t)+mask_t*UAU
            merge_image=merge_image.to(GConfig.DB_device)
            if is_resize:
                merge_image=random_image_resize(merge_image,low=resize_ratio,high=resize_ratio)
            boxes= MMOCR(det='DB_r18', recog=None).single_grad_inference(merge_image, feaName=[])
            pred=get_pred_from_boxes(boxes)
            gt=read_txt(gt)
            results.append(self.evaluator.evaluate_image(gt,pred))
            #draw
            #cv2_img=cv2.imread(name)
            temp_save_path=os.path.join(save_path,name.split('\\')[-1])
            #Draw_box(cv2_img,results,save_path)
            Draw_box(img_tensortocv2(merge_image), pred, temp_save_path)
        P, R, F = self.evaluator.combine_results(results)
        return P,R,F

    def evauate_test_path(self, t):

        #=================original=====================
        #savedir
            #t
                #original
                #60
                #....
                #200
        o_img_root=os.path.join(self.data_root,'test')
        o_gt_root = os.path.join(self.data_root, 'test_gt')
        o_save_dir = os.path.join(self.savedir, str(t),'original')
        if os.path.exists(o_save_dir) == False:
            os.makedirs(o_save_dir)
        P,R,F=self.evaluate_and_draw(self.adv_patch,o_img_root,o_gt_root,o_save_dir)
        e="iter:{},original:--P:{},--R:{},--F:{}".format(t,P,R,F)
        self.logger.info(e)

        #=================scale=====================
        #data_root
            #test_resize
            #test_resize_gt
                #60
                #...
                #200
        # resize_scales = [item / 10 for item in range(6, 21, 1)]#0.6 0.7 0.8 ... 2.0
        # for item in resize_scales:
        #     str_s=str(int(item * 100))
        #     r_img_root=os.path.join(self.data_root,'test_resize')
        #     r_gt_root = os.path.join(self.data_root, 'test_resize_gt',str_s)
        #     r_save_dir = os.path.join(self.savedir, str(t), str_s)
        #     if os.path.exists(r_save_dir) == False:
        #         os.makedirs(r_save_dir)
        #     P,R,F=self.evaluate_and_draw(self.adv_patch,r_img_root,r_gt_root,r_save_dir)
        #     e="iter:{},scale_ratio:{},P:{},R:{},F:{}".format(t,item,P,R,F)
        #     self.logger.info(e)



import os
if __name__ == '__main__':
    RAT = RepeatAdvPatch_Attack(data_root=os.path.join(os.path.abspath('.')[:-4],"AllData"),
                                savedir=os.path.join(os.path.abspath('.')[:-4],'result_save/140_140'), log_name='140_140.log',
                                alpha=5 / 255, batch_size=20, gap=50, T=501,
                                lambdaw=0, eps=60 / 255, decay=0.2,
                                adv_patch_size=(1, 3, 130, 130),
                                feamapLoss=False)
    RAT.train()

