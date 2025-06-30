import numpy as np
import torch
from medpy import metric
# from keras import metrics
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch import Tensor
from typing import Iterable,Set
from scipy.ndimage import distance_transform_edt as distance
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import eliminate_false_positives, resize_image_itk
from pytorch_grad_cam import AblationCAM,GradCAM
from pytorch_grad_cam.ablation_layer import AblationLayerVit
# from vit_model import vit_base_patch16_224_in21k as create_model
import xlsxwriter as xw
import os
from math import log
import matplotlib.pyplot as plt

import numpy as np
# import cc3d
from scipy.ndimage import label




class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda()  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1



def returnCAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape        #1,2048,7,7
    output_cam = []
    for idx in class_idx:  #输出每个类别的预测效果
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))
        #(1, 2048) * (2048, 7*7) -> (1, 7*7)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img)  #Format as CV_8UC1 (as applyColorMap required)
        output_cam.append(cam_img)
    return output_cam




def _dice_loss(predict, target):

	smooth = 1e-5

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)
	intersection = torch.sum(torch.mul(y_pred_f, y_true_f), dim=1)
	union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1) + smooth
	dice_score = (2.0 * intersection / union)

	dice_loss = 1 - dice_score

	return dice_loss


class Dice_Loss(nn.Module):
	def __init__(self):
		super(Dice_Loss, self).__init__()

	def forward(self, predicts, target):

		preds = torch.softmax(predicts, dim=1)
		dice_loss0 = _dice_loss(preds[:, 0, :, :], 1 - target)
		dice_loss1 = _dice_loss(preds[:, 1, :, :], target)
		loss_D = (dice_loss0.mean() + dice_loss1.mean())/2.0

		return loss_D


class Task_Interaction_Loss(nn.Module):

    def __init__(self):
        super(Task_Interaction_Loss, self).__init__()

    def forward(self, cls_predict, seg_predict, target):
        b, c = cls_predict.shape
        h, w = seg_predict.shape[2], seg_predict.shape[3]

        target = target.view(b, 1)
        # target = torch.zeros(b, c).cuda().scatter_(1, target, 1)
        target_new = torch.zeros(b, 1).cuda()
        cls_pred = Variable(torch.zeros(b, 1)).cuda()


        target_new = target
        # target_new[:, 1] = target[:, 1]

        cls_pred = cls_predict
        # cls_pred[:, 1] = cls_predict[:, 1]

        # Log Sum Exp
        # seg_pred2 = torch.ones_like(seg_predict)
        # seg_pred2 = seg_pred2-seg_predict

        seg_pred = torch.exp(seg_predict).sum(dim=(2, 3))/ (h * w)
        seg_pred_ = seg_pred - seg_pred.mean()
        # seg_pred = torch.logsumexp(seg_predict, dim=(2, 3)) / (h * w)
        # seg_pred2 = torch.logsumexp(seg_pred2, dim=(2, 3)) / (h * w)
        # seg_pre_total = torch.cat((seg_pred,seg_pred2),dim=1)


        # JS
        torch.sign(seg_pred_)
        seg_cls_kl = F.kl_div(cls_pred, (seg_pred), reduction='none')
        cls_seg_kl = F.kl_div(seg_pred, (cls_pred), reduction='none')

        seg_cls_kl = seg_cls_kl.sum(-1)
        cls_seg_kl = cls_seg_kl.sum(-1)


        indicator1 = (cls_pred > 1-cls_pred) == (target_new > 1-target_new)
        indicator2 = (seg_pred_ > 0) == (target_new > 1-target_new)

        return (cls_seg_kl * indicator1 + seg_cls_kl * indicator2).sum() / 2. / b
        # return (cls_seg_kl * indicator1).sum()/ b




def entorpy_loss(uout):
    L2_mean = 0
    all_sort = []
    for index_i in range(uout.shape[0]):
        # if i in high_list:
        #     index_i = high_list.index(i)
        #     all_sort.append(uout[index_i,4])
        # elif i in low_list:
        #     index_i = low_list.index(i)
        all_sort.append(uout[index_i])
    all_sort = torch.stack(all_sort, dim=0)
    for i in range(1, all_sort.shape[0] - 1):
        # L2_sum = L2_sum + ((all_sort[i].view(-1) - all_sort[i-1].view(-1))**2).sum() + ((all_sort[i].view(-1) - all_sort[i+1].view(-1))**2).sum()
        L2_mean = L2_mean + ((all_sort[i].view(-1) - all_sort[i - 1].view(-1)) ** 2).mean() + (
                    (all_sort[i].view(-1) - all_sort[i + 1].view(-1)) ** 2).mean()
    # print("all_sort.shape:", all_sort.shape)
    # print("L2_mean: ", L2_mean.item())
    return L2_mean


def compute_class_weights(histogram):
  classWeights = np.ones(2, dtype=np.float32)
  normHist = histogram / np.sum(histogram)
  for i in range(2):
    classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
  return classWeights


def focal_loss(input, target):
    '''
    :param input: 
    :param target:
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)
    N = inputs.size(0)
    C = inputs.size(1)
    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    frequency = torch.tensor((number_0, number_1), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)
    weights = torch.from_numpy(classWeights).float()
    weights = weights[target.view(-1)] 
    gamma = 2
    P = F.softmax(inputs, dim=1)  # shape [num_samples,num_classes]
    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)  # shape [num_samples,num_classes] one-hot encoding
    probs = (P * class_mask).sum(1).view(-1, 1)  # shape [num_samples,]
    log_p = probs.log()
    # print('in calculating batch_loss', weights.shape, probs.shape, log_p.shape)
    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p
    # print(batch_loss.shape)
    loss = batch_loss.mean()
    return loss



def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights



def calcShannonEnt(probList):
    shannonEnt=0.0
    probList = probList.flatten()
    for prob in probList:
        if prob == 0:
            prob = 1e-10
        shannonEnt=shannonEnt-prob*log(prob,2)
    return shannonEnt/probList.shape[0]



    

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_dice=10.0, weight_bce=1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        assert input.size() == target.size(), f'predict {input.size()} & target {target.size()} shape do not match'

        input = input.float()
        target = target.float()

        # BCE loss always computed
        bce_loss = self.bce(input, target)

        # Dice loss only when foreground exists
        if torch.sum(target).item() == 0:
            dice_loss = torch.tensor(0.0, device=input.device, requires_grad=True)
        else:
            pred = torch.sigmoid(input)
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)

            smooth = 1e-5
            intersection = torch.sum(pred_flat * target_flat)
            dice = (2. * intersection + smooth) / (
                torch.sum(pred_flat ** 2) + torch.sum(target_flat ** 2) + smooth
            )
            dice_loss = 1 - dice

        return self.weight_bce * bce_loss, self.weight_dice * dice_loss
        




def IoU(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """

    seg = seg.flatten()
    gt = gt.flatten()
    if type(seg) == torch.Tensor:
        seg[seg > ratio] = 1.0
        seg[seg < ratio] = 0.0
    else:
        seg[seg > ratio] = np.float32(1)
        seg[seg < ratio] = np.float32(0)

    mix = (gt * seg).sum()
    iou = mix/(gt.sum() + seg.sum()+1e-5-mix)
    return iou

def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """

    seg = seg.flatten()
    gt = gt.flatten()
    if type(seg) == torch.Tensor:
        seg[seg > ratio] = 1.0
        seg[seg < ratio] = 0.0
    else:
        seg[seg > ratio] = np.float32(1)
        seg[seg < ratio] = np.float32(0)
    dice = (2 * (gt * seg).sum()+1e-5)/(gt.sum() + seg.sum()+1e-5)
    return dice


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = dice_coeff(pred, gt)
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = metric.binary.hd95(pred, gt)
        asd_score = metric.binary.asd(pred, gt)
        return dice, hd95, asd_score
    else:
        return dice, 30,10




def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # assert one_hot(torch.Tensor(seg), axis=0)
    seg = seg.cpu().detach().numpy()
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res




class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        pred = input.reshape(-1)
        truth = target.reshape(-1)
        bce_loss = nn.BCELoss()(pred, truth).double()
        return bce_loss



def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, img_information=None, epoch=None):
    if img_information:
        Direction = img_information[0]
        Origin = img_information[1]
        Spacing = img_information[2]
        name = img_information[3][0].strip().split('/')[-1]

    image = image
    net.eval()
    with torch.no_grad():
        outputs,features = net(image)

        features = features.cpu().detach().numpy().mean(1)
        y = np.array((0,1,2,3))
        out= outputs.cpu().detach().numpy()

        out[out > 0.5] = np.float32(1)
        out[out < 0.5] = np.float32(0)


        out = out.astype(int)
        out = eliminate_false_positives(out.squeeze(1))
        out = out[:,np.newaxis,:,:]
        image = image.cpu().detach().numpy()
       

    label = label.squeeze(0).cpu().detach().numpy()
    # print(f"[DEBUG] pred shape: {out.shape}, label shape: {label.shape}")
    dice_score = calculate_metric_percase(out, label)

    iou = IoU(out, label)

    dice_pre = dice_score[0]
    hd_95 = dice_score[1]
    asd_score = dice_score[2]

    if test_save_path is not None:
        if epoch is not None:
            test_save_path = os.path.join(test_save_path, f"epoch_{epoch}")
            os.makedirs(test_save_path, exist_ok=True)
  
        path = test_save_path+'/Sample_test_name.txt'
        with open(path,'a',encoding='utf-8') as f:
            string = f"{name} | Dice: {dice_pre:.4f}, HD95: {hd_95:.4f}, IOU: {iou:.4f}, ASD: {asd_score:.4f}\n"
            f.write(string)

        prediction = out.squeeze(1)
        prediction = np.transpose(prediction, (0, 2, 1))

        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk = resize_image_itk(prd_itk , [256,256] ,interpolator = 'linear' )

        prd_itk.SetDirection([float(Direction[0][0]),float(Direction[1][0]),float(Direction[2][0]),
                             float(Direction[3][0]),float(Direction[4][0]),float(Direction[5][0]),
                             float(Direction[6][0]),float(Direction[7][0]),float(Direction[8][0])])
        prd_itk.SetOrigin([float(Origin[0][0]),float(Origin[1][0]),float(Origin[2][0])])
        prd_itk.SetSpacing([float(Spacing[0][0]),float(Spacing[1][0]),float(Spacing[2][0])])


        label = label.squeeze(1)
        label = np.transpose(label, (0, 2, 1))


        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk = resize_image_itk(lab_itk , [256,256] ,interpolator = 'linear' )


        lab_itk.SetDirection([float(Direction[0][0]),float(Direction[1][0]),float(Direction[2][0]),
                             float(Direction[3][0]),float(Direction[4][0]),float(Direction[5][0]),
                             float(Direction[6][0]),float(Direction[7][0]),float(Direction[8][0])])
        lab_itk.SetOrigin([float(Origin[0][0]),float(Origin[1][0]),float(Origin[2][0])])
        lab_itk.SetSpacing([float(Spacing[0][0]),float(Spacing[1][0]),float(Spacing[2][0])])

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+case + "_gt.nii.gz")

    return dice_score,hd_95,iou,asd_score,iou,asd_score,features,y
    # return dice_score,hd_95,iou,asd_score,asd_score,y


def xw_toExcel(data, fileName):  
    workbook = xw.Workbook(fileName)  
    worksheet1 = workbook.add_worksheet("sheet1")  
    worksheet1.activate()  
    title = ['mean_dice']  
    worksheet1.write_row('A1', title)  
    i = 2 
    for j in range(len(data)):
        insertData = data[j]
        # insertData = [data[j]["name"],data[j]["mean_dice"], data[j]["hd95"],
        #               data[j]["asd"], data[j]["iou"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, [insertData])
        i += 1
    workbook.close()  



