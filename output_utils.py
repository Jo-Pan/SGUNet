import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

def load_data(im_id, model_name=None, no_aug=False): 
    x = scipy.ndimage.imread('./data/image/' + im_id, mode='L')
    y = scipy.ndimage.imread('./data/label/' + im_id, mode='L')
    if model_name is not None:
        prob = scipy.ndimage.imread('./output/{}/{}'.format(model_name, im_id) , mode='L')
        if no_aug:
            prob = scipy.ndimage.imread('./output/no_testAug/{}/{}'.format(model_name, im_id) , mode='L')
        pred = prob>255/2.0
        return x,y,prob,pred
    else:
        return x,y

def myplot(ims, probs=None, preds= None, gt = None, cols = 3, figsize=(20,10), preds_level=244, postprocess=True,
          ids=None):
    rows = 1
    fig = plt.figure(figsize=figsize)
    dices = []
    if len(ims)>cols:
        rows = len(ims)//cols +1
    
    for i in range(len(ims)):
        ax = fig.add_subplot(rows, cols, i+1)
        #plt.subplot(rows, cols, i+1)
        #plt.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        title = str(i)
        if ids is not None:
            title += ' {}'.format(ids[i])
        
        # original image
        plt.imshow(ims[i], cmap='gray')
        
        # probability map
        if probs is not None:
            plt.imshow(probs[i], alpha=0.4, cmap='Oranges', vmax=255, vmin=0)
        
        # pred contour
        if preds is not None:
            if postprocess:
                preds[i] = postprocess_pred(preds[i])
            plt.contour(preds[i], levels=[preds_level], colors=('r'))
        
        # gt contour
        if gt is not None:
            plt.contour(gt[i], levels=[244], colors=('g'))
            if preds is not None:
                dice = dice_coef(gt[i]>244, preds[i].astype(int), axis=(0,1))
                title += ' [{:.1%}]'.format(dice)
                dices.append(dice)
        plt.title(title)
    if len(dices)>0: print('average dice: ', np.mean(dices))
    plt.tight_layout()

def postprocess_pred(pred):
    labeled = label(pred)
    unique_l = np.unique(labeled)
    if len(unique_l) <= 2:
        return pred
    maxl = 0
    maxarea = 0
    for l in unique_l:
        if l ==0:
            continue
        area = np.sum(labeled==l) 
        if area > maxarea:
            maxl = l
            maxarea = area
    labeled[labeled!=maxl] =0
    labeled[labeled!=0] = 1
    
    #labeled = binary_dilation(gaussian_filter(b,2), iterations=10)
    return labeled

def iou(gt, pred):
    if np.max(gt) >200:
        gt = gt>125
    intersetion = np.logical_and(pred==1, gt==1)
    union = np.logical_or(pred==1, gt==1)
    iou = np.sum(intersetion)/np.sum(union)
    return np.mean(iou)

def dice_coef(target, prediction, axis=(0, 1), smooth=0.01):
    """
    Dice: prediction (0 or 1)
    Soft Dice: prediction (prob 0 to 1)

    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    if np.max(target) >200:
        target = target >125

    intersection = np.sum(target * prediction, axis=axis)
    union = np.sum(target + prediction, axis=axis)
    numerator = 2. * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator
    return np.mean(coef)

def get_eval_metrics(gt, pred,  metrics = ['accuracy','recall', 'precision', 'specificity', 'dice','iou']):    
    pred = pred.astype(int)
    if np.max(gt) >100:
        gt = (gt>244).astype(int)
    out=[]
    
    TP = np.sum((gt==1) & (pred==1))
    FN = np.sum((gt==1) & (pred==0))
    FP = np.sum((gt==0) & (pred==1))
    TN = np.sum((gt==0) & (pred==0))
    
    for metric in metrics:
        if metric =='dice':
            out.append(dice_coef(gt, pred, axis=(0,1)))        
        elif metric == 'recall':
            out.append(TP/(TP+FN))
        elif metric == 'precision':
            out.append(TP/(TP+FP))
        elif metric == 'specificity':
            out.append(TN/(TN+FP))
        elif metric == 'accuracy':
            out.append((TP+TN)/ (TP+FN+FP+TN))
        elif metric == 'iou':
            out.append(TP/(TP+FP+FN))
    return out