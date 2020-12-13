#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib


dice = np.zeros(33);
jaccard = np.zeros(33);
sensitivity = np.zeros(33);
specificity = np.zeros(33);

for subject in range(0,33):

    image = nib.load('/home/alex/Downloads/Mouse-Segmentation-master/mouse_full_data_120720.nii')
    data = image.get_fdata()
    pred_pre = np.load('/home/alex/Downloads/Mouse-Segmentation-master/results/pred_' + str(subject) + '.npy')
    pred = np.reshape(pred_pre, [200,180,100])
    
    true_pre = data[subject,:,:,:,1]
    true = true_pre.transpose(2,0,1)
    
    
    pred_bin = np.copy(pred)
    
    pred_bin[pred_bin >= 0.5] = 1
    pred_bin[pred_bin < 0.5] = 0
    
    
    # slice = 100;
    
    
    
    TP = sum(sum(sum(pred_bin*true)))
    # b = 2*sum(sum(sum(TP)))
    # c = sum(sum(sum(pred_bin)))
    # d = sum(sum(sum(true)))
    TN = 100*180*200-sum(sum(sum(pred_bin+true-(pred_bin*true))))
    FP =sum(sum(sum(pred_bin-true*pred_bin)))
    FN = sum(sum(sum(true-true*pred_bin)))
    
    
    dice[subject] = (2*TP)/(FP+FN+(2*TP))
    jaccard[subject] = TP/(TP+FP+FN)
    sensitivity[subject] = TP/(TP+FN)
    specificity[subject] = TN/(FP+TN)









# plt.figure()
# plt.imshow(np.squeeze(true[slice,:,:]), cmap = 'gray')
# plt.colorbar()
# plt.figure()
# plt.imshow(np.squeeze(pred[slice,:,:]), cmap = 'gray')
# plt.colorbar()

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(np.squeeze(pred_bin[slice,:,:]), cmap = 'gray')
# ax1.set_title('pred')

# ax2.imshow(np.squeeze(pred_bin[slice,:,:]+ true[slice,:,:]), cmap = 'gray')
# ax2.set_title('pred vs true')




    
    








