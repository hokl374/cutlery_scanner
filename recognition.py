# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:51:13 2018

@author: hokl3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from skimage import io as skio
import skimage
import skimage.segmentation
from skimage import filters, measure, morphology, restoration, transform
import scipy
import pandas as pd
from skimage.color import rgb2gray

TARGET_SIZE = (720,1280)

def sep_and_strip_img(image):
    image_labels, segments = measure.label(image, background=0, return_num=True)
#    print("there were {} segments".format(segments))

    # remove artifacts connected to image border
    #cleared_labeled = skimage.segmentation.clear_border(image_labels)
    return image_labels

def downsize(image):
    shape = image.shape
    downsampled = transform.rescale(image,max(TARGET_SIZE[0]/shape[0],TARGET_SIZE[1]/shape[1]))
    return downsampled[:TARGET_SIZE[0],:TARGET_SIZE[1]]

def read_image(name):
    image = skio.imread(name)
    return rgb2gray(downsize(image))

def edge_filter(image,threshold=0.1):
    c = filters.frangi(image)
    c = restoration.denoise_bilateral(c,multichannel=False)
    threshold_val = filters.threshold_otsu(c)
    masked = morphology.closing(c > threshold_val*threshold)
    return masked

def otsu_process(image,threshold = 1):
    threshold_val = filters.threshold_otsu(image)
    masked = morphology.closing(image > threshold_val*threshold)
    masked = filters.gaussian(masked)
    final = sep_and_strip_img(masked)
    return final

def mean_process(image,threshold = 1.3):
    threshold_val = filters.threshold_mean(image)
    masked = morphology.closing(image > threshold_val*threshold)
    masked = filters.gaussian(masked)
    final = sep_and_strip_img(masked)
    return final

def get_region(masked):
    data = []
    for region in measure.regionprops(masked):
        # take regions with large enough areas
        if region.area >= masked.size/100:
            data.append(region)
    if data != []:
        region = sorted(data,key=lambda x: x.area,reverse=True)[0]
        return region
    else:
        return None

def region_transform(image,region,buffer_ratio = 1.35):
    minr, minc, maxr, maxc = region.bbox
    if (maxr-minr)*(buffer_ratio) < TARGET_SIZE[0]:
        dheight = int((maxr-minr)*(buffer_ratio-1))
        minr,maxr= minr-dheight,maxr+dheight
        minr,maxr= 0 if minr<0 else minr, TARGET_SIZE[0]-1 if maxr>=TARGET_SIZE[0] else maxr
        image = image[minr:maxr,:]
    if (maxc-minc)*buffer_ratio < TARGET_SIZE[1]:
        dwidth = int((maxc-minc)*(buffer_ratio-1))
        minc,maxc= minc-dwidth,maxc+dwidth
        minc,maxc= 0 if minc<0 else minc, TARGET_SIZE[1]-1 if maxc>=TARGET_SIZE[1] else maxc
        image = image[:,minc:maxc]
    rotated = skimage.transform.rotate(image,-np.rad2deg(region.orientation),preserve_range=True)
    return rotated

def recognise_dim(image,source,region_source,threshold = 0.01, sigma = [5,5],alpha = 0.05):
    rotated = region_transform(image,region_source)
    rotated = filters.gaussian(rotated,sigma = sigma)
    threshold_val = filters.threshold_mean(rotated)
    rotated = morphology.closing(rotated > threshold_val*threshold)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rotated,cmap='gray')    
    rotated = sep_and_strip_img(rotated)
    region = get_region(rotated)
    
    if region != None:
        minr, minc, maxr, maxc = region.bbox
        width,height = maxc-minc,maxr-minr
        ratio = width/height
        
        #width_stats = {'spoon':(765.111111111, 55.7702518163),'fork':(834.8, 69.9882847339),'knife':(894.2, 21.1745129814),'teaspoon':(528.0,33.5315319716)}
        width_stats = {'spoon':(765.111111111, 55.7702518163),'fork':(834.8, 69.9882847339),'knife':(894.2, 50),'teaspoon':(528.0,33.5315319716)}
        #Increased SD for knife to 50 to be more aligned with widths of spoon and fork
        
        #ratio_stats = {'spoon':(4.75301461134,0.434855062715),'fork':(6.2924293741, 0.232221245955),'knife':(9.67666724882,0.135291912287),'teaspoon':(5.17696281099,0.555381329707)}
        ratio_stats = {'spoon':(4.75301461134,0.434855062715),'fork':(6.2924293741, 0.4),'knife':(9.67666724882,0.4),'teaspoon':(5.17696281099,0.555381329707)}
        #Increased SD of spoon,fork,knife to 0.5
        
        #To 95% confidence that object is teaspoon and 95% confidence that object is not any other object
        print(width,ratio)
        
        width_confidence = {}
        ratio_confidence = {}
        for key,width_param in width_stats.items():
            prob = scipy.stats.norm.cdf((width-width_param[0])/width_param[1])
            if prob > 0.5:
                conf = ((1-prob)*2)
            else:
                conf = (prob*2)
            print(key,'width',conf)
            width_confidence[key] = (conf>=alpha,conf)
            
            prob = scipy.stats.norm.cdf((ratio-ratio_stats[key][0])/ratio_stats[key][1])
            if prob > 0.5:
                conf = ((1-prob)*2)
            else:
                conf = (prob*2)
            print(key,'ratio',conf)
            ratio_confidence[key] = (conf>=alpha,conf)
        
        width_positive, width_negative = list(filter(lambda x: x[1][0], width_confidence.items())), list(filter(lambda x: not x[1][0], width_confidence.items()))
        ratio_positive, ratio_negative = list(filter(lambda x: x[1][0], ratio_confidence.items())), list(filter(lambda x: not x[1][0], ratio_confidence.items()))
        
        
        if len(width_positive)==1 and len(ratio_positive)==1 and width_positive[0][0] == ratio_positive[0][0]:
            #Both Width and Ratio positively identify one cutlery
            return width_positive[0][0]
        elif len(width_positive) == 1 and width_positive[0][0] == 'teaspoon':
            #Width positively identifies teaspoon - redundant?
            return 'teaspoon'
        elif len(ratio_positive) == 1 and ratio_positive[0][0] == 'knife':
            #Ratio positively identifies knife - redundant?
            return 'knife'
        elif len(ratio_positive) == 1:
            #Only one clear winner for ratio
            return ratio_positive[0][0]
        elif len(width_positive) == 1:
            #Only one clear winner for width
            return width_positive[0][0]
        elif len(ratio_positive) == 2 and 'fork' in list(map(lambda x: x[0],ratio_positive)):
            #Use ORB as tiebreaker between fork and other
            best_match = find_match(source,region_source)
            if best_match == 'fork':
                return 'fork'
            else:
                return list(filter(lambda x: x!='fork',map(lambda x: x[0],ratio_positive)))[0]
        elif len(width_positive) == 2 and 'fork' in list(map(lambda x: x[0],width_positive)):
            #Use ORB as tiebreaker between fork and other
            best_match = find_match(source,region_source) #Handle fork and spoon
            if best_match == 'fork':
                return 'fork'
            else:
                return list(filter(lambda x: x!='fork',map(lambda x: x[0],width_positive)))[0]
        elif len(ratio_positive)!=0 and len(width_positive)!=0 and (sorted(width_positive,key=lambda x: x[1][0])[0][0] == sorted(ratio_positive,key=lambda x: x[1][0])[0][0]):
            #Both width and ratio have a common winner
            return sorted(ratio_positive,key=lambda x: x[1][0])[0][0]
        else:
            #Unable to identify
            print(width,ratio)
            print(width_positive,width_negative)
            print(ratio_positive, ratio_negative)
            return "unknown"
    else:
        #No opject identified
        return None

def create_pd_frame(region=False):
    if region is False:
        frame = pd.DataFrame(columns=["Area", "Orientation", "BBoxX", "BBoxY", "Type_o_Object"])
    else:
        minr, minc, maxr, maxc = region.bbox
        frame = pd.DataFrame([[region.area, region.orientation, maxr - minr, maxc - minc, False]],
                             columns=["Area", "Orientation", "BBoxX", "BBoxY", "Type_o_Object"])
    #print(frame)
    return frame


def new_plot(images):
    frames = []
    for image in images:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image,cmap='gray')
        data = create_pd_frame()
        for region in measure.regionprops(image):
            # take regions with large enough areas
            if region.area >= image.size/100:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                ax.add_patch(mpatch.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2))
                region_data = create_pd_frame(region=region)
                data = data.append(region_data, ignore_index=True)
        frames.append(data)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return

def match_score(transformed_image,template,max_distance=10):
    descriptor_extractor = skimage.feature.ORB(n_keypoints=150)

    descriptor_extractor.detect_and_extract(transformed_image)
    #keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors
    
    descriptor_extractor.detect_and_extract(template)
    #keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors
    
    return len(skimage.feature.match_descriptors(descriptors1, descriptors2, cross_check=True,max_distance = max_distance))

def find_match(image,region):
    template_names = {'spoon1.png':'spoon','spoon9.png':'spoon', 'fork2.png':'fork', 'fork9.png':'fork', 'knife1.png':'knife', 'knife8.png':'knife','teaspoon3.png':'teaspoon', 'teaspoon7.png':'teaspoon'}
    edged = edge_filter(image)
    masked = region_transform(edged,region)
    scores = []
    
    for name,cutlery in template_names.items():
        template = skio.imread(name)
        scores.append((name,cutlery,match_score(masked,template)))
    
#    print(scores)
    best_match = sorted(scores,key=lambda x:x[2],reverse=True)[0]
    next_match = sorted(scores,key=lambda x:x[2],reverse=True)[1]
#    print(best_match)
    
    if best_match[2]-next_match[2] > 10:
        return best_match[1]
    else:
        return None
#
#for name in ['knife'+str(i) for i in range(1,11)]+['teaspoon'+str(i) for i in range(1,12)]+['fork'+str(i) for i in range(1,11)]+['spoon'+str(i) for i in range(1,10)]:
#    image = read_image('photos/'+name+'.jpg')
#    masked = mean_process(image)
#    region = get_region(masked)
#    print('Filename:', name)
#    print('Cutlery:', recognise_dim(masked,image,region))
#        
#        



