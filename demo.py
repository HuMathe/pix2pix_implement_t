import cv2 
import numpy as np
imgs = (
    (46, 12),
    (26, 71),
    (64, 3),
    (90, 8),
)
demo = None
for row in imgs:
    rimg = None
    for img in row:
        raw_img_l = cv2.imread('datasets\\acades\\val\\'+str(img)+'.jpg')
        imgl = np.concatenate((raw_img_l[:,256:,:], raw_img_l[:,:256,:]), axis = 1)
        imgr = cv2.imread('results\\val\\'+str(img)+'.jpg')
        img = np.concatenate((imgl, imgr), axis = 1)
        if rimg is None:
            rimg = img 
        else:
            rimg = np.concatenate((rimg, img), axis = 1)
    if demo is None:
        demo = rimg 
    else:
        demo = np.concatenate((demo, rimg), axis = 0)
cv2.imwrite('demo.jpg', demo)