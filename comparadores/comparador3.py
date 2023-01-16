import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('base.png')
template = cv2.imread('pecas/a.jpg')

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method = eval(meth)
    heat_map = cv2.matchTemplate(image, template, method)

    h, w, _ = template.shape
    y, x = np.unravel_index(np.argmax(heat_map), heat_map.shape)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 5)

    cv2.imwrite(f'comparador3 - {method}.png',image)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))