"""
import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),10,(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
"""

########################################################

import cv2
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1
thick = 30

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thick, 0,-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),thick,0,-1)

    pass


img = np.zeros((512, 512), np.uint8) + 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        img = np.zeros((512,512), np.uint8) + 255
    elif k == 27 or k == ord(' '):
        break

cv2.destroyAllWindows()

save_img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
cv2.threshold(save_img, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite('numberTest.png', save_img)




