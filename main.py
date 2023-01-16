import cv2
import numpy as np
from matplotlib import pyplot as plt

pecas = []

def cada_peca(roi, x, y, w, h):
    try:
        inicio_coluna = x 
        fim_coluna = x + w 
        
        inicio_linha = y
        fim_linha = y + h 

        peca = roi[inicio_linha : fim_linha, inicio_coluna : fim_coluna]
        large_image = cv2.imread('base.png')

        result = cv2.matchTemplate(peca, large_image, cv2.TM_CCORR)   

        # We want the minimum squared difference
        mn,_,mnLoc,_ = cv2.minMaxLoc(result)

        # Draw the rectangle:
        # Extract the coordinates of our best match
        MPx,MPy = mnLoc

        # Step 2: Get the size of the template. This is the same size as the match.
        trows,tcols = peca.shape[:2]

        # Step 3: Draw the rectangle on large_image
        # MPx -= 22
        # MPy += 22
        cv2.rectangle(large_image, (MPx-tcols,MPy-trows),(MPx+tcols,MPy+trows),(0,0,255),2)

        # Display the original image with the rectangle around the match.
        cv2.imshow('output',large_image)

        # # The image is only displayed if we call this
        # cv2.waitKey(0)

        # cv2.imwrite(filename=f'pecas/a.jpg', img=peca)
        # cv2.imshow('Peça', peca)
    except:
        pass

# Criar um Tracker
# tracker = EuclideanDistTracker()

cap = cv2.VideoCapture(1)

# objetos  detectados pela camera
object = cv2.createBackgroundSubtractorMOG2()

pecas_anterior = 0
while True:
    qtd_pecas = 0

    ret, frame = cap.read()

    # Extraindo região importantes
    roi = frame[10:600, 10:600]

    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    # _, threshold = cv2.threshold(mask, 255, 40, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Calculando area e removendo elementos
        if 4000 > area > 1000:
            qtd_pecas += 1
            (x, y, w, h) = cv2.boundingRect(cnt)

            cada_peca(roi, x, y, w, h)

            cv2.rectangle(roi, (x, y), (x + w , y + h), (0, 0, 255), 2)
            cv2.putText(roi, str(area), (x+(w//5), y+(h//2)), 1, 1, ((0, 255, 0)))
    
            
    
    if qtd_pecas != pecas_anterior:
        print(qtd_pecas)
        pecas_anterior = qtd_pecas


    # cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('roi', roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()