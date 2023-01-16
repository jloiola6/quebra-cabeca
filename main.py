import cv2
import numpy as np
from matplotlib import pyplot as plt

class QuebraCabeca():

    def __init__(self):
        self.pecas_anterior = 0
        self.cap = cv2.VideoCapture(0)

        # objetos  detectados pela camera
        self.object = cv2.createBackgroundSubtractorMOG2()

    
    def exibir_resultado(self, titulo, frame):
        cv2.imshow(titulo, frame)

    
    def comparador_imagens(self, peca):
        self.large_image = cv2.imread('media/base.png')
        result = cv2.matchTemplate(peca, self.large_image, cv2.TM_SQDIFF_NORMED)   

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
        cv2.rectangle(self.large_image, (MPx-tcols,MPy-trows),(MPx+tcols,MPy+trows),(0,0,255),2)

        # Display the original image with the rectangle around the match.
        self.exibir_resultado('output', self.large_image)

        # # The image is only displayed if we call this
        # cv2.waitKey(0)


    def identificar_peca(self, frame, x, y, w, h):
        try:
            inicio_coluna = x 
            fim_coluna = x + w 
            
            inicio_linha = y
            fim_linha = y + h 

            peca = frame[inicio_linha : fim_linha, inicio_coluna : fim_coluna]
            self.exibir_resultado('Peça', peca)

            self.comparador_imagens(peca)

            # cv2.imwrite(filename=f'pecas/a.jpg', img=peca)
        except:
            pass

    
    def iniciar(self):
        while True:
            self.qtd_pecas = 0

            ret, frame = cap.read()

            # Extraindo região importantes
            self.roi = frame[10:600, 10:600]

            self.mask = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
            self._, self.threshold = cv2.threshold(self.mask, 40, 255, cv2.THRESH_BINARY_INV)
            # self._, self.threshold = cv2.threshold(self.mask, 100, 255, cv2.THRESH_BINARY)
            # _, threshold = cv2.threshold(mask, 255, 40, cv2.THRESH_BINARY)
            
            self.contours, self._ = cv2.findContours(self.threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in self.contours:
                self.area = cv2.contourArea(cnt)

                # Calculando area e removendo elementos
                if 4000 > self.area > 1000:
                    self.qtd_pecas += 1
                    (x, y, w, h) = cv2.boundingRect(cnt)

                    self.identificar_peca(self.roi, x, y, w, h)

                    cv2.rectangle(self.roi, (x, y), (x + w , y + h), (0, 0, 255), 2)
                    cv2.putText(self.roi, str(self.area), (x+(w//5), y+(h//2)), 1, 1, ((0, 255, 0)))
                    
            
            if self.qtd_pecas != self.pecas_anterior:
                print(self.qtd_pecas)
                self.pecas_anterior = self.qtd_pecas

            # self.exibir_resultado()
            # self.exibir_resultado('Frame', frame)
            # self.exibir_resultado('Mask', self.mask)
            self.exibir_resultado('roi', self.roi)
            self.exibir_resultado('Mascara', self.threshold)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



quebra_cabeca = QuebraCabeca()
cap = quebra_cabeca.cap

quebra_cabeca.iniciar()

cap.release()
cv2.destroyAllWindows()
