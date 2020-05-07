import cv2
import numpy as np
from skimage.metrics import structural_similarity


def main():
    x_img = cv2.imread('images/img1.jpg')
    y_img = cv2.imread('images/img2.jpg')

    x_tmp = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
    y_tmp = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)

    (score, df) = structural_similarity(x_tmp, y_tmp, full=True)
    df = (df * 255).astype("uint8")

    tr = cv2.threshold(df, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cont = cv2.findContours(tr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]

    for c in cont:
        area = cv2.contourArea(c)
        if area >= 15:
            x, y, a, b = cv2.boundingRect(c)
            cv2.rectangle(y_img, (x, y), (x + a, y + b), (255, 0, 0), 2)

    final = np.concatenate((x_img, y_img), axis=1)
    cv2.imshow("RESULT", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('images/finalImg.jpg', final)


if __name__ == '__main__':
    main()
