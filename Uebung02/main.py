import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
from affinTransformation import AffinTransformation
import math
from numpy.linalg import inv

def getPx(srcX, srcY, img):
    if srcX < 0 or srcY < 0:
        return 0
    if srcX >= img.shape[1] or srcY >= img.shape[0]:
        return 0
    return img[srcY, srcX]


def main():
    # Übung 2
    print("Übung 2")
    img = scipy.misc.imread(name="schraegbild_tempelhof.jpg", flatten=True)

    at = AffinTransformation()

    bx1 = 312
    by1 = 432
    ox1 = 312
    oy1 = 432

    bx2 = 343
    by2 = 423
    ox2 = 343
    oy2 = 432

    bx3 = 345
    by3 = 337
    ox3 = 312
    oy3 = 337

    bx4 = 363
    by4 = 337
    ox4 = 343
    oy4 = 337


    M = np.matrix([[bx1, by1, 1, 0, 0, 0, -ox1 * bx1, -ox1 * by1],
                   [0, 0, 0, bx1, by1, 1, -oy1 * bx1, -oy1 * by1],
                   [bx2, by2, 1, 0, 0, 0, -ox2 * bx2, -ox2 * by2],
                   [0, 0, 0, bx2, by2, 1, -oy2 * bx2, -oy2 * by2],
                   [bx3, by3, 1, 0, 0, 0, -ox3 * bx3, -ox3 * by3],
                   [0, 0, 0, bx3, by3, 1, -oy3 * bx3, -oy3 * by3],
                   [bx4, by4, 1, 0, 0, 0, -ox4 * bx4, -ox4 * by4],
                   [0, 0, 0, bx4, by4, 1, -oy4 * bx4, -oy4 * by4]
                   ])
    Minv = inv(M)
    vx = np.matrix([ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4])
    a = Minv.dot(vx.T)

    print(vx.T)


    # a enthält: a1, a2, a3, b1, b2, b3, c1, c2


    print(getBcord(a, 1, 1))
    print(getBcord(a, 0, 0))
    print(getBcord(a, 2, 2))
    print(getBcord(a, 3, 3))

    dest = np.zeros(img.shape)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            (nx, ny) = getBcord(a, x, y)
            dest[y,x] = getPx(nx, ny, img)

    at.plot_image(dest)

def getBcord(aVec, bx ,by ):
    #x = (aVec[0,0]*bx + aVec[1,0]*by + aVec[2,0])/( aVec[6,0] * bx + aVec[7,0]*by + 1)
    #y = (aVec[3,0]*bx + aVec[4,0]*by + aVec[5,0])/( aVec[6,0] * bx + aVec[7,0]*by + 1)
    a1 = aVec[0, 0]
    a2 = aVec[1, 0]
    a3 = aVec[2, 0]
    b1 = aVec[3, 0]
    b2 = aVec[4, 0]
    b3 = aVec[5, 0]
    c1 = aVec[6, 0]
    c2 = aVec[7, 0]
    divisor = (b1*c2 - b2*c1)*bx + (a2*c1 - a1*c2)*by + a1*b2 - a2*b1
    x = (b2 - c2 * b3) * bx + (a3 * c2 - a2) * by + a2 * b3 - a3 * b2
    y = (b3 * c1 - b1) * bx + (a1 - a3 * c1) * by + a3 * b1 - a1 * b3

    return (x/divisor, y/divisor)


if __name__ == "__main__":
    main()
