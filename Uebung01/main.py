import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
import math

Winkel = 0
OffsetX = 0
OffsetY = 0
VerzerrungX = 0
VerzerrungY = 0


def calcM(Winkel=0, OffsetX=0, OffsetY=0, ZerrungX=0, ZerrungY=0, StreckungX=1, StreckungY=1):


    rotM = np.array([[math.cos(Winkel * math.pi / 180), -math.sin(Winkel * math.pi / 180)],
                     [math.sin(Winkel * math.pi / 180), math.cos(Winkel * math.pi / 180)]])
    zerM = np.array([[1, ZerrungX],
                     [ZerrungY, 1]])
    strM =  np.array([[1.0/StreckungX, 0],
                     [0, 1.0/StreckungY]])

    Moff = np.array([[OffsetX],[OffsetY]])
    print(np.dot(zerM,rotM))
    retM = np.concatenate((np.dot(strM.T,np.dot(zerM,rotM)),Moff),axis=1)

    return retM


m = calcM(0, 0, -0, -1, 0, 1, 1)


def coordTransform(x, y, m):
    newX = m[0, 0] * x + m[0, 1] * y - m[0, 2]
    newY = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return (newX, newY)


def getPx(x, y, img, m):
    (srcX, srcY) = coordTransform(x, y, m)
    if srcX < 0 or srcY < 0:
        return 0
    if srcX >= img.shape[1] or srcY >= img.shape[0]:
        return 0
    return img[srcY, srcX]


def applyTransform(img, M, BilinearInterp):
    target = np.zeros(img.shape)
    print(target.shape)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            target[y, x] = getPx(x, y, img, m)

    print(target.shape)
    plt.imshow(target, cmap="gray")
    plt.show()


def main():
    img = scipy.misc.imread(name="gletscher.jpg", flatten=True)
    print(img.shape)
    applyTransform(img, m, True)


if __name__ == "__main__":
    main()
