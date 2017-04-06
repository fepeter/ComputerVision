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
    strM = np.array([[1.0 / StreckungX, 0],
                     [0, 1.0 / StreckungY]])

    Moff = np.array([[OffsetX], [OffsetY]])
    print(np.dot(zerM, rotM))
    retM = np.concatenate((np.dot(strM.T, np.dot(zerM, rotM)), Moff), axis=1)

    return retM


def coordTransform(x, y, m):
    newX = m[0, 0] * x + m[0, 1] * y - m[0, 2]
    newY = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return (newX, newY)


def getPx(srcX, srcY, img):
    if srcX < 0 or srcY < 0:
        return 0
    if srcX >= img.shape[1] or srcY >= img.shape[0]:
        return 0
    return img[srcY, srcX]


def applyTransform(img, m, BilinearInterp):
    target = np.zeros(img.shape)
    print(target.shape)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            (srcX, srcY) = coordTransform(x, y, m)
            if BilinearInterp:
                midX = int(srcX + 1)
                midY = int(srcY + 1)
                target[y, x] = getPx(srcX, srcY, img) * (midX - srcX) * (midY - srcY) + \
                               getPx(srcX + 1, srcY, img) * (srcX + 1 - midX) * (midY - srcY) + \
                               getPx(srcX, srcY + 1, img) * (midX - srcX) * (srcY + 1 - midY) + \
                               getPx(srcX + 1, srcY + 1, img) * (srcX + 1 - midX) * (srcY + 1 - midY)
            else:
                target[y, x] = getPx(srcX, srcY, img)
    return target


def main():
    m = calcM(5, 0, -0, -0, 0, 1, 1)
    img = scipy.misc.imread(name="gletscher.jpg", flatten=True)
    RGBimg = scipy.misc.imread(name="ambassadors.jpg", )
    print(img.shape)
    print(RGBimg.shape)

    # Transform gletscher.jpg
    out1 = applyTransform(img, m, True)

    # Transform ambassadors.jpg
    RGBnew = RGBimg
    for i in range(3):
        print("Run Color" + str(i))
        RGBnew[:, :, i] = applyTransform(RGBimg[:, :, i], m, False)

    plt.ion()
    plt.imshow(RGBnew)
    plt.show(block=True)


if __name__ == "__main__":
    main()
