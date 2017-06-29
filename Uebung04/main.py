# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
from time import *
from operator import itemgetter
import math
import cv2
from skimage.feature.tests.test_orb import img
import sift
import collections

def ransac(Iteration, MatchA, MatchB, Inlier, SizeConsenusSet):
    print("Ransac")
    MatchA = np.asarray(MatchA)
    MatchB = np.asarray(MatchB)
    indexes = list(np.random.random_integers(0, len(MatchA) - 1, Iteration * 4))
    homographien = []
    for i in range(Iteration):
        tmpA = buildMat(MatchA[indexes[i*4:i*4+4]], MatchB[indexes[i*4:i*4+4]])
        homographien.append(tmpA[2])
    distanzen = [0]*Iteration
    for i in range(Iteration):
        print("Ransac ", i, Iteration)
        tmpA = homographien[i]
        distanzen[i] = CalcEuclidicDistance(tmpA, MatchA[:,0:2], MatchB[:,0:2])


    maxIndex = np.argmin(distanzen)

    #plt.hist(distanzen, bins=500)
    #plt.show()

    return homographien[maxIndex]

def CalcEuclidicDistance(homo, ptsA, ptsB):
    calcPtsB = np.asarray(getBcord(homo,ptsA[:,0],ptsA[:,1])).T
    return np.sum(np.power(calcPtsB-ptsB,2))


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

def getOcord(aVec, bx, by):
    a1 = aVec[0, 0]
    a2 = aVec[1, 0]
    a3 = aVec[2, 0]
    b1 = aVec[3, 0]
    b2 = aVec[4, 0]
    b3 = aVec[5, 0]
    c1 = aVec[6, 0]
    c2 = aVec[7, 0]
    x = (a1 * bx + a2 * by + a3) / (c1 * bx + c2 * by + 1)
    y = (b1 * bx + b2 * by + b3) / (c1 * bx + c2 * by + 1)
    return (x,y)


def buildMat(WorldPointlist, PicPointlist):
    M = []
    vx = []
    for i in range(len(WorldPointlist)):
        (bx, by) = PicPointlist[i]
        (ox, oy) = WorldPointlist[i]
        M.append([bx, by, 1, 0, 0, 0, -ox * bx, -ox * by])
        M.append([0, 0, 0, bx, by, 1, -oy * bx, -oy * by])
        vx.append([ox])
        vx.append([oy])
    M = np.matrix(M)
    vx = np.matrix(vx)

    Minv = np.linalg.pinv(M)
    a = Minv.dot(vx)


    return (M, vx, a)

def loadIMGasNP(path):
    tmp = Image.open(path).convert('L')
    #tmp = scipy.misc.imread(name=path, )
    return np.asarray(tmp,dtype=np.uint8)



def main():
    print("Aufgabe 4")
    Handpunkte = True
    bp4= []
    bp5= []
    bp6= []

    img4 = loadIMGasNP("neue\IMG_20170622_110401.jpg")
    img5 = loadIMGasNP("neue\IMG_20170622_110407.jpg")
    img6 = loadIMGasNP("neue\IMG_20170622_110415.jpg")


    bp4.append((2945,862))
    bp4.append((4358,662))
    bp4.append((4309,1581))
    bp4.append((3050,1559))
    bp4.append((2310,1142 ))
    bp4.append((2417,1548))

    bp5.append((1641,914))
    bp5.append((2863,815))
    bp5.append((2819,1615))
    bp5.append((1737,1606))
    bp5.append((968,1168 ))
    bp5.append((1079,1602 ))

    bp6.append((263,873))
    bp6.append((1638,824))
    bp6.append((1598,1628))
    bp6.append((385,1648))





    plt.gray()
    #plt.imshow(np.asarray(img4, dtype=np.uint8))
    #x, y = zip(*bp4)
    #plt.scatter(x, y, c='r', s=20)
    #plt.show()

    plt.gray()
    #plt.imshow(np.asarray(img5, dtype=np.uint8))
    #x, y = zip(*bp5)
    #plt.scatter(x, y, c='r', s=20)
    #plt.show()


    if(Handpunkte):
        pts4, ds4 = sift.detect_and_compute(img4)
        pts5, ds5 = sift.detect_and_compute(img5)

        keep = pts4[:, 2] > 4.0
        pts4 = pts4[keep]
        ds4 = ds4[keep]
        print(len(ds4))
        keep = pts5[:, 2] > 4.0
        pts5 = pts5[keep]
        ds5 = ds5[keep]
        print(len(ds5))
        scores = sift.match(ds4,ds5)
        #sift.plot_matches(img4,img5,pts4,pts5,scores,show_below=True)
        #plt.show()
        keep = scores > 0
        matchPtsA = pts4[keep,0:2]
        matchPtsB = pts5[scores,0:2]
        matchPtsB = matchPtsB[keep]

        homographie = ransac(80, matchPtsA,matchPtsB, 100, 0)

        im3 = sift.appendimages(img4, img5)
        # show image
        plt.imshow(im3)

        # draw lines for matches
        cols1 = img4.shape[1]
        for x, y in matchPtsA[0:100]:
            newx, newy = getBcord(homographie, x, y)
            plt.plot([x, cols1 + newx], [y, newy], 'c')

        for x, y in matchPtsB[0:100]:
            plt.scatter(cols1 + x, y, c='r', s=20)

        plt.axis('off')
        plt.show()
    else:
        homographie = ransac(4, bp4, bp5, .2, 0)

        im3 = sift.appendimages(img4, img5)
        # show image
        plt.imshow(im3)

        # draw lines for matches
        cols1 = img4.shape[1]
        for x, y in bp4:
            newx, newy = getBcord(homographie, x, y)
            plt.plot([x, cols1 + newx],[y, newy], 'c')

        for x, y in bp5:
            plt.scatter(cols1 + x, y, c='r', s=20)

        plt.axis('off')
        plt.show()
        #sift.plot_features(img4, pts4, True)

if __name__ == "__main__":
    main()