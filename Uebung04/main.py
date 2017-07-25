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
import random

print("Calculatin Distance Pic")
t1 = clock()

sh = 3480
sw = 4640

distance = np.ones((sh, sw),dtype=np.float)
xLine = np.ones((1, sw),dtype=np.float)
yLine = np.ones((sh, 1),dtype=np.float)
for i in range(sw):
    xLine[0,i] = ((sw / 2) - i)

for i in range(sh):
    yLine[i,0] = ((sh / 2) - i)

xLine = (sw/2)-xLine.__abs__()
yLine = (sh/2)-yLine.__abs__()

xLine /= (sh/2)
yLine /= (sw/2)

distance[range(sh)] = xLine * 255

distance[:, range(sw)] *= yLine
t2 = clock()


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
    consensusSet = [0]*Iteration
    for i in range(Iteration):
        print("Ransac ", i, Iteration)
        tmpA = homographien[i]
        (distanzen[i],consensusSet[i]) = CalcEuclidicDistance(tmpA, MatchA[:,0:2], MatchB[:,0:2],Inlier)


    maxIndex = np.argmax(consensusSet)


    #plt.hist(distanzen, bins=500)
    #plt.show()

    return homographien[maxIndex]

def CalcEuclidicDistance(homo, ptsA, ptsB, Inlier):
    calcPtsB = np.asarray(getBcord(homo,ptsA[:,0],ptsA[:,1])).T
    distances = calcPtsB-ptsB
    inliers = len(distances[distances<Inlier])
    return (np.sum(np.power(distances,2)),inliers)


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
    tmp = scipy.misc.imread(name=path)
    return np.asarray(tmp,dtype=np.float32)

def loadIMGasNPbw(path):
    tmp = Image.open(path).convert('L')
    #tmp = scipy.misc.imread(name=path, )
    return np.asarray(tmp,dtype=np.uint8)


def transformation(A, src, method, rgb):
    # Bildmittelpunkt
    ox = 0#src.shape[1] // 2
    oy = 0#src.shape[0] // 2

    if rgb:
        sh, sw, sd = src.shape
    else:
        sh, sw = src.shape

    # Eckpunkte des transformierten Bildes berechnen
    x = np.array([0, sw, sw, 0]) - ox
    y = np.array([0, 0, sh, sh]) - oy
    print(x, y)

    corners = getOcord(A, x, y)
    #print(getOcord(A, x, y))
    #TODO
    #corners = getBcord()
    #print("corners", corners)
    #print(corners[0])
    cx = corners[0] + ox
    cy = corners[1] + oy
    #print("cx", cx, "cy", cy)

    # Groesse des neuen Bildes
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))
    offsetX, offsetY = (int(np.ceil(c.min())) for c in (cx, cy))
    print(offsetX, offsetY)
    #print(getBcord(A, 0 + cx.min(), 0+ cy.min()))
    #print(getBcord(A, dw + cx.min(), 0+ cy.min()))
    #print(getBcord(A, dw + cx.min(), dh+ cy.min()))
    #print(getBcord(A, 0 + cx.min(), dh+ cy.min()))
    #print("DH:", dh, "DW", dw)

    # distanz von der Bildmitte entspricht gewichtung (eigentlich Alpha Kanal.)


    #meshgrid für Bildkopie vorbereiten
    dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))

    (sx, sy) = getBcord(A, dx + cx.min(), dy + cy.min())


    if method == 'nn':
        sx, sy = sx.round().astype(int), sy.round().astype(int)
    else:
        vx = sx.flatten()
        vy = sy.flatten()

        v = np.vstack((vx, vy))

        p1 = np.floor(np.vstack((vx, vy))).astype('int16')
        p2 = np.vstack((p1[0] + 1, p1[1] + 0))
        p3 = np.vstack((p1[0] + 0, p1[1] + 1))
        p4 = np.vstack((p1[0] + 1, p1[1] + 1))

        a1 = np.prod(np.abs(p1 - v), axis=0)
        a2 = np.prod(np.abs(p2 - v), axis=0)
        a3 = np.prod(np.abs(p3 - v), axis=0)
        a4 = np.prod(np.abs(p4 - v), axis=0)

    # Maske für gültige Koordinaten
    mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)

    if rgb:
        dest = np.empty(shape=(dh, dw, 4), dtype=src.dtype)
    else:
        dest = np.empty(shape=(dh, dw, 2), dtype=src.dtype)

    if method == 'nn':
        if rgb:
            dest[dy[mask], dx[mask], 0:3] = src[sy[mask], sx[mask]]
            dest[dy[mask], dx[mask], 3] = distance[sy[mask], sx[mask]]
        else:
            dest[dy[mask], dx[mask], 0] = src[sy[mask], sx[mask]]
            dest[dy[mask], dx[mask], 1] = distance[sy[mask], sx[mask]]

    else:
        mask_flattened = (0 <= p1[0]) & (p1[0] < sw) & (0 <= p1[1]) & (p1[1] < sh) & \
                         (0 <= p2[0]) & (p2[0] < sw) & (0 <= p2[1]) & (p2[1] < sh) & \
                         (0 <= p3[0]) & (p3[0] < sw) & (0 <= p3[1]) & (p3[1] < sh) & \
                         (0 <= p4[0]) & (p4[0] < sw) & (0 <= p4[1]) & (p4[1] < sh)

        mask = np.reshape(mask_flattened, dest.shape[0:2])

        if rgb:
            a1 = np.vstack((a1, a1, a1)).T
            a2 = np.vstack((a2, a2, a2)).T
            a3 = np.vstack((a3, a3, a3)).T
            a4 = np.vstack((a4, a4, a4)).T

        dest[dy[mask], dx[mask]] = a4[mask_flattened] * src[p1[1][mask_flattened], [p1[0][mask_flattened]]] + \
                                   a3[mask_flattened] * src[p2[1][mask_flattened], [p2[0][mask_flattened]]] + \
                                   a2[mask_flattened] * src[p3[1][mask_flattened], [p3[0][mask_flattened]]] + \
                                   a1[mask_flattened] * src[p4[1][mask_flattened], [p4[0][mask_flattened]]]
    # Fill invalid coordinates.
    if rgb:
        dest[dy[~mask], dx[~mask]] = [0, 0, 0, 0]
    else:
        dest[dy[~mask], dx[~mask]] = [0, 0]

    blur = cv2.GaussianBlur(np.asarray(dest, dtype=np.float32) / 255.0, (11, 11), 0)
    high = np.asarray((np.asarray(blur, dtype=np.float32) - np.asarray(dest, dtype=np.float32) / 255.0).__abs__(),
                      dtype=np.float32)

    return (dest, offsetX, offsetY, blur, high)

def stitch(ImageList):
    print("Stichting")
    #img1, img2, offsetX1, offsetX2, offsetY1, offsetY2
    #Create Image in size of img1 and img2

    #Get Max/Min Values
    maxX=0
    maxY=0

    #getMaxX
    for item in ImageList:
        value = item[0].shape[0] + item[2]
        if(maxY < value ):
            maxY = value

    # getMaxX
    for item in ImageList:
        value = item[0].shape[1] + item[1]
        if (maxX < value):
            maxX = value

    minY = 0
    minX = 0

    # getMinX
    for item in ImageList:
        value = item[2]
        if (minY > value):
            minY = value

    # getMinX
    for item in ImageList:
        value = item[1]
        if (minX > value):
            minX = value

    stitched = np.zeros((maxY - minY, maxX - minX, 4), dtype=np.float32)  # 4 Dimensionen, r,g,b,a
    stitchedHigh = np.zeros((maxY - minY, maxX - minX, 3), dtype=np.float32)  # 4 Dimensionen, r,g,b,a

    #print(stitched.shape)
    #stitched = np.insert(stitched,0, img1, axis=1)

    # bsp mit Y: der oberste Wert ist 0. Wenn ein Bild im negativen bereich ist muss trotzdem 0 raus kommen
    # d.h. offsetY-minY
    stitched[:, :, 3] = 0.0000001
    stitchedHigh[:, :, :] = 0
    for (img, offX, offY, blur, high) in ImageList:
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 0] += blur[:, :, 0] * img[:, :, 3]
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 1] += blur[:, :, 1] * img[:, :, 3]
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 2] += blur[:, :, 2] * img[:, :, 3]
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 3] += blur[:, :, 3]

    for (img, offX, offY, blur, high) in ImageList:
        stitchedHigh[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 0] = np.maximum(stitchedHigh[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 0], high[:, :, 0])
        stitchedHigh[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 1] = np.maximum(stitchedHigh[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 1], high[:, :, 1])
        stitchedHigh[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 2] = np.maximum(stitchedHigh[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 2], high[:, :, 2])


    stitched[:, :, 0] = stitched[:, :, 0] / (stitched[:, :, 3] / 255.0)
    stitched[:, :, 1] = stitched[:, :, 1] / (stitched[:, :, 3] / 255.0)
    stitched[:, :, 2] = stitched[:, :, 2] / (stitched[:, :, 3] / 255.0)


    stitched[:, :, 0:3] += stitchedHigh
    del stitchedHigh
    stitched[:, :, 3] = 255.0
    stitched[:, :, 0:3] = stitched[:, :, 0:3] / (np.max(stitched) / 255.0)


    sti = np.asarray(stitched, dtype=np.uint8)

    return sti

def main():
    print("Aufgabe 4")
    Handpunkte = True
    bp4= []
    bp5= []
    bp6= []

    img4 = loadIMGasNPbw("neue\IMG_20170622_110401.jpg")
    img5 = loadIMGasNPbw("neue\IMG_20170622_110407.jpg")

    img4rgb = loadIMGasNP("neue\IMG_20170622_110401.jpg")
    img5rgb = loadIMGasNP("neue\IMG_20170622_110407.jpg")


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
        print("Handpunkte")
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

        homographie = ransac(2000, matchPtsA,matchPtsB, 70, 0)

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
        ImageList = []
        ImageList.append(transformation(np.matrix([1,0,0,0,1,0,0,0]).T, img4rgb, 'nn', True))
        ImageList.append(transformation(homographie, img5rgb, 'nn', True))

        stitched = stitch(ImageList)

        plt.imshow(stitched)
        plt.show()
    else:
        bp4.append((random.randrange(1400,4300),random.randrange(660,1600)))
        bp4.append((random.randrange(1400,4300), random.randrange(660,1600)))
        #bp4.append((random.randrange(2000), random.randrange(2000)))

        bp5.append((random.randrange(1400,4300), random.randrange(2000)))
        bp5.append((random.randrange(1400,4300), random.randrange(2000)))
        #bp5.append((random.randrange(2000), random.randrange(2000)))

        homographie = ransac(5000, bp4, bp5, 100, 0)

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

        ImageList = []
        ImageList.append(transformation([1,0,0,0,1,0,0,0], img4rgb, 'nn', True))
        ImageList.append(transformation(homographie, img5rgb, 'nn', True))

        stitched = stitch(ImageList)


        plt.imshow(stitched)
        plt.show()

if __name__ == "__main__":
    main()