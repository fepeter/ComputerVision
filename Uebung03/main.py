import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
from time import *
from operator import itemgetter
import math
#import cv2
from skimage.feature.tests.test_orb import img


def stitch(ImageList):
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

    stitched = np.zeros((maxY - minY, maxX - minX, 4), dtype=np.uint8) # 4 Dimensionen, r,g,b,a

    #print(stitched.shape)
    #stitched = np.insert(stitched,0, img1, axis=1)

    # bsp mit Y: der oberste Wert ist 0. Wenn ein Bild im negativen bereich ist muss trotzdem 0 raus kommen
    # d.h. offsetY-minY

    for (img, offX, offY) in ImageList:
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 0] = img[:, :, 0] * img[:, :, 3]
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 1] = img[:, :, 1] * img[:, :, 3]
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 2] = img[:, :, 2] * img[:, :, 3]
        stitched[offY - minY: img.shape[0] + offY - minY, offX - minX: img.shape[1] + offX - minX, 3] += img[:,:,3]

        stitched[:, :, 0] = stitched[:, :, 0] / (stitched[:, :, 3] / 255.0)
        stitched[:, :, 1] = stitched[:, :, 1] / (stitched[:, :, 3] / 255.0)
        stitched[:, :, 2] = stitched[:, :, 2] / (stitched[:, :, 3] / 255.0)
        stitched[:, :, 3] = 255
    return stitched

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

def transformation(A, a0, src, method, rgb):
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

    # Größe des neuen Bildes
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))
    offsetX, offsetY = (int(np.ceil(c.min())) for c in (cx, cy))
    print(offsetX, offsetY)
    #print(getBcord(A, 0 + cx.min(), 0+ cy.min()))
    #print(getBcord(A, dw + cx.min(), 0+ cy.min()))
    #print(getBcord(A, dw + cx.min(), dh+ cy.min()))
    #print(getBcord(A, 0 + cx.min(), dh+ cy.min()))
    #print("DH:", dh, "DW", dw)

    # distanz von der Bildmitte entspricht gewichtung (eigentlich Alpha Kanal.)
    distance = np.ones((sh, sw)) * 255
    dx, dy = np.meshgrid(np.arange(sh), np.arange(sw))



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
            dest[dy[mask], dx[mask], 0] = src[sy[mask], sx[mask],:]
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
        dest[dy[~mask], dx[~mask]] = [0, 0, 0, 255]
    else:
        dest[dy[~mask], dx[~mask]] = [0, 255]

    return (dest, offsetX, offsetY)

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

def main():
    print("Aufgabe 3")
    wp =[]

    bp2 = []
    bp1 = []
    bp3 = []

    img1 = scipy.misc.imread(name="imgs\IMG_20170504_131710_001.jpg")
    img2 = scipy.misc.imread(name="imgs\IMG_20170504_131710_010.jpg")
    img3 = scipy.misc.imread(name="imgs\IMG_20170504_131710_020.jpg")



    bp1.append((2696, 646))
    bp1.append((4140, 634))
    bp1.append((3931, 2340))
    bp1.append((2678, 2481))

    wpDiv = 6
    wp.append((0, 0))
    wp.append((2950/wpDiv, 50/wpDiv))
    wp.append((2970/wpDiv, 3900/wpDiv))
    wp.append((40/wpDiv, 4290/wpDiv))

    bp2.append((2031, 727))
    bp2.append((3427, 743))
    bp2.append((3285, 2427))
    bp2.append((2058, 2558))

    bp3.append((1506, 765))
    bp3.append((2894, 786))
    bp3.append((2800, 2447))
    bp3.append((1578, 2598))

    #plt.imshow(img1)
    #x, y = zip(*bp1)
    #plt.scatter(x, y, c='r', s=20)
    #plt.show()


    t1 = clock()
    (M1, _, a1) = buildMat(wp, bp1)
    t2 = clock()

    print("Build Mat in ", t2-t1)

    #print(a1)
    (M2, _, a2) = buildMat(wp, bp2)
    (M3, _, a3) = buildMat(wp, bp3)

    ImageList = []
    t1=clock()
    ImageList.append( transformation(a1, 0, img1, 'nn', True))
    t2 = clock()
    print("Transform IMG1 in ", t2 - t1)

    t1 = clock()
    ImageList.append( transformation(a2, 0, img2, 'nn', True))
    t2 = clock()
    print("Transform IMG2 in ", t2 - t1)

    t1 = clock()
    ImageList.append(transformation(a3, 0, img3, 'nn', True))
    t2 = clock()
    print("Transform IMG3 in ", t2 - t1)






    t1 = clock()
    #stitched = stitch(newImg1, newImg2, new, offsetX1, offsetX2, offsetY1, offsetY2)
    stitched = stitch(ImageList)
    t2 = clock()
    print("Stitch IMGs in ", t2 - t1)

    plt.gray()
    plt.imshow(stitched)
    plt.show()


if __name__ == "__main__":
    main()