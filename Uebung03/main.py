import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
from time import *
from operator import itemgetter
import math
import cv2
from skimage.feature.tests.test_orb import img
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

#plt.gray()
#plt.imshow(distance)
#plt.show()

print("Calculatin Distance done in ", t2-t1)

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
    stitched[:, :, 3] = 1
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
    stitched[:, :, 3] = 255
    stitched[:, :, 0:3] = stitched[:, :, 0:3] / (np.max(stitched) / 255)


    sti = np.asarray(stitched, dtype=np.uint8)

    return sti

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
        dest[dy[~mask], dx[~mask]] = [0, 0, 0, 0]
    else:
        dest[dy[~mask], dx[~mask]] = [0, 0]

    blur = cv2.GaussianBlur(np.asarray(dest, dtype=np.float32) / 255.0, (11, 11), 0)
    high = np.asarray((np.asarray(blur, dtype=np.float32) - np.asarray(dest, dtype=np.float32) / 255.0).__abs__(),
                      dtype=np.float32)

    return (dest, offsetX, offsetY, blur, high)

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



def main():
    print("Aufgabe 3")
    wp =[]

    bp2 = []
    bp1 = []
    bp3 = []

    bp4 = []
    bp5 = []
    bp6 = []
    bp7 = []

    wpDiv = 6
    wp.append((0, 0))
    wp.append((2950/wpDiv, 50/wpDiv))
    wp.append((2970/wpDiv, 3900/wpDiv))
    wp.append((40/wpDiv, 4290/wpDiv))

    img1 = loadIMGasNP("imgs\IMG_20170504_131710_001.jpg")
    img2 = loadIMGasNP("imgs\IMG_20170504_131710_010.jpg")
    img3 = loadIMGasNP("imgs\IMG_20170504_131710_020.jpg")

    bp1.append((2696, 646))
    bp1.append((4140, 634))
    bp1.append((3931, 2340))
    bp1.append((2678, 2481))

    bp2.append((2031, 727))
    bp2.append((3427, 743))
    bp2.append((3285, 2427))
    bp2.append((2058, 2558))

    bp3.append((1506, 765))
    bp3.append((2894, 786))
    bp3.append((2800, 2447))
    bp3.append((1578, 2598))

    wp2 = []
    wpDiv2 = 8
    wp2.append((0, 0))
    wp2.append((2950 / wpDiv2, 0 / wpDiv2))
    wp2.append((2910 / wpDiv2, 3000 / wpDiv2))
    wp2.append((40 / wpDiv2, 3000 / wpDiv2))


    img4 = loadIMGasNP("neue\IMG_20170622_110401.jpg")
    img5 = loadIMGasNP("neue\IMG_20170622_110407.jpg")
    img6 = loadIMGasNP("neue\IMG_20170622_110415.jpg")


    bp4.append((2945,862))
    bp4.append((4358,662))
    bp4.append((4309,1581))
    bp4.append((3050,1559))

    bp5.append((1641,914))
    bp5.append((2863,815))
    bp5.append((2819,1615))
    bp5.append((1737,1606))

    bp6.append((263,873))
    bp6.append((1638,824))
    bp6.append((1598,1628))
    bp6.append((385,1648))





    #plt.gray()
    #plt.imshow(np.asarray(img5, dtype=np.uint8))
    # x, y = zip(*bp4)
    # plt.scatter(x, y, c='r', s=20)
    #plt.show()

    #plt.gray()
    #plt.imshow(np.asarray(img6, dtype=np.uint8))
    # x, y = zip(*bp4)
    # plt.scatter(x, y, c='r', s=20)
    #plt.show()

    if(False):


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

        del(stitched)
        del(img1)
        del(img2, img3)

    (M4, _, a4) = buildMat(wp2, bp4)
    (M5, _, a5) = buildMat(wp2, bp5)
    (M6, _, a6) = buildMat(wp2, bp6)


    NewImageList = []
    NewImageList.append(transformation(a4, 0, img4, 'nn', True))
    NewImageList.append(transformation(a5, 0, img5, 'nn', True))
    #NewImageList.append(transformation(a6, 0, img6, 'nn', True))
    #NewImageList.append(transformation(a7, 0, img7, 'nn', True))


    #for item in NewImageList:
    #    plt.imshow(np.asarray(item[0], dtype=np.uint8))
    #    plt.show()



    stichted2 = stitch(NewImageList)

    fertig = stichted2[0:3000, 5000:]

    plt.gray()
    plt.imshow(fertig)
    plt.show()

if __name__ == "__main__":
    main()