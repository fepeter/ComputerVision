import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
import math
import cv2


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

def f_affin_transformation(A, a0, src, method, rgb):
    # Bildmittelpunkt
    ox = src.shape[1] // 2
    oy = src.shape[0] // 2

    if rgb:
        sh, sw, sd = src.shape
    else:
        sh, sw = src.shape

    # Eckpunkte des transformierten Bildes berechnen
    x = np.array([0, sw, sw, 0]) - ox
    y = np.array([0, 0, sh, sh]) - oy
    print(x,y)
    #corners = np.dot(A, np.matrix(np.hstack((x, y))))
    corners = getBcord

    cx = corners[0] + ox
    cy = corners[1] + oy
    print(cx, cy)
    # Größe des neuen Bildes
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

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
        dest = np.empty(shape=(dh, dw, 3), dtype=src.dtype)
    else:
        dest = np.empty(shape=(dh, dw), dtype=src.dtype)
    if method == 'nn':
        dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
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
        dest[dy[~mask], dx[~mask]] = [0, 0, 0]
    else:
        dest[dy[~mask], dx[~mask]] = 0

    return dest

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
    img1 = scipy.misc.imread(name="imgs\IMG_20170504_131710_001.jpg")
    img2 = scipy.misc.imread(name="imgs\IMG_20170504_131710_020.jpg")


    bp1.append((2696, 646))
    bp1.append((4140, 634))
    bp1.append((3931, 2340))
    bp1.append((2678, 2481))

    wp.append((300, 0))
    wp.append((2950, 50))
    wp.append((2970, 3900))
    wp.append((40, 4290))

    bp2.append((1506, 765))
    bp2.append((2894, 786))
    bp2.append((2800, 2447))
    bp2.append((1578, 2598))



    (M1, _, a1) = buildMat(wp, bp1)
    print(a1)
    (M2, _, a2) = buildMat(wp, bp2)

    newImg1 = f_affin_transformation(a1, 0, img1, 'nn', True)
    newImg2 = f_affin_transformation(a2, 0, img2, 'nn', True)

    plt.imshow(newImg1)
    plt.show()

    plt.imshow(newImg2)
    plt.show()

if __name__ == "__main__":
    main()