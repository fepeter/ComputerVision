import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
import math
import cv2


def rotate_coords( x, y, A, ox, oy):
    # rotiere x und y um den Mittelpunkt

    x, y = np.asarray(x) - ox, np.asarray(y) - oy

    l = np.matrix(np.hstack((np.ndarray.flatten(x), np.ndarray.flatten(y))))

    b = np.dot(np.linalg.pinv(A).T, l)

    b0 = b[0] + ox
    b1 = b[1] + oy

    return np.reshape(b0, x.shape), np.reshape(b1, y.shape)

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
    corners = np.dot(A, np.matrix(np.hstack((x, y))))
    cx = corners[0] + ox
    cy = corners[1] + oy

    # Größe des neuen Bildes
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

    dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))

    sx, sy = rotate_coords(dx + cx.min(), dy + cy.min(), A, ox, oy)

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

    print(mask)
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

    print(M, vx, a)
    return (M, vx, a)

def main():
    print("Aufgabe 3")
    wp =[]
    bp = []



    # P1
    bx1 = 312
    by1 = 432
    ox1 = 312
    oy1 = 432
    # P2
    bx2 = 343
    by2 = 423
    ox2 = 343
    oy2 = 432
    # P3
    bx3 = 345
    by3 = 337
    ox3 = 312
    oy3 = 337
    # P4
    bx4 = 363
    by4 = 337
    ox4 = 343
    oy4 = 337

    bp.append((bx1, by1))
    bp.append((bx2, by2))
    bp.append((bx3, by3))
    bp.append((bx4, by4))

    wp.append((ox1, oy1))
    wp.append((ox2, oy2))
    wp.append((ox3, oy3))
    wp.append((ox4, oy4))

    (_, _, a) = buildMat(wp, bp)
    RGBimg = scipy.misc.imread(name="schraegbild_tempelhof.jpg")

    newImg = f_affin_transformation(a, 0, RGBimg, 'nn', True)

    plt.imshow(newImg)
    plt.show()


if __name__ == "__main__":
    main()