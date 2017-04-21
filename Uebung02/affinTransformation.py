import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def main():
    img = read_image('/home/dennis/ComputerVision/Aufgabe1/gletscher.jpg', True)

    theta = np.radians(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale = np.array([[0.7, 0], [0, 0.7]])
    sc = np.array([[0.8, 0], [0, 1.2]])
    stretch = np.array([[1.5, 0.5], [0.5, 1.5]])

    A = np.dot(np.dot(np.dot(rot, scale), sc), stretch)

    img = f_affin_transformation(A, 0, img, 'bil', rgb=False)
    plot_image(img)



    img = read_image('/home/dennis/ComputerVision/Aufgabe1/ambassadors.jpg', False)

    sc = np.array([[0.8, 0], [0, 2]])
    stretch = np.array([[1.6, 0.01], [0.01, 1.6]])

    theta = np.radians(25)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    A = np.dot(np.dot(sc,  stretch), rot)

    img = f_affin_transformation(A, 0, img, 'bil', rgb=True)
    plot_image(img)


def f_affin_transformation(A, a0, src, method, rgb):

    # Bildmittelpunkt
    ox = src.shape[1]//2
    oy = src.shape[0]//2

    if rgb:
        sh, sw, sd = src.shape
    else:
        sh, sw = src.shape

    # Eckpunkte des transformierten Bildes berechnen
    x = np.array([0, sw, sw, 0]) - ox
    y = np.array([0, 0, sh, sh]) - oy
    corners = np.dot(A, np.vstack((x, y)))
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

        a1 = np.prod(np.abs(p1 - v), axis = 0)
        a2 = np.prod(np.abs(p2 - v), axis = 0)
        a3 = np.prod(np.abs(p3 - v), axis = 0)
        a4 = np.prod(np.abs(p4 - v), axis = 0)


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

        dest[dy[mask], dx[mask]] = a4[mask_flattened] * src[p1[1][mask_flattened], [p1[0][mask_flattened]]] +\
                                   a3[mask_flattened] * src[p2[1][mask_flattened], [p2[0][mask_flattened]]] + \
                                   a2[mask_flattened] * src[p3[1][mask_flattened], [p3[0][mask_flattened]]] +\
                                   a1[mask_flattened] * src[p4[1][mask_flattened], [p4[0][mask_flattened]]]
    # Fill invalid coordinates.
    if rgb:
        dest[dy[~mask], dx[~mask]] = [0, 0, 0]
    else:
        dest[dy[~mask], dx[~mask]] = 0

    return dest


def rotate_coords(x, y, A, ox, oy):
    # rotiere x und y um den Mittelpunkt

    x, y = np.asarray(x) - ox, np.asarray(y) - oy

    l = np.vstack((np.ndarray.flatten(x), np.ndarray.flatten(y)))

    b = np.dot(np.linalg.inv(A), l)

    b0 = b[0] + ox
    b1 = b[1] + oy

    return np.reshape(b0, x.shape), np.reshape(b1, y.shape)


def read_image(path, grey):
    if grey:
        img = io.imread(path, as_grey=True)
    else:
        img = io.imread(path)
    return img

def plot_image(img):
    plt.figure()
    if len(img.shape) == 2:
        plt.imshow(img, cmap='Greys_r')
    else:
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
