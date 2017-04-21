import numpy as np
import scipy.misc
import scipy
import matplotlib.pyplot as plt
import math


def main():
    #Übung 2
    print("Übung 2")
    img = scipy.misc.imread(name="schraegbild_tempelhof.jpg", flatten=True)

    plt.gray()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()