import matplotlib.pyplot as plt


def show_depth(depth):
    """
    Affiche la carte de profondeur.

    Args:
        depth (numpy.ndarray): Carte de profondeur à afficher.
    """
    plt.imshow(depth, cmap='plasma')
    plt.colorbar()
    plt.show()
