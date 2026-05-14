"""
loader.py — Carregamento de imagem com OpenCV (único uso permitido do OpenCV).

OpenCV é usado APENAS aqui para ler o arquivo de imagem do disco.
Todo o processamento posterior é feito com NumPy puro.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Carrega uma imagem do disco usando cv2.imread (único uso de OpenCV permitido).

    Retorna:
        ndarray de shape (H, W, 3) em formato BGR uint8.

    Raises:
        FileNotFoundError se o arquivo não existir.
        ValueError se a imagem não puder ser lida.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")
    return img
