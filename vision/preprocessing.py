"""
preprocessing.py — Pré-processamento de imagem implementado do zero com NumPy.

Etapas:
  1. Conversão BGR → Grayscale (fórmula de luminância ITU-R BT.601)
  2. Redimensionamento proporcional (para controlar complexidade)
  3. Suavização Gaussiana (kernel gerado manualmente + convolução 2D)

Nenhuma função de OpenCV, Pillow, scikit-image ou scipy é utilizada aqui.
Apenas NumPy para operações matriciais.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Conversão para escala de cinza
# ---------------------------------------------------------------------------

def bgr_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Converte imagem BGR (padrão OpenCV) para escala de cinza usando a fórmula
    de luminância percebida ITU-R BT.601:
        Y = 0.114·B + 0.587·G + 0.299·R

    Os coeficientes diferentes para cada canal refletem a sensibilidade do olho
    humano: somos mais sensíveis ao verde e menos ao azul.

    Args:
        img: ndarray (H, W, 3) BGR uint8

    Returns:
        ndarray (H, W) uint8 em escala de cinza
    """
    # Separar canais B, G, R (ordem BGR do OpenCV)
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)

    gray = 0.114 * B + 0.587 * G + 0.299 * R

    # Truncar para [0, 255] e converter para uint8
    return np.clip(gray, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Redimensionamento proporcional
# ---------------------------------------------------------------------------

def resize(img: np.ndarray, max_size: int = 256) -> np.ndarray:
    """
    Redimensiona proporcionalmente a imagem de forma que o maior lado tenha
    no máximo `max_size` pixels, usando interpolação bilinear manual.

    A redução é necessária porque o turtlesim tem resolução limitada —
    trabalhar com imagens muito grandes gera mais pontos de contorno do que
    o robô consegue percorrer de forma útil.

    Args:
        img: ndarray (H, W) ou (H, W, C)
        max_size: dimensão máxima do maior lado

    Returns:
        ndarray redimensionado
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img.copy()

    scale = max_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Coordenadas de origem para cada pixel destino
    row_idx = (np.arange(new_h) / scale).astype(np.float64)
    col_idx = (np.arange(new_w) / scale).astype(np.float64)

    # Índices inteiros e fracionários para interpolação bilinear
    r0 = np.clip(row_idx.astype(np.int64), 0, h - 2)
    c0 = np.clip(col_idx.astype(np.int64), 0, w - 2)
    r1 = r0 + 1
    c1 = c0 + 1

    dr = (row_idx - r0)[:, np.newaxis]  # (new_h, 1)
    dc = (col_idx - c0)[np.newaxis, :]  # (1, new_w)

    if img.ndim == 2:
        # Imagem grayscale
        f = img.astype(np.float64)
        out = (f[r0][:, c0] * (1 - dr) * (1 - dc)
               + f[r1][:, c0] * dr * (1 - dc)
               + f[r0][:, c1] * (1 - dr) * dc
               + f[r1][:, c1] * dr * dc)
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        channels = []
        for ch in range(img.shape[2]):
            f = img[:, :, ch].astype(np.float64)
            ch_out = (f[r0][:, c0] * (1 - dr) * (1 - dc)
                      + f[r1][:, c0] * dr * (1 - dc)
                      + f[r0][:, c1] * (1 - dr) * dc
                      + f[r1][:, c1] * dr * dc)
            channels.append(np.clip(ch_out, 0, 255).astype(np.uint8))
        return np.stack(channels, axis=2)


# ---------------------------------------------------------------------------
# 3. Filtro Gaussiano
# ---------------------------------------------------------------------------

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Gera um kernel gaussiano 2D de dimensões (size × size).

    Fórmula:
        G(x, y) = exp(-(x² + y²) / (2σ²))
    Normalizado para que a soma de todos os elementos seja 1,
    garantindo que o filtro não altere o brilho médio da imagem.

    Args:
        size:  tamanho do kernel (deve ser ímpar)
        sigma: desvio padrão da gaussiana

    Returns:
        ndarray (size, size) float64 normalizado
    """
    if size % 2 == 0:
        size += 1  # Garantir que seja ímpar

    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolução 2D manual entre uma imagem grayscale e um kernel.

    Utiliza padding de reflexão ("reflect") para evitar artefatos nas bordas:
    em vez de zeros, as bordas são espelhadas, produzindo uma continuidade
    mais natural e evitando escurecimento artificial nas margens.

    A implementação usa operações vetorizadas NumPy (sem loops Python sobre
    pixels) para manter desempenho aceitável.

    Args:
        image:  ndarray (H, W) float64
        kernel: ndarray (kH, kW) float64

    Returns:
        ndarray (H, W) float64 — imagem após convolução
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Padding de reflexão
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='reflect')

    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float64)

    # Convolução: somar produtos deslocados (operações matriciais NumPy)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i + h, j:j + w]

    return output


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Aplica suavização gaussiana a uma imagem grayscale.

    O tamanho 5×5 com σ=1.4 é a configuração clássica usada no algoritmo
    Canny original (Canny, 1986). Suaviza o ruído de alta frequência sem
    desfocar excessivamente as bordas que queremos detectar.

    Args:
        image: ndarray (H, W) uint8
        size:  tamanho do kernel gaussiano
        sigma: desvio padrão da gaussiana

    Returns:
        ndarray (H, W) uint8
    """
    kernel = gaussian_kernel(size, sigma)
    blurred = convolve2d(image.astype(np.float64), kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)
