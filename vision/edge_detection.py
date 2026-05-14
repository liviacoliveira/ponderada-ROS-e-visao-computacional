"""
edge_detection.py — Algoritmo de Canny implementado do zero com NumPy.

O algoritmo de Canny (1986) é composto por 4 etapas:
  1. Cálculo de gradientes com filtros de Sobel
  2. Supressão de não-máximos (Non-Maximum Suppression)
  3. Limiarização dupla (Double Thresholding)
  4. Rastreamento de bordas por histerese (Hysteresis Edge Tracking)

Nenhuma função de OpenCV, Pillow, scikit-image ou scipy é utilizada.
Apenas NumPy para operações matriciais.
"""

import numpy as np
from vision.preprocessing import convolve2d


# ---------------------------------------------------------------------------
# Kernels de Sobel para cálculo de gradiente
# ---------------------------------------------------------------------------

# Detecta variações horizontais (borda vertical)
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

# Detecta variações verticais (borda horizontal)
SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Etapa 1: Cálculo do gradiente com Sobel
# ---------------------------------------------------------------------------

def compute_gradients(image: np.ndarray):
    """
    Calcula magnitude e direção do gradiente usando filtros de Sobel.

    O filtro de Sobel aproxima a derivada parcial da intensidade em cada
    direção. Pixels onde a intensidade muda abruptamente (bordas) terão
    magnitude alta.

    Args:
        image: ndarray (H, W) float64 suavizado

    Returns:
        magnitude: ndarray (H, W) float64 — força da borda em cada pixel
        angle:     ndarray (H, W) float64 — direção do gradiente em graus [0, 180)
    """
    img = image.astype(np.float64)

    gx = convolve2d(img, SOBEL_X)  # Gradiente horizontal
    gy = convolve2d(img, SOBEL_Y)  # Gradiente vertical

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Direção em graus, mapeada para [0, 180) — simetria para NMS
    angle = np.degrees(np.arctan2(gy, gx)) % 180

    return magnitude, angle


# ---------------------------------------------------------------------------
# Etapa 2: Supressão de não-máximos (Non-Maximum Suppression)
# ---------------------------------------------------------------------------

def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Afina as bordas mantendo apenas os máximos locais na direção do gradiente.

    Para cada pixel, comparamos sua magnitude com os dois vizinhos na direção
    perpendicular à borda (= direção do gradiente). Se não for máximo local,
    o pixel é suprimido (zerado). Isso garante bordas de 1 pixel de largura.

    Os ângulos são quantizados em 4 direções:
        0°   → horizontal (vizinhos: esq/dir)
        45°  → diagonal descendente (vizinhos: sup-dir/inf-esq)
        90°  → vertical (vizinhos: sup/inf)
        135° → diagonal ascendente (vizinhos: sup-esq/inf-dir)

    Args:
        magnitude: ndarray (H, W) float64
        angle:     ndarray (H, W) float64 em graus [0, 180)

    Returns:
        ndarray (H, W) float64 com bordas afinadas
    """
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Quantizar ângulos em 4 direções (0, 45, 90, 135)
    quantized = np.zeros_like(angle)
    quantized[(angle >= 0)   & (angle < 22.5)]  = 0
    quantized[(angle >= 22.5) & (angle < 67.5)]  = 45
    quantized[(angle >= 67.5) & (angle < 112.5)] = 90
    quantized[(angle >= 112.5) & (angle < 157.5)] = 135
    quantized[(angle >= 157.5) & (angle <= 180)]  = 0

    # Percorrer pixels internos (bordas da imagem descartadas)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            m = magnitude[i, j]
            d = quantized[i, j]

            if d == 0:
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif d == 45:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif d == 90:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:  # 135
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if m >= n1 and m >= n2:
                suppressed[i, j] = m

    return suppressed


def non_maximum_suppression_fast(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Versão vetorizada (sem loops Python) da supressão de não-máximos.

    Usa slicing NumPy para comparar todos os pixels simultaneamente com seus
    vizinhos nas 4 direções, mantendo apenas os máximos locais.
    Muito mais rápida que a versão com loops para imagens grandes.

    Args:
        magnitude: ndarray (H, W) float64
        angle:     ndarray (H, W) float64 em graus [0, 180)

    Returns:
        ndarray (H, W) float64 com bordas afinadas
    """
    h, w = magnitude.shape
    suppressed = magnitude.copy()

    # Quantizar ângulos
    quantized = np.zeros_like(angle, dtype=np.int32)
    quantized[(angle >= 22.5)  & (angle < 67.5)]  = 1   # 45°
    quantized[(angle >= 67.5)  & (angle < 112.5)] = 2   # 90°
    quantized[(angle >= 112.5) & (angle < 157.5)] = 3   # 135°

    # Máscara de supressão (True = suprimir pixel)
    suppress_mask = np.zeros((h, w), dtype=bool)

    # Direção 0° — comparar com vizinhos esq/dir
    m0 = (quantized == 0) | (quantized == 0)
    d0 = quantized == 0
    suppress_mask[1:-1, 1:-1] |= (
        d0[1:-1, 1:-1] &
        ((suppressed[1:-1, 1:-1] < suppressed[1:-1, :-2]) |
         (suppressed[1:-1, 1:-1] < suppressed[1:-1, 2:]))
    )

    # Direção 45° — comparar com vizinhos sup-dir / inf-esq
    d45 = quantized == 1
    suppress_mask[1:-1, 1:-1] |= (
        d45[1:-1, 1:-1] &
        ((suppressed[1:-1, 1:-1] < suppressed[:-2, 2:]) |
         (suppressed[1:-1, 1:-1] < suppressed[2:, :-2]))
    )

    # Direção 90° — comparar com vizinhos sup/inf
    d90 = quantized == 2
    suppress_mask[1:-1, 1:-1] |= (
        d90[1:-1, 1:-1] &
        ((suppressed[1:-1, 1:-1] < suppressed[:-2, 1:-1]) |
         (suppressed[1:-1, 1:-1] < suppressed[2:, 1:-1]))
    )

    # Direção 135° — comparar com vizinhos sup-esq / inf-dir
    d135 = quantized == 3
    suppress_mask[1:-1, 1:-1] |= (
        d135[1:-1, 1:-1] &
        ((suppressed[1:-1, 1:-1] < suppressed[:-2, :-2]) |
         (suppressed[1:-1, 1:-1] < suppressed[2:, 2:]))
    )

    suppressed[suppress_mask] = 0
    # Zerar bordas da imagem
    suppressed[0, :] = suppressed[-1, :] = 0
    suppressed[:, 0] = suppressed[:, -1] = 0

    return suppressed


# ---------------------------------------------------------------------------
# Etapa 3: Limiarização dupla (Double Thresholding)
# ---------------------------------------------------------------------------

STRONG = 255
WEAK   = 128

def double_threshold(suppressed: np.ndarray, low_ratio: float = 0.05,
                     high_ratio: float = 0.15):
    """
    Classifica cada pixel em: forte (borda real), fraco (possível borda) ou
    suprimido (não é borda).

    Os limiares são definidos como frações do valor máximo de magnitude,
    tornando o algoritmo adaptativo ao conteúdo de cada imagem.

    - low_ratio: fração do máximo abaixo da qual o pixel é descartado
    - high_ratio: fração do máximo acima da qual o pixel é borda forte

    Args:
        suppressed: ndarray (H, W) float64 após NMS
        low_ratio:  limiar baixo relativo ao máximo
        high_ratio: limiar alto relativo ao máximo

    Returns:
        result:     ndarray (H, W) uint8 com valores STRONG/WEAK/0
        strong_mask: máscara booleana dos pixels fortes
        weak_mask:   máscara booleana dos pixels fracos
    """
    max_val = suppressed.max()
    if max_val == 0:
        return np.zeros_like(suppressed, dtype=np.uint8), np.zeros_like(suppressed, dtype=bool), np.zeros_like(suppressed, dtype=bool)

    high = max_val * high_ratio
    low  = max_val * low_ratio

    result = np.zeros(suppressed.shape, dtype=np.uint8)
    strong_mask = suppressed >= high
    weak_mask   = (suppressed >= low) & (suppressed < high)

    result[strong_mask] = STRONG
    result[weak_mask]   = WEAK

    return result, strong_mask, weak_mask


# ---------------------------------------------------------------------------
# Etapa 4: Rastreamento por histerese (Hysteresis Edge Tracking)
# ---------------------------------------------------------------------------

def hysteresis(result: np.ndarray, strong_mask: np.ndarray,
               weak_mask: np.ndarray) -> np.ndarray:
    """
    Promove pixels fracos conectados a pixels fortes para bordas reais.
    Pixels fracos isolados são descartados como ruído.

    O rastreamento usa BFS (Breadth-First Search): partindo de cada pixel
    forte, percorre todos os vizinhos fracos conectados e os promove.
    Isso garante que bordas contínuas não sejam quebradas por pequenas
    variações de intensidade.

    Args:
        result:      ndarray (H, W) uint8 com STRONG/WEAK/0
        strong_mask: máscara booleana dos pixels fortes
        weak_mask:   máscara booleana dos pixels fracos

    Returns:
        ndarray (H, W) uint8 com 255 nas bordas finais e 0 no restante
    """
    from collections import deque

    h, w = result.shape
    output = np.where(strong_mask, 255, 0).astype(np.uint8)

    # BFS a partir de cada pixel forte
    queue = deque(zip(*np.where(strong_mask)))
    visited = strong_mask.copy()

    # 8-conectividade: todos os 8 vizinhos
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if weak_mask[nr, nc]:
                    output[nr, nc] = 255
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return output


# ---------------------------------------------------------------------------
# Pipeline Canny completa
# ---------------------------------------------------------------------------

def canny(image: np.ndarray, low_ratio: float = 0.05,
          high_ratio: float = 0.15) -> np.ndarray:
    """
    Executa o algoritmo de Canny completo sobre uma imagem grayscale.

    O Canny foi escolhido por ser o detector de bordas mais robusto da
    literatura clássica. Comparado ao Sobel simples, produz bordas finas
    (1 pixel de largura), rejeita ruído de forma adaptativa via histerese
    e preserva a continuidade dos contornos — qualidades essenciais para
    gerar um caminho coerente para a tartaruga.

    Args:
        image:      ndarray (H, W) uint8 — imagem suavizada com gaussiana
        low_ratio:  limiar baixo relativo (padrão 5% do máximo)
        high_ratio: limiar alto relativo (padrão 15% do máximo)

    Returns:
        ndarray (H, W) uint8 — mapa binário de bordas (255=borda, 0=fundo)
    """
    # 1. Gradientes
    magnitude, angle = compute_gradients(image.astype(np.float64))

    # 2. Supressão de não-máximos (versão vetorizada)
    suppressed = non_maximum_suppression_fast(magnitude, angle)

    # 3. Limiarização dupla
    thresholded, strong_mask, weak_mask = double_threshold(
        suppressed, low_ratio, high_ratio
    )

    # 4. Histerese
    edges = hysteresis(thresholded, strong_mask, weak_mask)

    return edges
