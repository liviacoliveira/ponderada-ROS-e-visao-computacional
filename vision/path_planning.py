"""
path_planning.py — Planejamento de caminho para o turtlesim.

Etapas:
  1. Extrair coordenadas dos pixels de borda
  2. Amostragem para reduzir quantidade de pontos
  3. Segmentação em contornos conectados (componentes conexos)
  4. Ordenação de cada segmento pelo vizinho mais próximo (Nearest Neighbor)
  5. Mapeamento para o espaço de coordenadas do turtlesim (0..11.09)

Nenhuma função de OpenCV, Pillow, scikit-image ou scipy é utilizada.
Apenas NumPy para operações matriciais.
"""

import numpy as np
from typing import List, Tuple


# Dimensões do espaço turtlesim
TURTLESIM_SIZE = 11.09
MARGIN = 0.5  # margem para não sair da janela


# ---------------------------------------------------------------------------
# 1. Extrair pixels de borda
# ---------------------------------------------------------------------------

def extract_edge_points(edge_image: np.ndarray) -> np.ndarray:
    """
    Extrai as coordenadas (linha, coluna) de todos os pixels de borda.

    Args:
        edge_image: ndarray (H, W) uint8 — mapa binário de bordas (255=borda)

    Returns:
        ndarray (N, 2) com coordenadas [row, col] dos pixels de borda
    """
    points = np.argwhere(edge_image > 0)  # [[row, col], ...]
    return points


# ---------------------------------------------------------------------------
# 2. Segmentação em componentes conexos
# ---------------------------------------------------------------------------

def find_connected_components(edge_image: np.ndarray,
                               min_length: int = 10) -> List[np.ndarray]:
    """
    Encontra todos os segmentos de borda conectados usando BFS com 8-conectividade.

    Em vez de tratar todos os pontos de borda como uma nuvem global, agrupamos
    os pixels em "segmentos" — cadeias de pixels vizinhos. Isso permite que a
    tartaruga desenhe cada segmento de forma contínua e levante a caneta apenas
    entre segmentos distintos, produzindo um desenho mais fiel ao original.

    Args:
        edge_image:  ndarray (H, W) uint8
        min_length:  segmentos com menos de `min_length` pixels são descartados
                     (eliminam ruídos isolados)

    Returns:
        Lista de ndarrays, cada um com shape (K, 2) — coordenadas [row, col]
        de um componente conexo
    """
    from collections import deque

    h, w = edge_image.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    edge_mask = edge_image > 0

    for r0, c0 in zip(*np.where(edge_mask)):
        if visited[r0, c0]:
            continue

        # BFS para encontrar todos os pixels conectados
        component = []
        queue = deque([(r0, c0)])
        visited[r0, c0] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and edge_mask[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        if len(component) >= min_length:
            components.append(np.array(component))

    # Ordenar por tamanho decrescente — desenhar os maiores contornos primeiro
    components.sort(key=lambda c: len(c), reverse=True)
    return components


# ---------------------------------------------------------------------------
# 3. Ordenação por vizinho mais próximo
# ---------------------------------------------------------------------------

def nearest_neighbor_order(points: np.ndarray) -> np.ndarray:
    """
    Ordena os pontos de um segmento de borda pelo algoritmo do vizinho mais
    próximo (Nearest Neighbor / Greedy TSP).

    Ponto de início: o pixel mais próximo ao canto superior-esquerdo.
    A cada passo, move-se para o ponto não visitado mais próximo do atual.

    Embora não seja a solução ótima do Problema do Caixeiro Viajante, o NN
    é eficiente (O(n²)) e produz caminhos coerentes para contornos de borda,
    que tendem a ser localmente contínuos por natureza.

    Args:
        points: ndarray (N, 2) com coordenadas [row, col]

    Returns:
        ndarray (N, 2) com pontos reordenados
    """
    n = len(points)
    if n <= 1:
        return points

    # Ponto inicial: mais próximo ao canto superior-esquerdo (0,0)
    dists_to_origin = np.linalg.norm(points, axis=1)
    start = int(np.argmin(dists_to_origin))

    ordered = []
    remaining = list(range(n))
    current = start

    while remaining:
        ordered.append(current)
        remaining.remove(current)
        if not remaining:
            break

        # Calcular distâncias do ponto atual a todos os restantes
        pts_remaining = points[remaining]
        deltas = pts_remaining - points[current]
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2  # distância² (sem sqrt)
        nearest_idx = int(np.argmin(dists))
        current = remaining[nearest_idx]

    return points[ordered]


# ---------------------------------------------------------------------------
# 4. Mapeamento para o espaço turtlesim
# ---------------------------------------------------------------------------

def map_to_turtlesim(points: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Converte coordenadas de pixels (row, col) para coordenadas do turtlesim (x, y).

    Transformações aplicadas:
      - Escala: normaliza para o espaço útil do turtlesim (TURTLESIM_SIZE - 2*MARGIN)
      - Inversão de Y: no sistema de imagem, Y cresce para baixo; no turtlesim,
        Y cresce para cima — precisamos inverter.
      - Translação: adiciona margem para manter o desenho dentro da janela.
      - Troca de eixos: row→Y e col→X.

    Args:
        points:    ndarray (N, 2) com coordenadas [row, col] em pixels
        img_shape: (H, W) da imagem original

    Returns:
        ndarray (N, 2) com coordenadas [x, y] no espaço turtlesim
    """
    h, w = img_shape
    usable = TURTLESIM_SIZE - 2 * MARGIN
    scale = usable / max(h, w)

    rows = points[:, 0].astype(np.float64)
    cols = points[:, 1].astype(np.float64)

    # col → x (da esquerda para direita)
    x = cols * scale + MARGIN
    # row → y (invertido: row=0 → y=topo, row=H → y=baixo)
    y = (h - 1 - rows) * scale + MARGIN

    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# 5. Pipeline completa de planejamento
# ---------------------------------------------------------------------------

def plan_path(edge_image: np.ndarray,
              max_segments: int = 20,
              min_segment_len: int = 15,
              sample_step: int = 3) -> List[np.ndarray]:
    """
    Executa a pipeline completa de planejamento de caminho.

    Retorna uma lista de segmentos, onde cada segmento é um ndarray (K, 2)
    com coordenadas [x, y] no espaço turtlesim. Entre segmentos, a tartaruga
    levanta a caneta (teleport sem desenho).

    Args:
        edge_image:      ndarray (H, W) uint8 — mapa de bordas
        max_segments:    número máximo de segmentos a incluir (maiores primeiro)
        min_segment_len: tamanho mínimo de segmento (em pixels)
        sample_step:     pegar 1 ponto a cada `sample_step` pontos do segmento

    Returns:
        Lista de segmentos, cada um como ndarray (K, 2) em coords turtlesim
    """
    h, w = edge_image.shape

    # 1. Encontrar componentes conexos
    components = find_connected_components(edge_image, min_length=min_segment_len)

    # 2. Limitar ao número máximo de segmentos (maiores primeiro)
    components = components[:max_segments]

    segments_turtlesim = []
    for component in components:
        # 3. Amostrar pontos para reduzir densidade
        sampled = component[::sample_step]
        if len(sampled) < 2:
            continue

        # 4. Ordenar pelo vizinho mais próximo
        ordered = nearest_neighbor_order(sampled)

        # 5. Mapear para coordenadas turtlesim
        mapped = map_to_turtlesim(ordered, (h, w))
        segments_turtlesim.append(mapped)

    return segments_turtlesim
