"""
visualize.py — Visualização das etapas da pipeline com Matplotlib.

Permite inspecionar cada etapa do processamento:
  - Imagem original
  - Escala de cinza
  - Após suavização gaussiana
  - Mapa de bordas (Canny)
  - Caminho planejado para o turtlesim

Apenas Matplotlib é usado aqui (permitido pelo enunciado).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List


def show_pipeline(original_bgr: np.ndarray,
                  gray: np.ndarray,
                  blurred: np.ndarray,
                  edges: np.ndarray,
                  segments: List[np.ndarray],
                  save_path: str = None) -> None:
    """
    Exibe as 5 etapas da pipeline em uma única figura lado a lado.

    Args:
        original_bgr: imagem original em BGR (H, W, 3)
        gray:         imagem em escala de cinza (H, W)
        blurred:      imagem após suavização gaussiana (H, W)
        edges:        mapa de bordas binário (H, W)
        segments:     lista de segmentos em coordenadas turtlesim
        save_path:    se fornecido, salva a figura neste caminho
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("Pipeline de Visão Computacional — Turtle Draw", fontsize=14, fontweight='bold')

    # Etapa 0: imagem original (converter BGR -> RGB para exibição correta)
    original_rgb = original_bgr[:, :, ::-1]
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis('off')

    # Etapa 1: grayscale
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("1. Escala de Cinza\n(luminância ITU-R BT.601)", fontsize=11)
    axes[1].axis('off')

    # Etapa 2: suavização gaussiana
    axes[2].imshow(blurred, cmap='gray')
    axes[2].set_title("2. Suavização Gaussiana\n(5×5, σ=1.4)", fontsize=11)
    axes[2].axis('off')

    # Etapa 3: mapa de bordas
    axes[3].imshow(edges, cmap='gray')
    axes[3].set_title("3. Detecção de Bordas\n(Canny)", fontsize=11)
    axes[3].axis('off')

    # Etapa 4: caminho no turtlesim
    ax = axes[4]
    ax.set_xlim(0, 11.09)
    ax.set_ylim(0, 11.09)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.set_title("4. Caminho no Turtlesim", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    colors = plt.cm.tab20.colors
    total_points = 0
    for i, seg in enumerate(segments):
        color = colors[i % len(colors)]
        ax.plot(seg[:, 0], seg[:, 1], '-', color=color, linewidth=0.8, alpha=0.85)
        total_points += len(seg)

    ax.text(0.02, 0.02,
            f"{len(segments)} segmentos\n{total_points} pontos",
            transform=ax.transAxes, color='white', fontsize=8,
            verticalalignment='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualize] Figura salva em: {save_path}")

    plt.show()


def show_edges_only(edges: np.ndarray, save_path: str = None) -> None:
    """Exibe apenas o mapa de bordas."""
    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')
    plt.title("Mapa de Bordas — Canny")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def show_path(segments: List[np.ndarray], save_path: str = None) -> None:
    """Exibe o caminho planejado no espaço do turtlesim."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 11.09)
    ax.set_ylim(0, 11.09)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Caminho planejado — Turtlesim", fontsize=13, color='white', pad=12)
    ax.set_xlabel("X (turtlesim)", color='white')
    ax.set_ylabel("Y (turtlesim)", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    fig.patch.set_facecolor('#0f0f1a')

    colors = plt.cm.tab20.colors
    for i, seg in enumerate(segments):
        color = colors[i % len(colors)]
        ax.plot(seg[:, 0], seg[:, 1], '-', color=color, linewidth=1.0, alpha=0.9)
        # Marcar início e fim do segmento
        ax.plot(seg[0, 0], seg[0, 1], 'o', color=color, markersize=3)
        ax.plot(seg[-1, 0], seg[-1, 1], 's', color=color, markersize=3)

    total = sum(len(s) for s in segments)
    ax.text(0.02, 0.98,
            f"Segmentos: {len(segments)}\nPontos total: {total}",
            transform=ax.transAxes, color='white', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff20'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualize] Caminho salvo em: {save_path}")
    plt.show()
