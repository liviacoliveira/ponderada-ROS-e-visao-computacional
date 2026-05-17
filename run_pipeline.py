"""
run_pipeline.py — Script de execução e teste da pipeline de visão computacional.

Execute com:
    python3 run_pipeline.py

Ou com argumentos:
    python3 run_pipeline.py --image image/bulldog.jpg --size 300 --save
"""

import argparse
import sys
import os
import time

import numpy as np

# Adicionar raiz do projeto ao path para importar vision.*
sys.path.insert(0, os.path.dirname(__file__))

from vision.loader import load_image
from vision.preprocessing import bgr_to_gray, resize, gaussian_blur
from vision.edge_detection import canny
from vision.path_planning import plan_path
from vision.visualize import show_pipeline, show_path


def run_pipeline(image_path: str,
                 max_size: int = 256,
                 gaussian_size: int = 5,
                 gaussian_sigma: float = 1.4,
                 canny_low: float = 0.05,
                 canny_high: float = 0.15,
                 max_segments: int = 20,
                 min_segment: int = 15,
                 sample_step: int = 3,
                 save: bool = False):
    """
    Executa a pipeline completa e retorna os segmentos para o turtlesim.
    """
    print("=" * 60)
    print("  TURTLE DRAW — Pipeline de Visão Computacional")
    print("=" * 60)

    # ---- Etapa 1: Carregar ------------------------------------------------
    print(f"\n[1/5] Carregando imagem: {image_path}")
    t0 = time.time()
    original = load_image(image_path)
    print(f"      Shape: {original.shape}  ({time.time()-t0:.2f}s)")

    # ---- Etapa 2: Pré-processamento ----------------------------------------
    print(f"\n[2/5] Pré-processamento")

    print(f"      → Redimensionando para max_size={max_size}px...")
    small = resize(original, max_size=max_size)
    print(f"        Shape após resize: {small.shape[:2]}")

    print(f"      → Convertendo para escala de cinza (luminância ITU-R BT.601)...")
    gray = bgr_to_gray(small)

    print(f"      → Aplicando filtro gaussiano ({gaussian_size}×{gaussian_size}, σ={gaussian_sigma})...")
    t0 = time.time()
    blurred = gaussian_blur(gray, size=gaussian_size, sigma=gaussian_sigma)
    print(f"        Concluído em {time.time()-t0:.2f}s")

    # ---- Etapa 3: Detecção de bordas ---------------------------------------
    print(f"\n[3/5] Detecção de bordas — Canny")
    print(f"      → low_ratio={canny_low}, high_ratio={canny_high}")
    t0 = time.time()
    edges = canny(blurred, low_ratio=canny_low, high_ratio=canny_high)
    n_edge_pixels = int(np.sum(edges > 0))
    print(f"      → {n_edge_pixels} pixels de borda detectados  ({time.time()-t0:.2f}s)")

    # ---- Etapa 4: Planejamento de caminho ----------------------------------
    print(f"\n[4/5] Planejamento de caminho")
    print(f"      → max_segments={max_segments}, min_segment={min_segment}, sample_step={sample_step}")
    t0 = time.time()
    segments = plan_path(edges,
                         max_segments=max_segments,
                         min_segment_len=min_segment,
                         sample_step=sample_step)
    total_pts = sum(len(s) for s in segments)
    print(f"      → {len(segments)} segmentos, {total_pts} pontos total  ({time.time()-t0:.2f}s)")

    # ---- Etapa 5: Visualização --------------------------------------------
    print(f"\n[5/5] Visualização")
    save_path = "docs/pipeline_result.png" if save else None
    if save:
        os.makedirs("docs", exist_ok=True)
    show_pipeline(small, gray, blurred, edges, segments, save_path=save_path)

    # Exibir apenas o caminho no turtlesim
    path_save = "docs/turtlesim_path.png" if save else None
    show_path(segments, save_path=path_save)

    print("\n[OK] Pipeline concluída!")
    print(f"     Segmentos prontos para o turtlesim: {len(segments)}")
    print(f"     Total de pontos a percorrer: {total_pts}")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de visão computacional para o Turtle Draw"
    )
    parser.add_argument("--image",       default="image/bulldog.jpg",
                        help="Caminho da imagem de entrada")
    parser.add_argument("--size",        type=int,   default=256,
                        help="Tamanho máximo após redimensionamento (padrão: 256)")
    parser.add_argument("--gauss-size",  type=int,   default=5,
                        help="Tamanho do kernel gaussiano (padrão: 5)")
    parser.add_argument("--gauss-sigma", type=float, default=1.4,
                        help="Sigma do filtro gaussiano (padrão: 1.4)")
    parser.add_argument("--canny-low",   type=float, default=0.02,
                        help="Limiar baixo do Canny, relativo ao máximo (padrão: 0.02)")
    parser.add_argument("--canny-high",  type=float, default=0.08,
                        help="Limiar alto do Canny, relativo ao máximo (padrão: 0.08)")
    parser.add_argument("--max-seg",     type=int,   default=200,
                        help="Número máximo de segmentos (padrão: 200)")
    parser.add_argument("--min-seg",     type=int,   default=5,
                        help="Tamanho mínimo de segmento em pixels (padrão: 5)")
    parser.add_argument("--step",        type=int,   default=3,
                        help="Amostragem de pontos: pegar 1 a cada N (padrão: 3)")
    parser.add_argument("--save",        action="store_true",
                        help="Salvar figuras em docs/")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        max_size=args.size,
        gaussian_size=args.gauss_size,
        gaussian_sigma=args.gauss_sigma,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        max_segments=args.max_seg,
        min_segment=args.min_seg,
        sample_step=args.step,
        save=args.save,
    )
