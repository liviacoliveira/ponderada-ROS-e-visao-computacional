"""
draw_node.py — Nó ROS 2 que controla a tartaruga do turtlesim para
               reproduzir os contornos extraídos de uma imagem.

Arquitetura do nó:
  - Executa a pipeline de visão computacional ao inicializar
  - Usa o serviço /turtle1/set_pen para ligar/desligar o traço
  - Usa o serviço /turtle1/teleport_absolute para posicionar a tartaruga
  - Usa o serviço /reset para limpar a tela antes de desenhar
  - Desenha cada segmento de borda em sequência, levantando a caneta
    entre segmentos distintos

Escolha por teleport_absolute em vez de cmd_vel:
  O turtlesim tem dimensões limitadas e o caminho possui muitos pontos.
  O teleporte garante precisão milimétrica no posicionamento, enquanto
  cmd_vel acumularia erros de integração a cada passo. Para uma prova
  de conceito de visão computacional, a fidelidade ao contorno é mais
  importante do que simular dinâmica de movimento real.
"""

import os
import sys
import time
import rclpy
from rclpy.node import Node
from turtlesim.srv import SetPen, TeleportAbsolute
from std_srvs.srv import Empty
from typing import List
import numpy as np

# Adicionar raiz do projeto ao sys.path para importar os módulos de visão
# Encontra a pasta raiz subindo nos diretórios até achar a pasta 'vision'
_current_dir = os.path.abspath(os.path.dirname(__file__))
while _current_dir != '/' and not os.path.isdir(os.path.join(_current_dir, 'vision')):
    _current_dir = os.path.dirname(_current_dir)

_PKG_ROOT = _current_dir
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from vision.loader import load_image
from vision.preprocessing import bgr_to_gray, resize, gaussian_blur
from vision.edge_detection import canny
from vision.path_planning import plan_path


# ---------------------------------------------------------------------------
# Parâmetros configuráveis via parâmetros ROS 2
# ---------------------------------------------------------------------------
DEFAULT_IMAGE    = os.path.join(_PKG_ROOT, 'image', 'bulldog.jpg')
DEFAULT_MAX_SIZE = 256
DEFAULT_GAUSS_SZ = 5
DEFAULT_GAUSS_SG = 1.4
DEFAULT_CLOW     = 0.02
DEFAULT_CHIGH    = 0.08
DEFAULT_MAX_SEG  = 200
DEFAULT_MIN_SEG  = 5
DEFAULT_STEP     = 3
DEFAULT_DELAY    = 0.005   # segundos entre pontos consecutivos


class TurtleDrawNode(Node):
    """
    Nó ROS 2 responsável por:
      1. Rodar a pipeline de visão computacional
      2. Limpar a tela do turtlesim
      3. Percorrer cada segmento de contorno com a tartaruga
    """

    def __init__(self):
        super().__init__('turtle_draw_node')

        # ---- Declarar parâmetros ROS 2 (podem ser sobrescritos via CLI) ----
        self.declare_parameter('image',      DEFAULT_IMAGE)
        self.declare_parameter('max_size',   DEFAULT_MAX_SIZE)
        self.declare_parameter('gauss_size', DEFAULT_GAUSS_SZ)
        self.declare_parameter('gauss_sigma',DEFAULT_GAUSS_SG)
        self.declare_parameter('canny_low',  DEFAULT_CLOW)
        self.declare_parameter('canny_high', DEFAULT_CHIGH)
        self.declare_parameter('max_seg',    DEFAULT_MAX_SEG)
        self.declare_parameter('min_seg',    DEFAULT_MIN_SEG)
        self.declare_parameter('step',       DEFAULT_STEP)
        self.declare_parameter('delay',      DEFAULT_DELAY)

        # ---- Criar clientes de serviço -------------------------------------
        self.cli_reset    = self.create_client(Empty,           '/reset')
        self.cli_set_pen  = self.create_client(SetPen,          '/turtle1/set_pen')
        self.cli_teleport = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')

        # Aguardar serviços ficarem disponíveis (turtlesim pode demorar a subir)
        self.get_logger().info('Aguardando serviços do turtlesim...')
        for cli, name in [
            (self.cli_reset,    '/reset'),
            (self.cli_set_pen,  '/turtle1/set_pen'),
            (self.cli_teleport, '/turtle1/teleport_absolute'),
        ]:
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'Aguardando {name}...')

        self.get_logger().info('Todos os serviços disponíveis!')

        # ---- Executar pipeline e iniciar desenho ---------------------------
        self._run()

    # -----------------------------------------------------------------------
    # Auxiliares de serviço (chamadas síncronas)
    # -----------------------------------------------------------------------

    def _call_sync(self, client, request):
        """
        Chama um serviço ROS 2 de forma síncrona (spin_until_future_complete).
        Necessário porque o nó não usa callbacks assíncronos para o desenho.
        """
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().error('Chamada de serviço falhou!')
        return future.result()

    def _reset(self):
        """Reseta o turtlesim (limpa a tela e retorna a tartaruga ao centro)."""
        self.get_logger().info('Resetando turtlesim...')
        self._call_sync(self.cli_reset, Empty.Request())
        time.sleep(1.0)  # Aguardar a tela limpar

    def _set_pen(self, r: int, g: int, b: int, width: int, off: bool):
        """
        Configura a caneta da tartaruga.

        Args:
            r, g, b: cor da caneta (0-255)
            width:   espessura da linha em pixels
            off:     True para levantar a caneta (sem traço)
        """
        req = SetPen.Request()
        req.r     = r
        req.g     = g
        req.b     = b
        req.width = width
        req.off   = int(off)
        self._call_sync(self.cli_set_pen, req)

    def _teleport(self, x: float, y: float, theta: float = 0.0):
        """
        Teleporta a tartaruga para a posição (x, y) com orientação theta.
        Com a caneta ligada, deixa um traço; com a caneta desligada, move sem traço.

        Args:
            x, y:  coordenadas destino no espaço turtlesim [0..11.09]
            theta: ângulo de orientação em radianos (padrão 0.0)
        """
        req = TeleportAbsolute.Request()
        req.x     = float(x)
        req.y     = float(y)
        req.theta = float(theta)
        self._call_sync(self.cli_teleport, req)

    # -----------------------------------------------------------------------
    # Pipeline principal
    # -----------------------------------------------------------------------

    def _run_vision_pipeline(self) -> List[np.ndarray]:
        """
        Executa a pipeline de visão computacional e retorna os segmentos
        em coordenadas turtlesim.
        """
        # Ler parâmetros
        image_path  = self.get_parameter('image').value
        if not os.path.isabs(image_path):
            image_path = os.path.join(_PKG_ROOT, image_path)
            
        max_size    = self.get_parameter('max_size').value
        gauss_size  = self.get_parameter('gauss_size').value
        gauss_sigma = self.get_parameter('gauss_sigma').value
        canny_low   = self.get_parameter('canny_low').value
        canny_high  = self.get_parameter('canny_high').value
        max_seg     = self.get_parameter('max_seg').value
        min_seg     = self.get_parameter('min_seg').value
        step        = self.get_parameter('step').value

        self.get_logger().info(f'Carregando imagem: {image_path}')
        original = load_image(image_path)

        self.get_logger().info('Pré-processamento: grayscale + gaussiana...')
        small   = resize(original, max_size=max_size)
        gray    = bgr_to_gray(small)
        blurred = gaussian_blur(gray, size=gauss_size, sigma=gauss_sigma)

        self.get_logger().info('Detecção de bordas: Canny...')
        edges = canny(blurred, low_ratio=canny_low, high_ratio=canny_high)
        n_edges = int(np.sum(edges > 0))
        self.get_logger().info(f'  → {n_edges} pixels de borda detectados')

        self.get_logger().info('Planejamento de caminho...')
        segments = plan_path(edges,
                             max_segments=max_seg,
                             min_segment_len=min_seg,
                             sample_step=step)
        total_pts = sum(len(s) for s in segments)
        self.get_logger().info(
            f'  → {len(segments)} segmentos, {total_pts} pontos no total'
        )
        return segments

    def _draw_segments(self, segments: List[np.ndarray]):
        """
        Comanda a tartaruga para percorrer todos os segmentos.

        Para cada segmento:
          1. Levanta a caneta (pen off)
          2. Teleporta para o primeiro ponto do segmento
          3. Liga a caneta (pen on)
          4. Teleporta ponto a ponto ao longo do segmento (desenhando o traço)

        Cores variam por segmento para facilitar visualização da ordem.
        """
        delay = self.get_parameter('delay').value

        # Paleta de cores para os segmentos (R, G, B)
        palette = [
            (255, 255, 255),  # branco
            (100, 200, 255),  # azul claro
            (255, 150, 50),   # laranja
            (100, 255, 150),  # verde claro
            (255, 100, 150),  # rosa
            (200, 100, 255),  # violeta
            (255, 230, 50),   # amarelo
            (50,  200, 200),  # ciano
        ]

        total_segs = len(segments)
        self.get_logger().info(f'Iniciando desenho: {total_segs} segmentos...')

        for seg_idx, segment in enumerate(segments):
            n_pts = len(segment)
            color = palette[seg_idx % len(palette)]
            self.get_logger().info(
                f'  Segmento {seg_idx+1}/{total_segs}: {n_pts} pontos'
            )

            # 1. Levantar caneta e mover para o início do segmento
            self._set_pen(*color, width=2, off=True)
            self._teleport(segment[0, 0], segment[0, 1])

            # 2. Ligar caneta
            self._set_pen(*color, width=2, off=False)

            # 3. Percorrer o segmento ponto a ponto
            for pt in segment[1:]:
                self._teleport(pt[0], pt[1])
                if delay > 0:
                    time.sleep(delay)

        # Levanta a caneta ao finalizar
        self._set_pen(255, 255, 255, width=2, off=True)
        self.get_logger().info('Desenho concluído!')

    def _run(self):
        """Orquestra a pipeline completa: visão → reset → desenho."""
        try:
            # 1. Pipeline de visão
            segments = self._run_vision_pipeline()

            if not segments:
                self.get_logger().error(
                    'Nenhum segmento gerado! Verifique a imagem e os parâmetros.'
                )
                return

            # 2. Limpar tela do turtlesim
            self._reset()

            # 3. Desenhar
            self._draw_segments(segments)

        except Exception as e:
            self.get_logger().error(f'Erro durante execução: {e}')
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TurtleDrawNode()
    # O nó não precisa ficar em spin — o trabalho é feito no __init__
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
