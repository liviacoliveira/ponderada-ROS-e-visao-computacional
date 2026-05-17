# Ponderada - Turtle Draw 

Nessa ponderada, foi elaborado um pipeline de visão computacional do zero + pacote ROS 2 que faz a tartaruga do Turtlesim reproduzir os contornos de uma imagem.

---

## Visão Geral

O sistema implementa uma pipeline completa em 4 etapas:

```
Imagem → Pré-processamento → Detecção de Bordas → Planejamento de Caminho → Turtlesim
```

| Etapa | Técnica | Implementação |
|---|---|---|
| Pré-processamento | Grayscale (luminância) + Gaussiana 5×5 | NumPy puro |
| Detecção de bordas | Canny (Sobel + NMS + Histerese) | NumPy puro |
| Planejamento | Componentes conexos + Nearest Neighbor | NumPy puro |
| Controle ROS 2 | TeleportAbsolute + SetPen | rclpy |

---

## Estrutura do Projeto

A estrutura do projeto em questão está organizada da seguinte forma:

```
ponderada-ROS-e-visao-computacional/
├── image/
│   └── bulldog.jpg              # Imagem de entrada
├── vision/                      # Pipeline de visão (NumPy puro)
│   ├── loader.py                # cv2.imread — único uso do OpenCV
│   ├── preprocessing.py         # Grayscale, resize, gaussiana
│   ├── edge_detection.py        # Algoritmo de Canny completo
│   ├── path_planning.py         # Contornos → caminho turtlesim
│   └── visualize.py             # Visualização com Matplotlib
├── turtle_draw/                 # Pacote ROS 2
│   ├── package.xml
│   ├── setup.py
│   ├── turtle_draw/
│   │   └── draw_node.py         # Nó de controle da tartaruga
│   └── launch/
│       └── draw.launch.py
├── run_pipeline.py              # Teste da pipeline sem ROS 2
└── docs/
    └── relatorio.md             # Documentação técnica
```

---

## Dependências

| Dependência | Versão recomendada | Uso |
|---|---|---|
| ROS 2 | Humble / Iron / Jazzy | Middleware |
| Python | 3.10+ | — |
| NumPy | ≥ 1.24 | Processamento matricial |
| OpenCV (cv2) | ≥ 4.5 | **Apenas** `cv2.imread` |
| Matplotlib | ≥ 3.5 | Visualização |

### Instalar dependências Python

```bash
pip install numpy opencv-python matplotlib
```

---

## Como Executar

### Opção 1 — Apenas a pipeline de visão (sem ROS 2)

Útil para testar e visualizar as etapas antes de rodar no turtlesim.

```bash
# Na raiz do projeto
python3 run_pipeline.py

# Com parâmetros customizados
python3 run_pipeline.py --image image/bulldog.jpg --size 300 --save

# Ver todas as opções
python3 run_pipeline.py --help
```

### Opção 2 — Pacote ROS 2 completo (turtlesim + desenho)

Toda a execução agora é simplificada para rodar em um único terminal usando um *launch file*, partindo da raiz do projeto:

```bash
# 1. Carregar ambiente ROS 2
source /opt/ros/jazzy/setup.bash   # Ou iron/humble (depende da versão instalada)

# 2. Ir para a pasta do pacote e compilar
cd turtle_draw
colcon build --symlink-install
source install/setup.bash

# 3. Voltar para a raiz e executar (sobe turtlesim + nó de desenho juntos)
cd ..
ros2 launch turtle_draw draw.launch.py
```

#### Parâmetros do launch file (todos opcionais)

```bash
ros2 launch turtle_draw draw.launch.py \
    image:=/caminho/para/imagem.jpg \
    max_size:=256     \  # tamanho máximo da imagem (px)
    max_seg:=150      \  # máximo de segmentos a desenhar
    min_seg:=5        \  # tamanho mínimo de segmento (px)
    step:=3           \  # amostrar 1 a cada N pontos
    delay:=0.005         # delay entre pontos (s), 0 = máximo vel
```

## Ajuste Fino dos Parâmetros

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `max_size` | 256 | Imagens maiores → mais detalhe, mais lento |
| `gauss_sigma` | 1.4 | Maior → bordas mais suaves, menos ruído |
| `canny_low` | 0.05 | Menor → mais bordas fracas detectadas |
| `canny_high` | 0.15 | Maior → apenas bordas muito nítidas |
| `max_seg` | 150 | Mais segmentos → desenho mais completo |
| `min_seg` | 5 | Menor → pega detalhes menores (como olhos e nariz) |
| `step` | 3 | Menor → mais pontos, mais fidelidade |
| `delay` | 0.005 | 0 = velocidade máxima |

---

## Vídeo demonstrativo

