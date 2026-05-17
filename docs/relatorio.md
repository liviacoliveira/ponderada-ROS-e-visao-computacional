## 1. Pré-processamento

### 1.1 Conversão para Escala de Cinza

A imagem é carregada em formato BGR (padrão do OpenCV) e convertida para escala de cinza usando a fórmula de luminância percebida **ITU-R BT.601**:

```
Y = 0.114·B + 0.587·G + 0.299·R
```

**Justificativa:** Os coeficientes não são iguais porque o olho humano tem sensibilidades diferentes para cada comprimento de onda — somos mais sensíveis ao verde (~59%) e menos ao azul (~11%). Usar essa fórmula em vez de uma média simples (Y = (R+G+B)/3) preserva melhor a percepção de contraste nas bordas, o que melhora a qualidade da detecção posterior.

### 1.2 Redimensionamento

A imagem é redimensionada proporcionalmente para que o maior lado tenha no máximo 256 pixels, usando **interpolação bilinear** implementada manualmente com NumPy.

**Justificativa:** O turtlesim tem espaço limitado (~11×11 unidades) e resolução visual baixa. Trabalhar com a imagem original (~1024×683px) geraria dezenas de milhares de pontos de borda — muito mais do que a tartaruga consegue percorrer de forma visualmente coerente. O redimensionamento equilibra fidelidade visual e tempo de execução.

### 1.3 Suavização Gaussiana

Aplicamos um filtro gaussiano **5×5 com σ=1.4**, implementado do zero:

1. **Kernel gaussiano** gerado com a fórmula `G(x,y) = exp(-(x²+y²)/(2σ²))`, normalizado para soma 1.
2. **Convolução 2D** com padding de reflexão (evita artefatos de borda com zeros).

**Justificativa:** Bordas reais são transições suaves de intensidade. Ruído de alta frequência (granulado, compressão JPEG) cria falsas bordas. A gaussiana elimina esse ruído antes do detector de bordas, reduzindo detecções espúrias. O tamanho 5×5 com σ=1.4 é a configuração original proposta por Canny (1986) e é um ponto de equilíbrio entre suavização e preservação de bordas verdadeiras.

---

## 2. Detecção de Bordas — Algoritmo de Canny

Implementamos o algoritmo de Canny completo em 4 sub-etapas:

### 2.1 Gradientes com Filtros de Sobel

Os filtros de Sobel aproximam as derivadas parciais da intensidade:

```
Kx = [[-1, 0, 1],    Ky = [[-1, -2, -1],
      [-2, 0, 2],          [ 0,  0,  0],
      [-1, 0, 1]]           [ 1,  2,  1]]
```

- **Magnitude:** `G = √(Gx² + Gy²)` — mede a força da borda
- **Direção:** `θ = arctan2(Gy, Gx)` — indica a direção perpendicular à borda

### 2.2 Supressão de Não-Máximos (NMS)

Para cada pixel, comparamos sua magnitude com os dois vizinhos na **direção do gradiente** (quantizado em 4 direções: 0°, 45°, 90°, 135°). Se o pixel não for máximo local, é zerado. Isso produz bordas de **1 pixel de largura**.

**Implementação:** versão vetorizada com slicing NumPy (sem loops Python sobre pixels) para desempenho adequado.

### 2.3 Limiarização Dupla

Dois limiares adaptativos (frações do valor máximo de magnitude):
- **High (15%):** pixels com `G ≥ high` → borda forte (certeza)
- **Low (5%):** pixels com `low ≤ G < high` → borda fraca (candidata)
- Abaixo de `low` → descartado

### 2.4 Rastreamento por Histerese

BFS (busca em largura) partindo de cada borda forte: bordas fracas **conectadas** a bordas fortes são promovidas a bordas reais; bordas fracas isoladas são descartadas como ruído.

**Justificativa para escolha do Canny:** Comparado a simples Sobel + limiar único, o Canny produz bordas finas (1px), contínuas e com muito menos falsos positivos. A histerese é o diferencial — ela elimina ruído sem quebrar bordas reais. Para o turtlesim, bordas finas e contínuas são essenciais: bordas espessas gerariam sobreposição de traços e bordas quebradas gerariam saltos na trajetória.

---

## 3. Planejamento de Caminho

### 3.1 Segmentação por Componentes Conexos

Em vez de tratar os pixels de borda como uma nuvem global, agrupamos pixels vizinhos (8-conectividade) em **segmentos contínuos** via BFS. Segmentos com menos de 15 pixels são descartados (ruído residual). Os maiores segmentos são priorizados.

### 3.2 Ordenação por Vizinho Mais Próximo

Dentro de cada segmento, os pixels são reordenados pelo algoritmo **Greedy Nearest Neighbor**: partindo do pixel mais próximo à origem, sempre movemos ao ponto não visitado mais próximo. Isso cria um caminho contínuo ao longo do contorno, evitando saltos aleatórios.

### 3.3 Mapeamento para o Turtlesim

Conversão de coordenadas de pixel [row, col] para [x, y] turtlesim (11.09×11.09 unidades):
- Escala proporcional com margem de 0.5 unidades
- Inversão do eixo Y (imagem: Y↓ / turtlesim: Y↑)

---

## 4. Controle ROS 2

O nó `draw_node` usa **três serviços** do turtlesim:

| Serviço | Uso |
|---|---|
| `/reset` | Limpa a tela antes de desenhar |
| `/turtle1/set_pen` | Liga/desliga caneta; define cor e espessura |
| `/turtle1/teleport_absolute` | Posiciona a tartaruga com precisão |

**Estratégia de desenho:** Para cada segmento, a caneta é **desligada** e a tartaruga teleporta para o ponto inicial; em seguida, a caneta é **ligada** e a tartaruga percorre o segmento ponto a ponto. Entre segmentos, cores diferentes ajudam a identificar os contornos visualmente.

**Escolha por teleport vs. cmd_vel:** O teleporte garante precisão absoluta de posicionamento. O `cmd_vel` acumularia erros de integração numérica a cada passo, distorcendo o desenho final. Para o objetivo desta atividade — reproduzir fielmente um contorno extraído de visão computacional — a precisão supera o realismo cinemático.

---

## 5. Dificuldades Encontradas

- **Desempenho da convolução:** A convolução 2D pura com loops Python sobre pixels era impraticável (>5min para imagem 256px). Solução: versão vetorizada com slicing NumPy que processa todos os deslocamentos do kernel de forma matricial.
- **Ordenação de pontos:** Pixels de borda não têm ordem natural. O algoritmo de vizinho mais próximo funciona bem para contornos localmente contínuos, mas pode criar "atalhos" em cruzamentos de bordas.
- **Escala de limiares:** Limiares fixos (ex: 100/200) não funcionam para imagens com diferentes faixas de contraste. A solução foi usar **limiares relativos** ao máximo de magnitude de cada imagem.
- **Extração de Contorno vs Texturas:** O algoritmo de Canny é um detector de bordas genérico, não um segmentador semântico. Isso significa que ele detecta todas as transições abruptas (textura do fundo, dobras do pelo, linhas no chão), e não apenas a "silhueta" principal do cachorro. A dificuldade foi realizar o ajuste fino iterativo dos parâmetros (aumentar o *blur* e baixar os limiares) para encontrar um balanço onde as orelhas fossem detectadas sem que o fundo tomasse conta do desenho inteiro.
