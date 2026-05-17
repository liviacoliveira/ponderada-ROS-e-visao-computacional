# Documentação Técnica

Este documento é a documentação técnica oficial da Ponderada de ROS e Visão Computacional. Ele detalha as decisões de projeto, a arquitetura e a matemática por trás da implementação de uma pipeline de visão computacional, bem como o método utilizado para realizar o mapeamento e o controle da tartaruga no simulador Turtlesim do ROS 2. 

---

## 1. Pré-processamento

### 1.1 Conversão para Escala de Cinza

Trabalhar com imagens coloridas (3 canais) triplica o custo computacional sem adicionar informações significativas para a detecção de bordas (que se baseia em contraste de luminosidade). Optou-se por usar a fórmula de luminância percebida ITU-R BT.601, pois os coeficientes não são iguais: o olho humano tem sensibilidades diferentes para cada cor (somos mais sensíveis ao verde e menos ao azul). Isso preserva melhor a percepção de contraste do que uma média simples (R+G+B)/3.

**Implementação:** A imagem é carregada em formato BGR (padrão do OpenCV) e convertida para um único canal em escala de cinza usando a fórmula: `Y = 0.114·B + 0.587·G + 0.299·R`.

### 1.2 Redimensionamento

O simulador turtlesim possui um canvas de espaço limitado (~11×11 unidades) e baixa resolução visual (o traço da caneta é espesso). Trabalhar com uma imagem original de alta resolução (ex: 1024x683px) geraria dezenas de milhares de pontos, tornando o traçado no simulador extremamente poluído, demorado e confuso. A decisão foi fixar um limite de tamanho, equilibrando fidelidade visual e tempo de execução.

**Implementação:** A imagem é redimensionada proporcionalmente para que o maior lado tenha no máximo 256 pixels. O algoritmo utilizado foi a interpolação bilinear, implementada de forma manual com operações matriciais em NumPy.

### 1.3 Suavização Gaussiana

Bordas reais em imagens são transições suaves de intensidade. No entanto, ruídos de alta frequência (granulado fotográfico, artefatos de compressão JPEG) criam bordas falsas. A aplicação prévia de um filtro Gaussiano foi escolhida para eliminar esse ruído, evitando que o algoritmo de Canny gerasse detecções espúrias. Optou-se por um kernel 5x5 com σ=1.4 por ser o ponto de equilíbrio clássico sugerido na literatura para preservação de bordas verdadeiras.

**Implementação:** A suavização é feita gerando um kernel gaussiano com a fórmula matemática da distribuição normal 2D, seguido de uma convolução 2D. Para evitar artefatos de contorno escuro, utilizou-se a técnica de padding por reflexão.

---

## 2. Detecção de Bordas (Algoritmo de Canny)

Comparado a uma detecção simples com Sobel e um único limiar, optou-se por implementar a pipeline completa de Canny. O Sobel simples gera bordas muito espessas e ruidosas, enquanto o Canny, através da Supressão de Não-Máximos e da Histerese, produz bordas muito finas (1 pixel de largura) e contínuas. Para o turtlesim, isso é essencial, pois bordas finas evitam o sombreamento excessivo de traços na mesma região.

A implementação consiste em 4 sub-etapas:

1. **Gradientes com Filtros de Sobel:**
   - **Implementação:** Aproximam-se as derivadas parciais da intensidade convoluindo a imagem com os kernels Kx e Ky de Sobel. Em seguida, calcula-se a magnitude `G = √(Gx² + Gy²)` e a direção `θ = arctan2(Gy, Gx)` da borda.

2. **Supressão de Não-Máximos (NMS):**
   - **Implementação:** Cada pixel tem sua magnitude comparada com seus dois vizinhos na direção do gradiente (arredondado para 0°, 45°, 90° ou 135°). Se não for um máximo local, seu valor é zerado. A versão foi vetorizada com slicing em NumPy.

3. **Limiarização Dupla (Double Threshold):**
   - **Implementação:** O uso de frações relativas ao valor máximo da imagem classifica os pixels em bordas "fortes", "fracas" e "ruído" (descartado). 

4. **Rastreamento por Histerese:**
   - **Implementação:** Executa-se uma Busca em Largura (BFS) partindo das bordas fortes. As bordas fracas que estiverem diretamente conectadas a uma borda forte são transformadas em bordas válidas, enquanto as isoladas são descartadas.

---

## 3. Planejamento de Caminho

### 3.1 Segmentação e Ordenação

A saída do algoritmo de Canny é apenas uma máscara binária indicando onde existem bordas, mas a tartaruga precisa de uma sequência ordenada de coordenadas. Tentar ir do pixel A ao B sem ordem criaria "riscos" atravessando a tela inteira. A solução escolhida foi primeiro agrupar pixels conectados formando segmentos independentes, e depois ordená-los internamente.

**Implementação:**
- **Componentes Conexos:** Agrupamos pixels vizinhos (8-conectividade) via BFS. Segmentos excessivamente pequenos (ex: < 5 pixels) são eliminados por serem ruído residual.
- **Vizinho Mais Próximo (Nearest Neighbor):** Dentro de um mesmo segmento, inicia-se em um extremo e caminha-se iterativamente para o pixel não-visitado mais próximo, criando um caminho fluido e localmente contínuo.

### 3.2 Mapeamento para o Turtlesim

O sistema de coordenadas da imagem (pixels, origem no topo-esquerdo) não é o mesmo do turtlesim (escala de 0 a 11.09, origem no centro). Foi necessário criar uma transformação afim para traduzir o desenho corretamente.

**Implementação:** Conversão de [row, col] para [x, y], aplicando uma escala que preserva a proporção da imagem dentro de uma margem de segurança de 0.5 unidades, invertendo o eixo Y para acompanhar o comportamento da tela.

---

## 4. Controle ROS 2

Para desenhar um contorno extraído de uma imagem, a fidelidade da posição final é crucial. A escolha entre usar tópicos de velocidade (`cmd_vel`) ou chamadas de serviço de teleporte (`teleport_absolute`) foi feita em favor do teleporte. O `cmd_vel` acumula pequenos erros de integração numérica (física de tempo real) que fariam com que o final do desenho não se conectasse corretamente com o início, deformando as formas. O teleporte garante precisão matemática absoluta para cada vértice.

**Implementação:**
O nó `draw_node` controla a tartaruga manipulando três serviços sequencialmente:
1. `turtle1/set_pen`: Desliga a caneta para evitar rabiscar o fundo.
2. `turtle1/teleport_absolute`: Move a tartaruga instantaneamente até o início de um novo segmento.
3. `turtle1/set_pen`: Liga a caneta.
4. `turtle1/teleport_absolute`: Visita todos os pontos do segmento atual, desenhando a linha efetiva.

---

## 5. Dificuldades Encontradas

- **Desempenho da convolução:** A convolução 2D pura com loops Python sobre pixels era impraticável (>5min para imagem 256px). Solução: versão vetorizada com slicing NumPy que processa todos os deslocamentos do kernel de forma matricial.
- **Ordenação de pontos:** Pixels de borda não têm ordem natural. O algoritmo de vizinho mais próximo funciona bem para contornos localmente contínuos, mas pode criar "atalhos" em cruzamentos de bordas.
- **Escala de limiares:** Limiares fixos (ex: 100/200) não funcionam para imagens com diferentes faixas de contraste. A solução foi usar **limiares relativos** ao máximo de magnitude de cada imagem.
- **Extração de Contorno vs Texturas:** O algoritmo de Canny é um detector de bordas genérico, não um segmentador semântico. Isso significa que ele detecta todas as transições abruptas (textura do fundo, dobras do pelo, linhas no chão), e não apenas a "silhueta" principal do cachorro. A dificuldade foi realizar o ajuste fino iterativo dos parâmetros (aumentar o *blur* e baixar os limiares) para encontrar um balanço onde as orelhas fossem detectadas sem que o fundo tomasse conta do desenho inteiro.
