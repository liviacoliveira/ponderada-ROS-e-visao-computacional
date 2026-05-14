"""
draw.launch.py — Launch file que sobe o turtlesim e o nó de desenho juntos.

Uso:
    ros2 launch turtle_draw draw.launch.py
    ros2 launch turtle_draw draw.launch.py image:=/caminho/da/imagem.jpg
    ros2 launch turtle_draw draw.launch.py max_seg:=30 delay:=0.005
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Declarar argumentos configuráveis via CLI ------------------------
    args = [
        DeclareLaunchArgument('image',
            default_value='image/bulldog.jpg',
            description='Caminho da imagem de entrada'),
        DeclareLaunchArgument('max_size',
            default_value='256',
            description='Tamanho máximo da imagem após redimensionamento'),
        DeclareLaunchArgument('gauss_size',
            default_value='5',
            description='Tamanho do kernel gaussiano (ímpar)'),
        DeclareLaunchArgument('gauss_sigma',
            default_value='1.4',
            description='Sigma do filtro gaussiano'),
        DeclareLaunchArgument('canny_low',
            default_value='0.05',
            description='Limiar baixo do Canny (fração do máximo)'),
        DeclareLaunchArgument('canny_high',
            default_value='0.15',
            description='Limiar alto do Canny (fração do máximo)'),
        DeclareLaunchArgument('max_seg',
            default_value='20',
            description='Número máximo de segmentos de contorno'),
        DeclareLaunchArgument('min_seg',
            default_value='15',
            description='Tamanho mínimo de um segmento (pixels)'),
        DeclareLaunchArgument('step',
            default_value='3',
            description='Amostragem: pegar 1 a cada N pontos'),
        DeclareLaunchArgument('delay',
            default_value='0.01',
            description='Delay em segundos entre pontos consecutivos'),
    ]

    # ---- Nó turtlesim -------------------------------------------------------
    turtlesim_node = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='turtlesim',
        output='screen',
    )

    # ---- Nó de desenho ------------------------------------------------------
    draw_node = Node(
        package='turtle_draw',
        executable='draw_node',
        name='turtle_draw_node',
        output='screen',
        parameters=[{
            'image':       LaunchConfiguration('image'),
            'max_size':    LaunchConfiguration('max_size'),
            'gauss_size':  LaunchConfiguration('gauss_size'),
            'gauss_sigma': LaunchConfiguration('gauss_sigma'),
            'canny_low':   LaunchConfiguration('canny_low'),
            'canny_high':  LaunchConfiguration('canny_high'),
            'max_seg':     LaunchConfiguration('max_seg'),
            'min_seg':     LaunchConfiguration('min_seg'),
            'step':        LaunchConfiguration('step'),
            'delay':       LaunchConfiguration('delay'),
        }],
    )

    return LaunchDescription(args + [turtlesim_node, draw_node])
