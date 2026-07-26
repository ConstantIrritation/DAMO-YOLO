
# Semantic Analysis of Centering in Convolutional Neural Networks

## 📄 Abstract

Несмотря на широкое использование слоёв нормализации для стабилизации обучения глубоких сетей, семантическая роль операции центрирования остаётся малоизученной: существующие подходы к интерпретируемости - saliency maps, activation patching, linear probing, sparse autoencoders - работают напрямую с активациями, требуют дополнительного обучения или плохо формализуемы, поэтому в этой работе вместо модификации активаций исследуется геометрия представлений - косинусные углы между векторами признаков разных классов и фона на разных этапах прохождения сигнала через сеть. Выдвинута гипотеза, что центрирование в Batch Normalization подавляет фоновую, домен-специфичную компоненту сигнала и усиливает класс-специфичные признаки. Гипотеза проверена на предобученной YOLO-подобной модели следующим образом: с помощью хуков извлечены активации до и после свёртки и центрирования в каждой паре Conv–BN и отслежена динамику косинусных расстояний внутри класса, между классами и между классами и фоном. Результаты показывают, что после центрирования углы внутри класса становятся более сонаправленными, между классами — близкими к ортогональным, а между классами и фоном — противонаправленными, при этом ни один отдельный слой не выделяет конкретный класс избирательно: классовая структура формируется совместной работой свёртки, модифицирующей сигнал вместе с фоном, и центрирования, избирательно подавляющего накопленную общую компоненту. Полученный взгляд даёт дешёвую, не требующую обучения альтернативу SAE и activation patching для задач интерпретируемости и объясняет, почему уже известные приёмы вроде AdaBN, заменяющие BN-статистики source-домена на статистики target-домена, оказываются эффективны для доменной адаптации.




## Repository Structure
 
```
Diploma/
│
├── plots.ipynb                       # Cross-model summary plots
│
├── configs/                          
│   └── damoyolo_tinynasL20_T.py      # Model & dataset configs
│
├── my_help_functions/
│   ├── hooks.py                      # Forward-hook registration/removal for Conv2d–BatchNorm2d pairs
│   ├── tools.py                      # Model loading, cosine-similarity utilities, inference pipeline
│   ├── cosine_matrix.py              # Maps COCO object classes to positions on flattened/collage images
│   ├── create_collage.py             # Builds image collages sampling objects across COCO classes
│   ├── vis_model_arch.py             # Parses Graphviz architecture files and builds dependency graphs
│   └── plots.py                      # Plotting utilities (scrollable figures, architecture visualisation, tables)
│
├── damo/                             # DAMO-YOLO utils from original repository
│
├── architecture_all_deploy.txt       # Model structure for connection vis
│
└── requirements.txt

```

## Reproducing the results


### Pipeline description

<details>
<summary></summary>

### Hook registration

`register_conv_bn_hooks()` iterates over consecutive `(Conv2d, BatchNorm2d)` pairs and registers a forward hook.

The hook extracts activation snapshots. For example, `after conv`.

### Calculate cosine similarities

After the forward pass, for each pair of classes, including the background, the cosine similarity is calculated.

</details>

### Installation

<details>
<summary></summary>

Step1. Install DAMO-YOLO.
```shell
git clone https://github.com/ConstantIrritation/Diploma.git
cd Diploma
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython;
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI # for Windows
```

Step3. Download a pretrained torch from [the benchmark table](https://github.com/tinyvision/damo-yolo#model-zoo) for Tiny model: damoyolo_tinynasL20_T_420.pth
</details>


### Running Experiments
To reproduce any results check **plots.ipynb** for corresponding functions
