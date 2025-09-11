# PaddleOCR C++ 推理项目

## 项目概述

此目录包含用于 PaddleOCR 模型的 C++ 推理部署代码。PaddleOCR 是一个基于 PaddlePaddle 的 OCR（光学字符识别）工具包。此特定子项目专注于使用 C++ 部署 PaddleOCR 模型，以获得更好的性能，特别适用于需要 CPU 和 GPU 部署的生产环境。它支持各种 OCR 流程，包括文本检测 (det)、文本识别 (rec)、文本方向分类 (cls)、布局分析 (layout) 和表格识别 (table)。

主要组件包括：
- C++ 推理逻辑的源代码 (`src/`, `include/`)。
- 使用 CMake 的构建系统配置 (`CMakeLists.txt`, `tools/build.sh`)。
- 文档和示例 (`readme.md`, `docs/`)。

## 构建和运行

### 先决条件

1.  **环境**: Linux (推荐使用 Docker) 或 Windows。
2.  **依赖项**:
    *   **Paddle 推理库**: 下载或编译 PaddlePaddle 推理库。请参阅 [Paddle 推理库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)。
    *   **OpenCV**: 编译 OpenCV (推荐版本 3.4.7，但更新版本可能也适用)。请参阅 `readme.md` 获取编译说明。
    *   **CUDA & cuDNN** (可选，用于 GPU 支持): 如果在 GPU 上部署，请确保 CUDA 和 cuDNN 库可用。
    *   **TensorRT** (可选，用于 TensorRT 支持): 如果使用 TensorRT 进行优化。

### 构建

1.  **准备依赖项**:
    *   下载/编译 Paddle 推理库并记下其路径 (例如, `paddle_inference`)。
    *   编译 OpenCV 并记下其安装路径 (例如, `opencv3`)。
    *   如果使用 GPU，请确保 CUDA (`cuda/lib64`) 和 cuDNN (`lib/x86_64-linux-gnu/` 或类似路径) 的路径已知。
2.  **配置构建**:
    *   编辑 `tools/build.sh` 以设置环境中所需的路径:
        *   `OPENCV_DIR`: OpenCV 安装路径。
        *   `LIB_DIR`: Paddle 推理库路径 (`paddle_inference` 或 `build/paddle_inference_install_dir`)。
        *   `CUDA_LIB_DIR`: CUDA 库路径 (例如, `/usr/local/cuda/lib64`)。
        *   `CUDNN_LIB_DIR`: cuDNN 库路径 (例如, `/usr/lib/x86_64-linux-gnu/`)。
    *   或者，您可以直接将这些路径传递给 `cmake` 命令行 (请参阅 `CMakeLists.txt` 了解变量)。
3.  **运行构建脚本**:
    ```bash
    cd tools
    sh build.sh
    ```
    此脚本会清理 `build` 目录，使用指定的选项 (例如，默认 `WITH_MKL=ON`, `WITH_GPU=OFF`) 通过 CMake 配置项目，并使用 `make` 编译项目。

    *Windows 注意事项*: 请参阅 `docs/windows_vs2019_build.md` 获取使用 Visual Studio 的具体说明。

### 运行示例

1.  **导出 PaddleOCR 模型**: 您需要 Paddle 推理模型 (`.pdmodel`, `.pdiparams`) 用于检测、识别、分类等。从 Python PaddleOCR 项目导出它们。将它们放在一个目录中 (例如, `inference/`)。请参阅 `readme.md` 了解预期的目录结构 (例如, `inference/det_db/`, `inference/rec_rcnn/`)。
2.  **执行**:
    构建后，会在 `build` 目录中生成一个名为 `ppocr` 的可执行文件。使用适当的参数运行它：
    ```bash
    ./build/ppocr --<param1>=<value1> --<param2>=<value2> ...
    ```
    常见流程和参数：
    *   **检测 + 识别 + 分类**:
        ```bash
        ./build/ppocr --det_model_dir=inference/det_db \
            --rec_model_dir=inference/rec_rcnn \
            --cls_model_dir=inference/cls \
            --image_dir=../../doc/imgs/12.jpg \
            --use_angle_cls=true \
            --det=true \
            --rec=true \
            --cls=true
        ```
    *   **检测 + 识别**:
        ```bash
        ./build/ppocr --det_model_dir=inference/det_db \
            --rec_model_dir=inference/rec_rcnn \
            --image_dir=../../doc/imgs/12.jpg \
            --use_angle_cls=false \
            --det=true \
            --rec=true \
            --cls=false
        ```
    *   **布局分析 + 表格识别**:
        ```bash
        ./build/ppocr --det_model_dir=inference/det_db \
            --rec_model_dir=inference/rec_rcnn \
            --table_model_dir=inference/table \
            --image_dir=../../ppstructure/docs/table/table.jpg \
            --layout_model_dir=inference/layout \
            --type=structure \
            --table=true \
            --layout=true
        ```

    请参阅 `readme.md` 获取参数和不同 OCR 任务示例命令的全面列表。使用 `--help` 查看 `include/args.h` 中定义的所有可用标志。

    **重要**: 默认识别模型输入形状为 `3, 48, 320`。如果使用需要 `3, 32, 320` 的旧模型，请添加 `--rec_img_h=32`。

## 开发规范

*   **语言**: 使用 C++11，如 CMake 配置 (`-std=c++11`) 所示。
*   **构建系统**: CMake 是主要的构建系统。
*   **依赖管理**: 通过 CMake 配置链接依赖项，如 Paddle 推理库、OpenCV、gflags、glog、protobuf 等。路径在 CMake 配置步骤中指定。
*   **代码结构**:
    *   `src/`: 包含主要的 C++ 源文件，包括 `main.cpp`。
    *   `include/`: 包含定义 OCR 组件 (检测、分类、识别) 和实用函数的头文件。
    *   `main.cpp` 是入口点，处理参数解析 (使用 `gflags`)、模型加载、图像处理和调用适当的 OCR 流水线函数。
    *   `PPOCR` 类 (在 `paddleocr.h/cpp` 中) 通过调用单个组件 (`DBDetector`, `Classifier`, `CRNNRecognizer`) 来编排 OCR 过程。
*   **使用的库**:
    *   **Paddle 推理库**: 用于运行 PaddlePaddle 模型的核心库。
    *   **OpenCV**: 用于图像加载、操作和可视化。
    *   **gflags**: 用于命令行参数解析。
    *   **glog**: 用于日志记录 (尽管也直接使用 `std::cout`/`std::cerr`)。
    *   **jsoncpp**: 用于 JSON 解析/写入 (例如，用于客户端/服务器通信或结果格式化)。
*   **GPU 支持**: 条件编译选项 (`WITH_GPU`, `WITH_TENSORRT`) 允许为 CPU 或 GPU 目标构建。
*   **静态与动态链接**: 构建系统支持 Paddle 推理库的静态链接 (`WITH_STATIC_LIB=ON`) 和动态链接 (`WITH_STATIC_LIB=OFF`)。