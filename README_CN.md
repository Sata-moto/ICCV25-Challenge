以下是该 README 文件的中文翻译：

---

<!--  
SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>  
SPDX-FileContributor: Amir Mohammadi <amir.mohammadi@idiap.ch>  
SPDX-License-Identifier: MIT  
-->

# DeepID 2025 挑战赛（ICCV 2025）演示与基线模型

本代码仓库既可作为基线模型使用，也可作为参赛者提交自己解决方案的工具，适用于 ICCV 2025 的身份文件合成篡改检测挑战赛（DeepID 2025）。
参赛者可以使用此代码运行官方基线，也可以在此基础上修改、集成自己的模型，并构建 Docker 镜像以提交参赛方案。

---

## 测试基线模型

如果你在安装了 NVIDIA GPU 和 Docker 的计算机上运行该代码，可以通过 API 启动 [TruFor](https://github.com/grip-unina/TruFor) 作为基线模型。

首先运行以下命令启动模型 API：

```bash
docker compose up -d --build
```

然后通过以下命令测试 API：

```bash
pytest -sv test_api.py
```

你需要一个 Python 环境，并安装以下依赖：`pytest requests numpy pillow`。
测试输出结果如下：

```text
===================================================================================== 测试会话开始 =====================================================================================
平台：Linux -- Python 3.11.12，pytest-8.3.5，pluggy-1.5.0 -- .pixi/envs/test/bin/python3.11
缓存目录: .pytest_cache
根目录: .
收集到 5 个测试项                                                                                                                                       

test_api.py::test_server_is_running 通过
test_api.py::test_detect_endpoint 
原始图像 pristine1.jpg 得分: 0.83
原始图像 pristine2.jpg 得分: 0.82
篡改图像 tampered1.png 得分: 0.00
篡改图像 tampered2.png 得分: 0.00
通过
test_api.py::test_localize_endpoint 
原始图像 pristine1.jpg 白色区域占比: 99.27%
原始图像 pristine2.jpg 白色区域占比: 93.51%
篡改图像 tampered1.png 黑色区域占比: 11.12%
篡改图像 tampered2.png 黑色区域占比: 71.09%
通过
test_api.py::test_detect_and_localize_endpoint 
原始图像 pristine1.jpg 侦测+定位得分: 0.83，白色区域占比: 99.27%
原始图像 pristine2.jpg 侦测+定位得分: 0.82，白色区域占比: 93.51%
篡改图像 tampered1.png 侦测+定位得分: 0.00，黑色区域占比: 11.12%
篡改图像 tampered2.png 侦测+定位得分: 0.00，黑色区域占比: 71.09%
通过
test_api.py::test_api_compliance API 合规性测试通过！
通过

======================================================================= 5 项测试全部通过，耗时 9.44s ========================================================================
```

---

## 提交你自己的模型到 DeepID 挑战赛

你可以修改本仓库，将自己的模型集成到 API 中。
两个赛道（*检测* 和 *定位*）在同一个 Docker 容器中实现，但你也可以只选择其中一个实现。

从删除官方基线 [`src/trufor`](src/trufor/) 开始，并将你自己的模型实现到 [`src/model.py`](src/model.py) 中。

---

### 检测赛道（Track 1）

若只参与检测任务，请修改 [`src/main.py`](src/main.py) 中如下标注了 `TODO` 的部分：

```python
@app.post("/detect")
def detect(image: UploadFile):
        ...

        # TODO: 你的代码从此处开始 ***
        # 在此处进行图像预处理、推理，并返回一个浮点分数
        img = preprocess_image(img)
        score = MODEL.detect(img)
        # *** 你的代码到此结束 ***
```

如果不参与定位赛道，请删除 `localize` 函数部分代码：

```python
# 如果不参与定位赛道，请删除以下全部内容

@app.post(
    "/localize",
    ...
)
def localize(image: UploadFile):
    try:
        ...
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"图像处理失败: {format_exception(e)}",
        )
```

---

### 定位赛道（Track 2）

若只参与定位任务，请修改 [`src/main.py`](src/main.py) 中如下标注了 `TODO` 的部分：

```python
@app.post(
    "/localize",
    ...
)
def localize(image: UploadFile):
    try:
        ...

        # TODO: 你的代码从此处开始 ***
        # 在此处进行图像预处理、推理，并返回二值掩码图
        img = preprocess_image(img)
        mask = MODEL.localize(img)
        # *** 你的代码到此结束 ***

        ...
```

如果不参与检测赛道，请删除 `detect` 函数。

---

### 联合检测与定位接口（可选但推荐）

如果你的模型能同时处理检测与定位任务，推荐实现 `/detect_and_localize` 接口。该接口在一次推理中完成两个任务，可提高效率。

在 [`src/main.py`](src/main.py) 中修改如下部分：

```python
@app.post("/detect_and_localize", response_model=dict)
def detect_and_localize(image: UploadFile):
    try:
        ...

        # TODO: 你的代码从此处开始 ***
        # 在此处进行图像预处理、推理，并返回分数和掩码
        img = preprocess_image(img)
        score, mask = MODEL.detect_and_localize(img)
        # *** 你的代码到此结束 ***

        ...
```

该接口应返回一个二值 PNG 掩码图像作为响应主体，并在 `X-Score-Value` 响应头中返回检测得分（浮点数，0 到 1 之间，越接近 1 越真实，越接近 0 越可疑）。

---

### 修改 requirements.txt 和 Dockerfile

修改 `requirements.txt` 以包含你的模型所需的依赖项，可以删除 `trufor` 相关依赖，例如：

```text
# API 服务依赖
fastapi[standard]==0.115.12
python-multipart==0.0.20
numpy==2.2.2
pillow==11.0.0

# trufor 依赖（可删除）
torch==2.6.0
timm==1.0.15
```

不要删除服务相关依赖，API 运行需要它们。可以根据你的需求修改这些依赖的版本。

然后在 `Dockerfile` 中加入模型权重：

```dockerfile
# TODO: 在此复制你的模型权重文件，例如：
COPY weights weights
```

---

### 测试你的模型

模型集成完成后，可按 [测试基线](#测试基线模型) 的方式进行测试。

---

### 提交 Docker 镜像

测试无误后，通过以下命令打包 Docker 镜像：

```bash
./prepare_submission.sh <team_name> <track_name> <algorithm_name> <version>
```

* `<track_name>` 需为以下之一：`track_1`（检测）、`track_2`（定位）、或 `track_both`（联合）。

该命令将生成镜像文件 `<team_name>_<track_name>_<algorithm_name>_<version>.tgz`，用于参赛提交。

---

## 问题反馈

如果你在使用代码时遇到任何问题，请在仓库中提交 issue，我们会尽快提供帮助。
