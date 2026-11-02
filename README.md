<div align="center">

# go-insightface

<p><strong>纯 Go、无 CGO 的 InsightFace 人脸检测与识别库：SCRFD 检测 + ArcFace 特征提取 + 精确身份级匹配</strong></p>

<p>运行时动态加载 ONNX Runtime 1.23~1.31 · 启动时版本协商 · TensorRT/CUDA 自动启用 · Windows / Linux / macOS（x64 + arm64）</p>

<p>
  <a href="https://github.com/younland/go-insightface"><img src="https://img.shields.io/badge/github-younland%2Fgo--insightface-181717?logo=github&logoColor=white&style=for-the-badge" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge" alt="MIT License"></a>
  <a href="go.mod"><img src="https://img.shields.io/badge/Go-%3E%3D1.27-00ADD8?logo=go&logoColor=white&style=for-the-badge" alt="Go >= 1.27"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%C2%B7%20Linux%20%C2%B7%20macOS-2563eb?style=for-the-badge" alt="Platforms">
</p>

<p>
  <a href="#features">核心特性</a> ·
  <a href="#install">安装</a> ·
  <a href="#quickstart">快速开始</a> ·
  <a href="#config">配置说明</a> ·
  <a href="#assets">资产供给</a> ·
  <a href="#details">功能细节</a> ·
  <a href="#dev">开发指南</a> ·
  <a href="#license">开源协议</a>
</p>

</div>

<a id="features"></a>

### 🎯 纯 Go 绑定，一套代码跑 ORT 1.23 ~ 1.31

启动时读取 `OrtGetApiBase()->GetVersionString()` 解析真实版本，按 `GetAPI(min(minor, 31))` 获取匹配的 `OrtApi` 函数表，只调用该版本实际存在的函数。低于 1.23 拒绝启动，高于 1.31 默认警告后放行。仅依赖 `github.com/ebitengine/purego`，交叉编译无需 C 工具链。

### ⚡ Execution Provider 自动启用

每次加载会话时查询库的可用 provider 列表，按 **TensorRT → CUDA** 优先级自动挂载（V2 provider-options API，默认参数），失败时记录日志并回退 CPU；CPU 构建零配置直接可用。SDK 不暴露 provider 配置项，GPU 用法完全自动。

### 🖥️ SCRFD + ArcFace 全流水线

检测（SCRFD，5 点 landmarks）、对齐、ArcFace 特征提取、L2 归一化一次调用完成。`Detect` / `Embed` / `Analyze` 并发安全，`Close` 等待在途调用且幂等。

### 🔍 精确身份级匹配

`Matcher` 对 3,000+ 人规模图库做精确搜索：每个身份取最优样本比对，同时施加余弦相似度阈值与 Top-1/Top-2 异身份间隔判定；内置基准 3,000 身份 × 3 样本 × 512 维约 `5.11 ms/op`、`0 B/op`。

<a id="install"></a>

## 📦 安装

```bash
go get github.com/younland/go-insightface
```

> [!NOTE]
> 模块路径自迁移到 younland/onnxruntime 起改为 `github.com/younland/go-insightface`，
> 与仓库地址一致，无需 `replace`。历史 tag（v0.2.0 及更早）仍以旧模块路径
> `github.com/lib-x/go-insightface` 发布；引用这些旧版本需自行添加
> `replace github.com/lib-x/go-insightface => github.com/younland/go-insightface v0.2.0`，
> 且旧依赖在 Windows 上无法编译，建议直接使用改名后的新版本。

| 要求 | 版本 / 说明 |
| --- | --- |
| Go | ≥ 1.27（`go.mod` 锁定 1.27.2，工具链自动切换） |
| ONNX Runtime 动态库 | 1.23 ~ 1.31 官方 CPU/GPU 构建，运行时才加载 |
| 操作系统 | Windows x64/arm64 · Linux x64/aarch64 · macOS x86_64/arm64 |
| 模型权重 | 由调用方自备并授权，本模块不含任何权重 |

> [!NOTE]
> 本模块不包含、不发布 ONNX Runtime 二进制与模型权重。动态库的获取方式见[资产供给](#assets)。

<a id="quickstart"></a>

## 🚀 快速开始

```go
package main

import (
	"context"
	"image/jpeg"
	"os"

	insightface "github.com/younland/go-insightface"
)

func run() error {
	file, err := os.Open("face.jpg")
	if err != nil {
		return err
	}
	defer file.Close()

	img, err := jpeg.Decode(file)
	if err != nil {
		return err
	}

	engine, err := insightface.New(insightface.Config{
		RuntimeLibrary:  "/opt/onnxruntime/lib/libonnxruntime.so.1.23.2",
		DetectorModel:   "/opt/models/detector.onnx",
		RecognizerModel: "/opt/models/recognizer.onnx",
	})
	if err != nil {
		return err
	}
	defer engine.Close()

	faces, err := engine.Analyze(context.Background(), img)
	if err != nil {
		return err
	}
	for _, face := range faces {
		_ = face.Detection.Score
		_ = face.Embedding.Values() // 已拷贝、L2 归一化的 512 维向量
	}
	return nil
}
```

`Engine.Detect` 只检测；`Engine.Embed` 对指定 landmarks 对齐并提特征；`Engine.Analyze` 检测 + 对齐 + 特征一步到位。上下文取消在 Go 边界检查；已进入原生的推理无法被中断。

<a id="config"></a>

## ⚙️ 配置说明

`insightface.Config` 字段：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RuntimeLibrary` | string | —（必填） | ONNX Runtime 动态库路径 |
| `DetectorModel` | string | —（必填） | SCRFD 检测模型路径 |
| `RecognizerModel` | string | —（必填） | ArcFace 识别模型路径 |
| `DetectorInputWidth` / `DetectorInputHeight` | int | `640` / `640` | 检测输入尺寸，须为 32 的正整数倍 |
| `DetectionThreshold` | float32 | `0.5` | 检测得分阈值，取值 `[0,1]` |
| `NMSThreshold` | float32 | `0.4` | NMS IoU 阈值，取值 `[0,1]` |
| `IntraOpThreads` | int | ORT 默认 | 算子内并行线程数，`0` 用默认，不得为负 |
| `SessionConfig` | map[string]string | 关闭 intra-op 自旋 | 追加 ONNX Runtime 会话配置项（经 `AddSessionConfigEntry` 应用），同名键覆盖默认项；默认注入 `session.intra_op.allow_spinning=0`（空闲时不占用 CPU），键与值均须非空 |

`MatcherConfig` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `MinSimilarity` | float32 | 接受匹配的最小余弦相似度（示例值 `0.55`，须按部署数据标定） |
| `MinMargin` | float32 | Top-1 与次优异身份的相似度间隔下限（示例值 `0.05`） |

集成测试使用的环境变量（仅测试，非运行时配置）：

| 环境变量 | 作用 |
| --- | --- |
| `ORT_LIBRARY` | ONNX Runtime 动态库路径 |
| `DETECTOR_MODEL` / `RECOGNIZER_MODEL` | 外部检测 / 识别模型路径 |
| `FACE_IMAGE` | 端到端测试人脸照片 |
| `REFERENCE_RESULT` | 可选，独立实现参考结果 JSON，余弦相似度须 ≥ 0.999 |
| `ORT_TEST_MODEL` | 单输入单输出小模型路径（`internal/ort` 与 `runtimefs` 集成测试） |

<a id="assets"></a>

## 📦 资产供给

模型与动态库有三种供给方式，权重一律由调用方自备：

| 方式 | 适用场景 | 入口 |
| --- | --- | --- |
| 普通文件路径 | 最简单，直接传绝对路径 | `insightface.New` |
| 构建期下载校验 | `go generate` / Makefile / 容器构建 | `cmd/insightface-assets` |
| 调用方嵌入 | 单二进制分发 | `insightface.NewFromFS` + `runtimefs` |

### 构建期下载

在 manifest 中声明有授权的来源与 SHA-256：

```json
{
  "detector": {
    "url": "https://models.example/detector.onnx",
    "sha256": "<64 lowercase hex characters>"
  },
  "recognizer": {
    "url": "https://models.example/recognizer.onnx",
    "sha256": "<64 lowercase hex characters>"
  }
}
```

在已 `go get` 本模块的工程内执行（不带 `@version`，由工程的 go.mod 解析）：

```bash
go run github.com/younland/go-insightface/cmd/insightface-assets \
  -manifest insightface-assets.json \
  -output ./faceassets/files
```

> [!NOTE]
> 只有包含 younland/onnxruntime 迁移的提交才能在 Windows 上编译（v0.2.0 及更早的
> tag、以及迁移前推送的 main 历史均含旧依赖）；Windows 用户请等待改名后的新 tag
> 或使用本地路径 `replace`，Linux/macOS 不受影响。历史 tag 为旧模块路径
> `github.com/lib-x/go-insightface`，对其使用 `@version` 运行会报路径不匹配。

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `-manifest` | 是 | 资产 manifest JSON 路径 |
| `-output` | 是 | 校验后输出目录（须专用于本工具） |
| `-cache` | 否 | 下载缓存目录 |
| `-os` / `-arch` | 否 | 目标平台，默认当前 `GOOS`/`GOARCH` |
| `-timeout` | 否 | 下载总时限，默认 30 分钟 |

省略 `runtime` 条目时，工具自动选择 pinned 的官方 ONNX Runtime 1.23.2 压缩包；整包校验通过后以一次序列化发布操作替换输出目录。模型 URL 与 SHA-256 始终由调用方提供。详见 [asset provisioning](docs/assets.md)。

### 调用方嵌入

嵌入只发生在调用方应用中；动态库必须物化为真实文件（操作系统加载器要求路径），模型可直接从 `fs.FS` 流式读取：

```go
import (
    "context"
    "embed"

    insightface "github.com/younland/go-insightface"
    "github.com/younland/go-insightface/runtimefs"
)

//go:embed files
var bundled embed.FS

func newEngine(ctx context.Context) (*insightface.Engine, error) {
    source, err := runtimefs.New(bundled, "")
    if err != nil {
        return nil, err
    }
    return insightface.NewFromFS(ctx, insightface.Config{
        RuntimeLibrary:  "files/runtime/libonnxruntime.so.1.23.2",
        DetectorModel:   "files/models/detector.onnx",
        RecognizerModel: "files/models/recognizer.onnx",
    }, source)
}
```

`runtimefs` 会把动态库物化到内容寻址的用户缓存。只用 `New` 的应用不需要它。

<a id="details"></a>

## 🧩 功能细节

### Execution Provider 策略

| 场景 | 行为 |
| --- | --- |
| CPU 构建 | 零配置直接可用，基线路径 |
| GPU 构建（含 TensorRT/CUDA EP） | 自动按 TensorRT → CUDA 挂载，最大化算力 |
| GPU 挂载失败（缺驱动/CUDA 运行库） | 记录日志，自动回退 CPU 重建会话 |
| 其他 provider（如 Azure EP） | 不在自动名单内，忽略 |

启用 GPU 后会话加载日志会输出实际使用的 provider 列表；强制 CPU 推理请使用 CPU 构建的 ONNX Runtime。

### 身份匹配

```go
matcher, err := insightface.NewMatcher(insightface.MatcherConfig{
	MinSimilarity: 0.55, // 示例值，须按部署数据标定
	MinMargin:     0.05,
}, gallery) // []GalleryIdentity{{ID: "person-a", Samples: []insightface.Embedding{...}}}
if err != nil {
	return err
}

result, err := matcher.Match(query)
if err != nil {
	return err
}
if result.Accepted {
	_ = result.Best.Identity
}
```

`ReplaceGallery` 以原子快照热更新图库；`MatchVector` 接受原始向量并自动归一化。

> [!IMPORTANT]
> 余弦相似度不是置信度或概率。上述阈值仅为示例，必须用部署场景数据标定 FPIR/FNIR 后再上线。换用不同识别模型后，历史 embedding 不能数学换算到新模型空间，需从留存的源图像重新注册。

### 性能基准

| 项目 | 数值 | 环境 |
| --- | --- | --- |
| 精确匹配 3,000 身份 × 3 样本 × 512 维 | 约 `5.11 ms/op`、`0 B/op`、`0 allocs/op` | Linux AMD64，i5-12500H（见 docs/validation.md） |

<a id="dev"></a>

## 🛠️ 开发指南

```bash
go test ./...          # 单元测试
go test -race ./...    # 竞态检测
go vet ./...           # 静态检查
go test -run '^$' -bench . -benchmem ./...   # 基准
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build ./...   # 交叉编译验证
```

目录结构：

```text
├── assets/                  # 构建期资产下载与校验（manifest、SHA-256、发布锁）
├── cmd/insightface-assets/  # 资产下载 CLI
├── internal/
│   ├── arcface/             # ArcFace 对齐 + 特征提取
│   ├── imageutil/           # 图像预处理与张量化
│   ├── ort/                 # ONNX Runtime 适配层（版本协商、provider 自动启用）
│   └── scrfd/               # SCRFD 检测解码（anchor、NMS、landmarks）
├── runtimefs/               # 内容寻址缓存，物化嵌入的动态库
├── docs/                    # 资产、部署、模型契约、规格、验证记录
├── engine.go                # 公共 Engine API（New / NewFromFS / Detect / Embed / Analyze）
├── matcher.go               # 身份级精确匹配
└── embedding.go             # 不可变 L2 归一化 embedding
```

CI（GitHub Actions，Ubuntu）执行 `-race` 测试、vet、基准与 Linux ARM64 交叉编译。

集成测试模型从 `yakhyo/face-reidentification` v0.0.1 release 下载，下载后校验 SHA-256：

```bash
mkdir -p .test-assets/models
curl -fL \
  -o .test-assets/models/det_10g.onnx \
  https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx
curl -fL \
  -o .test-assets/models/w600k_r50.onnx \
  https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx

printf '%s  %s\n' \
  5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91 \
  .test-assets/models/det_10g.onnx \
  4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43 \
  .test-assets/models/w600k_r50.onnx | sha256sum --check
```

端到端测试：

```bash
ORT_LIBRARY=/path/libonnxruntime.so.1.23.2 \
DETECTOR_MODEL="$PWD/.test-assets/models/det_10g.onnx" \
RECOGNIZER_MODEL="$PWD/.test-assets/models/w600k_r50.onnx" \
FACE_IMAGE=/path/face.jpg \
go test -tags=integration -run TestEngineRunsLicensedExternalModels .
```

> [!WARNING]
> 上述外部权重为 InsightFace 衍生的非商业研究模型，下载来源未授予生产使用权。命令仅用于测试复现，商业、政府、军事或生产用途须先取得覆盖该用途的明确授权。

<a id="license"></a>

## ⚠️ 许可与敏感部署

库代码以 MIT 协议开源；模型权重有各自独立的许可条款。人脸图像与 embedding 属生物识别数据：静态与传输加密、访问受限、审计注册与查询操作、制定留存/删除策略、避免记录图像字节或 embedding。集成前请阅读[部署与运维指引](docs/deployment.md)。

## 📜 开源协议

[MIT License](LICENSE) © 2026 lib-x contributors
