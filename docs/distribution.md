# 构建与检查分发文件

源码发行包（sdist，扩展名为 `.tar.gz`）保留源码和开发附件。安装包（wheel，扩展名为 `.whl`）用于安装运行库。两者用途不同，因此包含的文件也不同。

## 源码包附件清单

`MANIFEST.in` 指定需要额外收入源码包的文件：

| 文件或目录 | 内容 |
| --- | --- |
| `README.md`、`CHANGELOG.md` | 使用入口和更新记录 |
| `docs/methods.md` | 方法与参数说明 |
| `docs/validation.md` | 已完成的验证及其边界 |
| `docs/distribution.md` | 本构建说明 |
| `examples/*.py` | 两个使用示例，需要用户提供输入影像和输出路径 |
| `tests/*.py` | 四个测试模块、`helpers.py`、`__init__.py` 和 `validate_distribution.py` |

`pyproject.toml` 声明运行依赖、构建依赖和许可证；构建工具同时收入 `src/indinetextract/` 下的运行源码。`LICENSE` 和 `THIRD_PARTY_NOTICES.md` 由 `project.license-files` 指定，同时进入源码包和 wheel。

wheel 包含运行库、分发元数据和许可证文件。文档、示例和测试保留在源码包中，不随 wheel 安装。

## 构建和格式检查

在维护者的开发环境中安装构建工具，然后从源码根目录执行：

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

构建依赖要求 `setuptools>=77.0.3`，以支持标准许可证标识（SPDX 表达式）和 `project.license-files`。当前许可证表达式为 `BSD-3-Clause`。配置格式更新不改变许可证正文和第三方声明。

默认的 `python -m build` 先生成源码包，再从源码包构建 wheel。已有构建依赖的受控环境可使用 `python -m build --no-isolation`，直接使用当前环境中的工具。

`twine check` 检查分发元数据和 README 格式，不验证算法，也不会上传文件。

## 从源码包复查

将生成的 `.tar.gz` 解压到一个新目录，进入包含 `pyproject.toml` 的目录，在已安装运行依赖的环境中执行：

```bash
python -m pip install .
python -m unittest discover -s tests -t . -v
```

测试使用合成数据，不需要研究影像、MATLAB 或 FSL。命令在解压目录中执行，可以检查源码包是否带齐测试模块及其辅助文件。

若需要检查 wheel 中的两进程 GIG-ICA 和结果保存/载入，先安装对应 wheel，再从同一解压目录执行以下命令；输出目录必须尚不存在：

```bash
python -m tests.validate_distribution --output-root ./synthetic-distribution-check
```

此命令默认只用合成数据。可选参数 `--fsl-dir` 用于额外的 FSL 对照，具体范围见 [验证说明](validation.md)。构建、格式检查、单元测试和上传是不同步骤；上述命令均不发布到 PyPI。
