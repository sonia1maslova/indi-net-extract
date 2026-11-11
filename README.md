# IndiNetExtract

Python分发名（安装名）：`indi-net-extract`。Python导入名和源码包目录：`indinetextract`。

模板引导的个体网络提取工具包，开发预览版 `0.1.0.dev0`，已发布到[PyPI](https://pypi.org/project/indi-net-extract/0.1.0.dev0/)。

输入被试ID列表、逐项对应的静息态影像路径，以及全体共用的模板和mask。输出每名被试的连续网络图、时间序列及运行记录。支持经典双回归β和独立NumPy/SciPy GIG-ICA。

**每名被试只输入一份已经准备好的4D NIfTI。** 数据清洗、扫描标准化、拼接和空间配准在包外完成。批处理列表表示不同被试，不表示要拼接的扫描。算法内部所需的中心化、白化和双回归时间序列归一化仍保留。

## 安装

安装当前开发预览版：

```bash
python -m pip install "indi-net-extract==0.1.0.dev0"
```

也可以从本仓库安装源码：

```bash
git clone https://github.com/sonia1maslova/indi-net-extract.git
cd indi-net-extract
python -m pip install .
```

## 快速使用

在已具备NumPy、SciPy、NiBabel、threadpoolctl的环境中，将本包 `src` 加入 `PYTHONPATH`，或安装本地构建产物。已测试的软件版本和验证边界见 `docs/validation.md`；以下示例不下载或修改数据。

安装已下载的wheel时，文件名中的分发名按打包规则写成 `indi_net_extract`：

```bash
python -m pip install /path/to/indi_net_extract-0.1.0.dev0-py3-none-any.whl
```

安装后仍使用 `from indinetextract import ...`。分发名中的连字符不用于Python导入语句。

```python
from indinetextract import extract_batch

def main():
    batch = extract_batch(
        subject_ids=["sub-001", "sub-002"],
        fmri_paths=[
            "/data/sub-001/rest_preprocessed.nii.gz",
            "/data/sub-002/rest_preprocessed.nii.gz",
        ],
        templates="/data/templates.nii.gz",
        mask="/data/analysis_mask.nii.gz",
        output_root="/results/gig_ica",
        method="gig_ica",
        n_jobs=2,
        n_threads=2,
    )
    print(batch.summary)

if __name__ == "__main__":
    main()
```

并行采用独立进程，脚本需要上面的 `__main__` 保护。只处理一名时，也传入长度为1的两个列表。输入文件名不必含被试ID；包按列表位置配对，不推测文件属于谁。

| 参数 | 含义 |
| --- | --- |
| `subject_ids` | 非空且唯一的字符串ID列表 |
| `fmri_paths` | 等长路径列表，每条路径是一名被试的一份4D影像 |
| `templates` | 同一份4D空间模板，第四维为K个成分 |
| `mask` | 同一份3D二值分析mask |
| `method` | `"dual_regression"` 或 `"gig_ica"` |
| `n_jobs=1` | 最多同时处理的被试数，即进程数 |
| `n_threads=2` | 每名被试数值计算的线程预算 |
| `label="extraction"` | 被试目录下的本次输出标签；可用不同标签区分方法、模板或测量日 |
| `on_error="continue"` | 记录失败后继续；`"raise"` 表示停止提交新被试，等待正在运行的被试结束后报错 |
| `work_dir=None` | 可选临时存储目录；每名被试使用独立临时子目录 |

例如 `n_jobs=3, n_threads=2` 表示最多3名被试同时拟合，每名的数值运算预算为2线程。它不保证总内存足够；GIG-ICA仍需白化和SVD工作内存。默认串行执行，不自动增加进程。

## 输出

```text
output_root/
  _batches/<batch_id>/
    inputs.tsv                   # 保存原ID—影像配对
    summary.tsv                  # 顺序与输入一致，包含状态及失败原因
    metadata.json                # 共用设置、环境及源码指纹
  sub-001/extraction/
    continuous_components.nii.gz  # X×Y×Z×K，连续成分
    timecourses.tsv              # T×K，行是时间点，无标题行
    components.tsv               # 0起始图像索引与成分ID
    mask.nii.gz
    metadata.json
  sub-002/extraction/
    ...
```

连续图的第四维是**成分**，不是时间；第i张图对应第i个模板。默认成分ID为C1…CK。双回归输出空间β，GIG-ICA输出成分权重，两者不当作统计Z或概率。

一名完成后立即保存。`BatchResult`只返回轻量记录和文件路径，不把全体影像留在内存。失败被试不产生一个伪成功的结果目录。默认不覆盖已有结果；显式调用结果对象的 `save(overwrite=True)` 时，旧目录改名为备份并保留。

## 独立后处理与模板Dice

已经提取好的连续图可以重新载入，改变平滑或Top-N时不用重新拟合：

```python
from indinetextract import NetworkResult, postprocess_maps, template_dice, overlap_fraction

fit = NetworkResult.load("/results/gig_ica/sub-001/extraction")
binary = postprocess_maps(
    fit,
    fwhm_mm=6,
    count_reference="/data/fixed_binary_reference.nii.gz",
)
binary.save("/results/gig_ica", label="fwhm_6mm")

dice = template_dice(binary, reference="/data/fixed_binary_reference.nii.gz")
print(dice.per_component, dice.mean_dice)
dice.save("/results/gig_ica", label="fwhm_6mm/template_dice")

pleiotropy = overlap_fraction(binary)  # 3D NIfTI对象，值为所选成分比例
```

- `count_reference`必须是二值图，在mask内计数得到每成分N。也可以改用显式的 `top_n=[...]`，两者必须二选一。
- 用于提取的模板允许连续值；用于计数和Dice的参考图必须明确为0/1。切换提取模板不自动改变N。
- 平滑使用 `G(M*f)/G(M)`，仅沿空间维，保留原mask。每个宽度都从原始连续图开始。NIfTI空间单位unknown时按mm解释并记录；正宽度平滑拒绝剪切网格。
- Top-N按有限的有符号数值降序选严格N个，允许零值和负值；并列按原始Fortran体素索引。候选不足时报错，不自动缩小N。
- Dice矩阵行是个体成分，列是参考成分；对应分数是对角线。不自动换序、翻符号或选择最佳配对。
- 单侧为空时Dice为0，双侧为空为NaN；存在无效对应成分时完整K成分均值为NaN。JSON中缺失数值保存为null，TSV中为nan。
- `overlap_fraction`为K张二值图的逐体素平均；它不自动构成统计显著性检验。

## 输入约束与验证边界

所有影像必须事先对齐，空间形状一致，affine数值在记录的容差内一致。mask显式提供，保持不变。mask外不参与计算；mask内NaN/Inf或时间常量BOLD序列会报错，包不自动插补或删除体素。

独立Python GIG核心保留原算法源码许可与控制流程。此前合成数据及三名真实被试有MATLAB对照，真实数据不是所有数值严格相等。此包封装的验证范围和具体检查见 [docs/validation.md](docs/validation.md)；不能只依据模板Dice较高就判定个体网络科学有效。

当前实际验证使用Python3.13项目环境；其他Python版本和Windows原生并行尚需分别测试。声明最低Python版本不等于所有组合均已测试。

自动单元测试不依赖MATLAB或FSL；本轮另外完成了两种模板的小型FSL对照。更新算法一致性结论时，应保留输入、源码、参数和数值误差。

## 来源与许可

GIG数值核心位于 `_algorithms/gig_ica.py`，来自已有NumPy/SciPy移植；原始来源为GIFT `icatb_gigicar.m`，作者Yuhui Du、Yong Fan。版权与许可保留在文件头及 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。其他初版代码采用BSD-3-Clause许可。
