# MiniMax H3 @ RX 7900 XT (20GB) Windows 11 部署与提速全记录

> **Languages**: [中文](#中文) | [English](#english)

---

<a id="中文"></a>
# 中文

> **致谢**：本项目基于原教程作者的开仓教程 [a756598009-CMYK/MiniMax-H3-AMD7900XTX-Win11](https://github.com/a756598009-CMYK/MiniMax-H3-AMD7900XTX-Win11) 完成部署，模型整合包与工作流思路均来自作者，在此致敬。本文档记录的是在其基础上的提速调优与排障经验。

> **最新成绩（2026-09-21，热态实测）**
>
> | 任务 | 参数 | 总耗时 |
> |---|---|---|
> | 文生视频 T2V | 608×352 / 124帧 / 6步 | **90 秒** |
> | 参考图生视频 R2V（双参考图） | 608×352 / 124帧 / 6步 / max保脸 | **105~142 秒** |
> | 参考图生视频 R2V（单参考图） | 608×352 / 124帧 / 6步 / max保脸 | **178 秒**（基准） |
>
> 同日起点为 11~15 分钟（I2V）/ 34 分钟（R2V），总提速约 **7~14 倍**。

---

## 一、硬件与环境

| 项目 | 配置 |
|---|---|
| 显卡 | AMD Radeon RX 7900 XT（gfx1100，20GB） |
| 内存 | 32GB |
| 硬盘 | Fanxiang S790C 1TB NVMe ×2（C/D 各一） |
| 虚拟内存 | C+D 各 64GB，共 128GB（≥教程要求的 124GB） |
| 系统 | Windows 11 |
| 部署目录 | `D:\ComfyUI_windows_portable_amd`（官方 AMD 便携版 v0.34.0） |
| 模型库 | `D:\ai\ComfyUI_models`（独立目录，经 `extra_model_paths.yaml` 挂载） |

## 二、最终技术栈（提速的核心）

| 组件 | 版本 | 说明 |
|---|---|---|
| PyTorch | **2.15.0a0+rocm10.1 nightly**（gfx1100 专用） | 来自 AMD 官方 nightly 源，是最大的单项提速 |
| ROCm | 10.1（HIP 7.16） | 随 torch nightly 以 pip 包形式安装 |
| sage-attention | **2.2 自动调优版**（patientx 编译） | 注意力加速；已锁定冠军配置跳过逐形状跑分（见第十节） |
| Triton | triton-windows 3.7.1 | 自定义内核编译器 |
| INT8 加速 | ComfyUI-INT8-Fast-ROCM（patientx） | H3 INT8 模型的 WMMA/hipBLASLt INT8 GEMM |
| Spectrum | ComfyUI-Spectrum-MiniMax-H3（xmarre） | 切比雪夫特征预测，跳过约 1/3 采样步的 transformer 计算 |
| ComfyUI-Manager | 4.2.2（pip 预发布版 + `--enable-manager`） | 节点管理 |

### 安装命令（关键，全网教程大多写错或缺失）

```bat
:: 1. nightly torch 全家桶（不能用 PyPI，必须用 AMD 官方源；直接下 whl 文件会缺 rocm 元包）
python_embeded\python.exe -m pip install -U --pre "torch[device-gfx1100]" "torchvision[device-gfx1100]" torchaudio rocm-sdk-devel --index-url https://nightly.repo.amd.com/rocm/whl-next/

:: 2. sage-attention 2.2（需要 nightly torch 提供的 torch._inductor.kernel.custom_op，稳定版 2.9 会导入失败）
python_embeded\python.exe -m pip install --force-reinstall --no-deps "https://github.com/patientx/sageattention-autotune/releases/download/0908/sageattention-2.2.0-py3-none-any.whl"

:: 3. Triton 与 INT8 节点
python_embeded\python.exe -m pip install -U "triton-windows>=3.7,<3.8"
git clone https://github.com/patientx/ComfyUI-INT8-Fast-ROCM ComfyUI/custom_nodes/ComfyUI-INT8-Fast-ROCM
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3
```

## 三、启动参数（`D:\ComfyUI_windows_portable_amd\start_comfyui.bat`）

```bat
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-pinned-memory --disable-mmap --reserve-vram 1.0 --enable-manager
```

| 参数 | 作用 |
|---|---|
| `--use-sage-attention` | 启用 sage 注意力（核心加速） |
| `--disable-mmap` | 规避 safetensors 内存映射读取时的 access violation 崩溃（本机实测连续三次崩溃的规避方案） |
| `--disable-pinned-memory` | 配合上一条，降低大模型加载期的内存峰值 |
| `--reserve-vram 1.0` | 20GB 卡只预留 1GB，多挤 1GB 装模型，减少每步 PCIe 搬运（默认教程值 2.0 偏大） |
| `--enable-manager` | 启用 Manager |

> 日志用 Git 自带 tee 双写：`... 2>&1 | "C:\Program Files\Git\usr\bin\tee.exe" comfyui-run.log`

## 四、模型清单（H3 R2V 五件套，均放对应目录）

- `diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors`（R2V 主模型）
- `diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors`（I2V 主模型）
- `text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `vae\minimax_h3_video_vae_fp16.safetensors` + `vae\minimax_h3_audio_vae_fp32.safetensors`
- `loras\minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`（Turbo LoRA）

## 五、工作流提速配置要点（参考图生视频 R2V）

1. **加载器用 `MiniMaxH3INT8FastLoader`**（INT8-Fast-ROCM 的 minimax h3 预设）—— INT8 模型直读 + WMMA 加速
2. **Turbo LoRA + 6 步**：6 步是甜点位。实测 10 步多花 67% 采样时间，且 turbo 蒸馏 LoRA 步数过多会过曝偏色，画质反而变差
3. **`ref_image_size=max`：保脸的关键**。match 模式会把参考图压到生成分辨率（几十万像素），脸必崩；max 用 2048px 短边保身份，代价是慢，但值得
4. **Spectrum 节点串在 LoRA 之后**：约 1/3 步数用预测代替实算；若发现画面细节变差，节点右键 Bypass 即可回退
5. **随机种子设为 randomize**：ComfyUI 有增量执行——参考图和提示词不变时，重跑自动跳过整个编码阶段（省掉最大的单块时间）。种子随机保证每次出新片
6. 分辨率 608×352、124 帧（24fps ≈ 5 秒）是 20GB 显存的甜点位

## 六、参数安全区（20GB 显存，2026-09-21 实测）

| 参数组合 | 结果 |
|---|---|
| 608×352 / 124帧 / 6步 | ✅ 安全甜点位，T2V 90 秒、R2V 105~178 秒 |
| 480×832 竖屏 / 124帧 / 10步 | ⚠️ 边缘：总耗时 10~12 分钟，仅解码就约 5 分钟 |
| 480×832 竖屏 / 124帧（连续多跑） | ❌ 实测触发 VAE 解码阶段**真死锁**（见第九节），需重启进程 |
| 步数 6 → 10 | ❌ 多花 67% 时间，turbo LoRA 下过曝偏色，无画质收益 |

**经验法则**：解码阶段中间激活值与像素数成正比。480×832 比 608×352 多 87% 像素，采样、解码全部同比变慢，且解码显存峰值逼近动态调度器能腾出的上限。想拍竖屏，减帧数（61/89 帧）比硬顶 124 帧稳。

## 七、速度演进实测（同一台 7900 XT）

| 阶段 | 配置 | R2V 总耗时 | 采样速度 |
|---|---|---|---|
| 起步 | 稳定版 torch + pytorch 注意力 | 34 分钟 | 117 秒/步 |
| +sage 1.0.6、reserve-vram 1.0 | | ~26 分钟 | 55~73 秒/步 |
| +Spectrum | | 22 分钟 | — |
| +nightly torch + sage 2.2 | | 178 秒 | 9~17 秒/步 |
| **最终（热态连跑）** | 同上 | **R2V 双图 105~142 秒 / T2V 90 秒** | **7.2~13.7 秒/步** |

## 八、冷态与热态——"突然变慢"的最大误会来源（2026-09-21 全天追查结论）

同一天内实测到同一条基准任务 178 秒 → 589 秒 → 又回到 105 秒。最终定位**机器没有任何硬件/驱动退化**，波动全部来自冷态成本：

1. **冷加载**：重启 ComfyUI（或长时间不用模型被卸出）后，第一趟要从硬盘重读约 35GB（19.5GB 主模型 + 15GB 文本编码器 + 双 VAE），再加 41 秒 Model 初始化。这一趟 8~10 分钟是正常的
2. **磁盘缓存被洗掉**：短时间内下载/删除几十 GB 文件（例如试新模型）、Windows 大版本更新后的开机后台整理，都会把文件缓存冲掉。此时实测 python 冷读只剩 **74 MB/s**（正常 1500+ MB/s），加载时间翻 10 倍
3. **磁盘被抢**：下载、哈希校验、杀毒扫描等并发磁盘活动会把模型加载拖到几分钟
4. **热态才是真实速度**：模型常驻后连跑，采样 7~14 秒/步、编码可完全跳过，T2V 90 秒一条

**结论**：判断快慢要看「连跑第二条」的耗时；重启后第一条慢 ≠ 机器变慢。

## 九、真卡死 vs 假卡死——三分钟鉴别法

ComfyUI 的 tqdm 进度条有缓冲、VAE 解码阶段**完全不打印进度**，日志长时间沉默很常见。鉴别方法（PowerShell 性能计数器）：

| 状态 | GPU（python compute） | CPU 增量 | 磁盘 | 结论 |
|---|---|---|---|---|
| 编码/加载中 | ~0% | 有 | 持续读取 | 假卡死，等 |
| 采样中 | 95~100% 持续 | 低 | ~0 | 正常 |
| **VAE 解码中** | **180~340% 满负荷** | 单核忙 | ~0 | **假卡死**（日志无进度条，480×832 解码约 5 分钟，等） |
| **真死锁** | **0%** | **0.00s** | **0 MB/s** | 真死，杀进程重启 |

```powershell
# GPU 占用（PID 用 netstat -ano | findstr :8188 查）
(Get-Counter "\GPU Engine(pid_<PID>*)\Utilization Percentage").CounterSamples | ? {$_.CookedValue -gt 1}

# CPU 增量：两次取样差值，0.00s = 线程全部干等 = 真死锁
```

真死锁处理：`taskkill /PID <PID> /F`，双击桌面「ComfyUI-7900XT启动」重启（约 20~40 秒上线）。

## 十、采样初始化卡死——根因与修复（2026-09-12 晚，已修复）

### 症状

- 任务停在 `0/6 [00:00, Model Initializing...]`，日志 5 分钟以上不更新
- 查 GPU 占用发现占用来自 dwm / 壁纸程序，python 完全不动 = 真卡死
- 另一种形式：`Error running sage attention: CUDA error: unspecified launch failure`，之后一切任务都失败

### 根因

sage-attention 2.2 自动调优版对**每种新张量形状**（分辨率/帧数/是否带参考视频变了就算新形状）要逐套跑分 12 个候选 Triton 内核。个别候选在 gfx1100 上会把 GPU 跑死：跑分永不返回 → 挂起；或直接 launch failure → 进程毒化。更麻烦的是跑分结果会写盘到 `C:\Users\<用户>\.cache\sageattention\autotune_cache.pkl`，崩溃前写入的"带毒"配置会让之后**每次都在同一位置挂**。

### 修复（已实施，速度无损）

把候选配置锁死为实测冠军 `(32, 16, 2, 2)`（磁盘缓存里 60 条形状记录的绝大多数冠军都是它）：

- 改动文件：`python_embeded\Lib\site-packages\sageattention\triton_autotune.py`，原件备份在同目录 `triton_autotune.py.bak`
- 锁定后 `_eager_autotune_select` 直接短路返回，**永不再跑分**——卡死源从机制上消除，且换新形状时省掉 1~2 分钟跑分
- 同时清除了带毒缓存：`autotune_cache.pkl` 和 `C:\Users\<用户>\.triton\cache`（清后首次运行需重新编译内核几分钟，一次性）

### 若再遇卡死，自助三板斧

```bat
:: 1. 杀掉卡死进程
netstat -ano | findstr :8188 | findstr LISTENING
taskkill /PID <上面的PID> /F

:: 2. 删除可能带毒的缓存
del "C:\Users\<用户>\.cache\sageattention\autotune_cache.pkl"
rmdir /s /q "C:\Users\<用户>\.triton\cache"

:: 3. 双击桌面「ComfyUI-7900XT启动」快捷方式重启
```

> **注意**：pip 升级/重装 sageattention 会覆盖补丁，升级后需重新锁定配置（参照 `.bak` 对比）。

## 十一、使用纪律（保住热态速度）

- 编码阶段（提示词框）有几分钟无输出是正常的，**不要中途取消**——本机实测取消会让文本编码器卡死，只能重启
- 升级/换装后的第一次运行包含内核编译，会额外慢几分钟，属一次性成本
- 提示词和参考图定稿后，**连跑只换种子**，编码阶段耗时为 0，是最快的用法
- 跑片时避免同时下载/拷贝大文件，磁盘争用会直接拖慢模型加载
- 同一工作流连着跑；中途切工作流会触发模型卸载/重载

## 十二、失败的尝试（避免重复踩坑）

| 尝试 | 结论 |
|---|---|
| **FastH3 4 步蒸馏模型**（Kijai 版 21.8GB + 剥离门控 19.5GB） | 首跑 286 秒看似不错，但 20GB 卡上显存颠簸（二跑 235 秒/步），dense 模式提速有限，画质用户不接受，已全删（43GB）。**不推荐 20GB 卡尝试** |
| `PYTORCH_TUNABLEOP_ENABLED=1` | 调优全程零输出像死机，结果文件 0 字节后进程消失——不要用 |
| sage-attention 2.2 配 torch 2.9 稳定版 | 缺 `torch._inductor.kernel.custom_op` 导入失败，必须 nightly |
| 直接 pip 安装 whl 文件升级 torch | nightly torch 依赖的 `rocm==x.x.x` 元包只在 AMD 官方 index，必须用 `--index-url https://nightly.repo.amd.com/rocm/whl-next/` 整体解析 |
| `load_torch_file` access violation (0xC0000005) | 与模型文件、页面文件均无关；`--disable-mmap` 后稳定 |
| `MiniMaxH3INT8FastLoader` 全网搜不到 | 教程作者整合包内的改名封装，本体是 patientx/ComfyUI-INT8-Fast-ROCM 的 UNetLoaderINTW8A8（minimax h3 预设），自建别名节点解决 |

## 十三、回滚方法

```bat
:: 退回稳定版 torch（sage 2.2 会随之失效，需同时装回 1.0.6）
python_embeded\python.exe -m pip uninstall -y torch torchvision torchaudio rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-device-gfx1100 rocm-sdk-libraries rocm-bootstrap amd-torch-device-gfx1100 amd-torch-device-gfx110x amd-torchvision-device-gfx1100
python_embeded\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
python_embeded\python.exe -m pip install sageattention==1.0.6 --no-build-isolation
```

pip 包备份：`D:\ai\pip_freeze_backup.txt`；nightly 安装包留存：`D:\ai\nightly_wheels\`。

## 十四、时间线速查

| 日期 | 事件 |
|---|---|
| 2026-09-08 | 完成部署，打通 R2V 工作流；解决 ComfyUI-Manager 缺失、Easy-Use 扩展报错 |
| 2026-09-09 | 调优至 178 秒基准；整理 MP 像素-分辨率对照 |
| 2026-09-12 | 修复 sage 自动调优卡死（锁定冠军配置）；恢复快速启动模式 |
| 2026-09-21 | 全天追查"变慢"：证实机器无退化（磁盘 1526MB/s、热态 7.2s/it），定位为冷态成本 + 10 步 + 竖屏大分辨率叠加；试 FastH3 4 步模型后放弃并清除；最终热态成绩 T2V 90 秒、R2V 双图 105~142 秒 |

---
---

<a id="english"></a>
# English

> **Credits**: This project was deployed based on the open-source tutorial [a756598009-CMYK/MiniMax-H3-AMD7900XTX-Win11](https://github.com/a756598009-CMYK/MiniMax-H3-AMD7900XTX-Win11) by the original author. The model bundle and workflow ideas come from the author — full credit. This document records the speed-tuning and troubleshooting experience built on top of it.

> **Latest results (2026-09-21, warm-state, measured)**
>
> | Task | Parameters | Total time |
> |---|---|---|
> | Text-to-Video (T2V) | 608×352 / 124 frames / 6 steps | **90 s** |
> | Reference-to-Video (R2V, dual reference images) | 608×352 / 124 frames / 6 steps / max face-preservation | **105~142 s** |
> | Reference-to-Video (R2V, single reference image) | 608×352 / 124 frames / 6 steps / max face-preservation | **178 s** (baseline) |
>
> Starting point the same day: 11~15 min (I2V) / 34 min (R2V). Overall speedup: **7~14×**.

---

## 1. Hardware & Environment

| Item | Configuration |
|---|---|
| GPU | AMD Radeon RX 7900 XT (gfx1100, 20GB) |
| RAM | 32GB |
| Storage | Fanxiang S790C 1TB NVMe ×2 (one on C:, one on D:) |
| Virtual memory | 64GB on C: + 64GB on D: = 128GB (≥ the 124GB required by the tutorial) |
| OS | Windows 11 |
| Install directory | `D:\ComfyUI_windows_portable_amd` (official AMD portable v0.34.0) |
| Model library | `D:\ai\ComfyUI_models` (standalone directory, mounted via `extra_model_paths.yaml`) |

## 2. Final Tech Stack (the core of the speedup)

| Component | Version | Notes |
|---|---|---|
| PyTorch | **2.15.0a0+rocm10.1 nightly** (gfx1100-specific) | From AMD's official nightly repo; the single biggest speedup |
| ROCm | 10.1 (HIP 7.16) | Installed as pip packages alongside the torch nightly |
| sage-attention | **2.2 autotuned build** (compiled by patientx) | Attention acceleration; champion config locked to skip per-shape benchmarking (see Section 10) |
| Triton | triton-windows 3.7.1 | Custom kernel compiler |
| INT8 acceleration | ComfyUI-INT8-Fast-ROCM (patientx) | WMMA/hipBLASLt INT8 GEMM for H3 INT8 models |
| Spectrum | ComfyUI-Spectrum-MiniMax-H3 (xmarre) | Chebyshev feature prediction; skips transformer compute on ~1/3 of sampling steps |
| ComfyUI-Manager | 4.2.2 (pip pre-release + `--enable-manager`) | Node manager |

### Install commands (critical — most tutorials online get these wrong or omit them)

```bat
:: 1. Nightly torch stack (do NOT use PyPI; must use AMD's official repo. Installing whl files directly misses the rocm meta-packages)
python_embeded\python.exe -m pip install -U --pre "torch[device-gfx1100]" "torchvision[device-gfx1100]" torchaudio rocm-sdk-devel --index-url https://nightly.repo.amd.com/rocm/whl-next/

:: 2. sage-attention 2.2 (requires torch._inductor.kernel.custom_op from nightly torch; fails to import on stable 2.9)
python_embeded\python.exe -m pip install --force-reinstall --no-deps "https://github.com/patientx/sageattention-autotune/releases/download/0908/sageattention-2.2.0-py3-none-any.whl"

:: 3. Triton and INT8 nodes
python_embeded\python.exe -m pip install -U "triton-windows>=3.7,<3.8"
git clone https://github.com/patientx/ComfyUI-INT8-Fast-ROCM ComfyUI/custom_nodes/ComfyUI-INT8-Fast-ROCM
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3
```

## 3. Launch parameters (`D:\ComfyUI_windows_portable_amd\start_comfyui.bat`)

```bat
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-pinned-memory --disable-mmap --reserve-vram 1.0 --enable-manager
```

| Parameter | Purpose |
|---|---|
| `--use-sage-attention` | Enable sage attention (core acceleration) |
| `--disable-mmap` | Avoids access-violation crashes when safetensors are read via memory mapping (fix verified after three consecutive crashes) |
| `--disable-pinned-memory` | Works with the above; lowers memory peak during large model loading |
| `--reserve-vram 1.0` | On a 20GB card only 1GB is reserved — squeezes 1 extra GB for the model and reduces per-step PCIe transfers (the tutorial default of 2.0 is too conservative) |
| `--enable-manager` | Enable Manager |

> Log mirrored with Git's built-in tee: `... 2>&1 | "C:\Program Files\Git\usr\bin\tee.exe" comfyui-run.log`

## 4. Model List (H3 R2V five-piece set, placed in their respective directories)

- `diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors` (R2V main model)
- `diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors` (I2V main model)
- `text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `vae\minimax_h3_video_vae_fp16.safetensors` + `vae\minimax_h3_audio_vae_fp32.safetensors`
- `loras\minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (Turbo LoRA)

## 5. Workflow Speed Configuration Essentials (Reference-to-Video R2V)

1. **Use the `MiniMaxH3INT8FastLoader`** (the minimax h3 preset from INT8-Fast-ROCM) — direct INT8 model loading + WMMA acceleration
2. **Turbo LoRA + 6 steps**: 6 steps is the sweet spot. Measured: 10 steps cost 67% more sampling time, and with a turbo distilled LoRA too many steps cause overexposure and color shift — quality actually gets worse
3. **`ref_image_size=max`: the key to preserving faces**. `match` mode compresses the reference image to the generation resolution (a few hundred thousand pixels) and faces always collapse; `max` keeps identity with a 2048px short edge — slower, but worth it
4. **Place the Spectrum node after the LoRA**: ~1/3 of steps use prediction instead of real computation; if you notice degraded detail, right-click → Bypass to roll back
5. **Set the seed to randomize**: ComfyUI has incremental execution — when the reference image and prompt are unchanged, reruns skip the entire encoding stage (the single biggest time block). A randomized seed guarantees a new clip every run
6. Resolution 608×352, 124 frames (24fps ≈ 5 s) is the sweet spot for a 20GB card

## 6. Parameter Safety Zone (20GB VRAM, measured 2026-09-21)

| Parameter combination | Result |
|---|---|
| 608×352 / 124 frames / 6 steps | ✅ Safe sweet spot: T2V 90 s, R2V 105~178 s |
| 480×832 portrait / 124 frames / 10 steps | ⚠️ Edge: 10~12 min total, ~5 min of that is decoding alone |
| 480×832 portrait / 124 frames (repeated runs) | ❌ Triggers a **true deadlock** in the VAE decode stage (see Section 9); process restart required |
| Steps 6 → 10 | ❌ 67% more time, overexposure/color shift under turbo LoRA, no quality gain |

**Rule of thumb**: intermediate activations in the decode stage scale with pixel count. 480×832 has 87% more pixels than 608×352 — sampling and decoding slow down proportionally, and the decode VRAM peak approaches the ceiling the dynamic scheduler can free up. For portrait video, reduce frame count (61/89 frames) instead of forcing 124.

## 7. Speed Evolution (same 7900 XT)

| Stage | Configuration | R2V total | Sampling speed |
|---|---|---|---|
| Starting point | Stable torch + PyTorch attention | 34 min | 117 s/step |
| +sage 1.0.6, reserve-vram 1.0 | | ~26 min | 55~73 s/step |
| +Spectrum | | 22 min | — |
| +nightly torch + sage 2.2 | | 178 s | 9~17 s/step |
| **Final (warm-state back-to-back runs)** | same as above | **R2V dual-ref 105~142 s / T2V 90 s** | **7.2~13.7 s/step** |

## 8. Cold State vs Warm State — the Biggest Source of "Suddenly Slow" (full-day investigation, 2026-09-21)

The same baseline task measured 178 s → 589 s → back to 105 s within a single day. Conclusion: **no hardware/driver degradation at all** — the variance came entirely from cold-state costs:

1. **Cold load**: after restarting ComfyUI (or when models were evicted after long idle), the first run re-reads ~35GB from disk (19.5GB main model + 15GB text encoder + dual VAE), plus 41 s of model initialization. An 8~10 min first run is normal
2. **Disk cache washed out**: downloading/deleting tens of GB in a short time (e.g., trying new models), or post-Windows-update background housekeeping, flushes the file cache. Measured cold python read then drops to **74 MB/s** (normal: 1500+ MB/s) — loading takes 10× longer
3. **Disk contention**: concurrent downloads, hash checks, antivirus scans drag model loading to several minutes
4. **Warm state is the real speed**: with models resident, back-to-back runs sample at 7~14 s/step and encoding is fully skipped — T2V in 90 s

**Conclusion**: judge speed by the *second consecutive run*; a slow first run after restart ≠ the machine got slower.

## 9. Real Freeze vs Fake Freeze — a 3-Minute Diagnostic

ComfyUI's tqdm progress bar is buffered and the VAE decode stage **prints no progress at all** — long log silences are common. Diagnostic method (PowerShell performance counters):

| State | GPU (python compute) | CPU delta | Disk | Verdict |
|---|---|---|---|---|
| Encoding / loading | ~0% | yes | sustained reads | Fake freeze — wait |
| Sampling | sustained 95~100% | low | ~0 | Normal |
| **VAE decoding** | **saturated 180~340%** | single core busy | ~0 | **Fake freeze** (no progress in log; 480×832 decode ≈ 5 min — wait) |
| **True deadlock** | **0%** | **0.00s** | **0 MB/s** | Truly dead — kill and restart |

```powershell
# GPU usage (find PID with netstat -ano | findstr :8188)
(Get-Counter "\GPU Engine(pid_<PID>*)\Utilization Percentage").CounterSamples | ? {$_.CookedValue -gt 1}

# CPU delta: difference between two samples; 0.00s = all threads idle = true deadlock
```

True deadlock handling: `taskkill /PID <PID> /F`, then double-click the desktop shortcut "ComfyUI-7900XT启动" to restart (online again in ~20~40 s).

## 10. Sampling-Initialization Freeze — Root Cause and Fix (night of 2026-09-12, fixed)

### Symptoms

- Task stuck at `0/6 [00:00, Model Initializing...]`, log silent for 5+ minutes
- GPU usage (when checked) belongs to dwm / wallpaper software; python completely idle = true freeze
- Another form: `Error running sage attention: CUDA error: unspecified launch failure`, after which every task fails

### Root cause

sage-attention 2.2 (autotuned build) benchmarks 12 candidate Triton kernels **for every new tensor shape** (changing resolution/frame count/reference-video presence counts as a new shape). Some candidates hang the GPU on gfx1100: a benchmark that never returns → hang; or a launch failure → poisoned process. Worse, benchmark results are written to disk at `C:\Users\<user>\.cache\sageattention\autotune_cache.pkl` — a "poisoned" config written before a crash makes it **hang at the same spot on every subsequent run**.

### Fix (implemented, zero speed loss)

Lock the candidate config to the measured champion `(32, 16, 2, 2)` (the vast majority of the 60 shape records in the on-disk cache name it champion):

- Modified file: `python_embeded\Lib\site-packages\sageattention\triton_autotune.py`; original backed up in the same directory as `triton_autotune.py.bak`
- After locking, `_eager_autotune_select` short-circuits and returns immediately — **benchmarks never run again**: the freeze source is eliminated mechanically, and new shapes save the 1~2 min benchmarking time
- Also cleared the poisoned caches: `autotune_cache.pkl` and `C:\Users\<user>\.triton\cache` (first run after clearing recompiles kernels for a few minutes — one-time cost)

### If a freeze happens again — self-service three-step fix

```bat
:: 1. Kill the frozen process
netstat -ano | findstr :8188 | findstr LISTENING
taskkill /PID <PID from above> /F

:: 2. Delete potentially poisoned caches
del "C:\Users\<user>\.cache\sageattention\autotune_cache.pkl"
rmdir /s /q "C:\Users\<user>\.triton\cache"

:: 3. Double-click the desktop shortcut "ComfyUI-7900XT启动" to restart
```

> **Note**: upgrading/reinstalling sageattention via pip overwrites the patch — re-lock the config after upgrades (compare against the `.bak`).

## 11. Usage Discipline (keep the warm-state speed)

- The encoding stage (prompt box) may produce no output for several minutes — **do not cancel mid-way**: canceling has been observed to freeze the text encoder, requiring a restart
- The first run after an upgrade/swap includes kernel compilation and is a few minutes slower — a one-time cost
- Once the prompt and reference image are finalized, **run back-to-back changing only the seed**: encoding time drops to 0 — the fastest usage pattern
- Avoid downloading/copying large files during renders — disk contention directly slows model loading
- Run the same workflow consecutively; switching workflows mid-session triggers model unload/reload

## 12. Failed Attempts (so you don't repeat them)

| Attempt | Conclusion |
|---|---|
| **FastH3 4-step distilled model** (Kijai build 21.8GB + gate-stripped 19.5GB) | First run 286 s looked decent, but VRAM thrashing on a 20GB card (second run 235 s/step), limited dense-mode gains, unacceptable quality to the user — deleted entirely (43GB). **Not recommended for 20GB cards** |
| `PYTORCH_TUNABLEOP_ENABLED=1` | Zero output throughout like a freeze; result file 0 bytes then the process vanished — do not use |
| sage-attention 2.2 with stable torch 2.9 | Missing `torch._inductor.kernel.custom_op` → import failure; nightly required |
| Upgrading torch by installing whl files directly with pip | The `rocm==x.x.x` meta-package required by nightly torch exists only on AMD's official index — must resolve everything with `--index-url https://nightly.repo.amd.com/rocm/whl-next/` |
| `load_torch_file` access violation (0xC0000005) | Unrelated to model files and pagefiles; stable since `--disable-mmap` |
| `MiniMaxH3INT8FastLoader` ungooglable | A renamed wrapper inside the tutorial author's bundle; the underlying node is UNetLoaderINTW8A8 (minimax h3 preset) from patientx/ComfyUI-INT8-Fast-ROCM — solved with a custom alias node |

## 13. Rollback Procedure

```bat
:: Roll back to stable torch (sage 2.2 stops working; reinstall 1.0.6 as well)
python_embeded\python.exe -m pip uninstall -y torch torchvision torchaudio rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-device-gfx1100 rocm-sdk-libraries rocm-bootstrap amd-torch-device-gfx1100 amd-torch-device-gfx110x amd-torchvision-device-gfx1100
python_embeded\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
python_embeded\python.exe -m pip install sageattention==1.0.6 --no-build-isolation
```

pip package backup: `D:\ai\pip_freeze_backup.txt`; retained nightly wheels: `D:\ai\nightly_wheels\`.

## 14. Timeline

| Date | Event |
|---|---|
| 2026-09-08 | Deployment complete; R2V workflow running; fixed missing ComfyUI-Manager and Easy-Use extension errors |
| 2026-09-09 | Tuned to the 178 s baseline; compiled an MP-pixel-to-resolution reference table |
| 2026-09-12 | Fixed sage autotune freeze (locked champion config); restored fast-start mode |
| 2026-09-21 | All-day "got slower" investigation: confirmed no hardware degradation (disk 1526MB/s, warm-state 7.2 s/it); attributed variance to cold-state costs + 10 steps + large portrait resolution stacking; tried and abandoned the FastH3 4-step model (deleted); final warm-state results: T2V 90 s, R2V dual-ref 105~142 s |