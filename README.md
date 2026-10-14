# SoulX-LiveAct CuDNN：精简部署与开发交接版

这是当前项目的源码快照，便于另一位开发者独立部署、运行和继续接线。原始项目：[Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct)。

保留：双 H100/H200 推理、FP8 GEMM、原生 cuDNN BF16 注意力、规范化融合、两槽 VAE 解码重叠、开放 PCM 常驻人物生成，以及全本地 ASR → LLM → 中文 TTS 和鉴权语音网页。默认女声 **zf_003**，另有 9 个可选中文女声。

**完整数字人对话尚未接通。** 语音问答与人物生成目前是两个独立入口；后续需要将回答 PCM 接入人物生成，再实现浏览器同步流式音视频、打断与长时验收。本包不宣称已完成持续 20 FPS。源项目的指定人物样片及 H6 双卡常驻功能已经验证，换环境、图片或音频后仍需重新验证。

## 1. 包内结构

```text
tools/                 部署、模型校验、启动与常驻客户端
deploy/                HPC3 Slurm 提交模板
dialogue/              本地问答、轮次编排、鉴权与进程监管
model_liveact/         LiveAct 与已实施的推理优化
src/audio_analysis/    音频窗口、有限前瞻与开放 PCM
wan/、kokoro/          上游模型实现，保留原始版权注释
templates/、static/    登录与语音网页
deployment/            模型文件/音色清单，不含权重
requirements/          运行版本与约束
examples/input.json    可修改的图片、音频与动作配置
source-snapshot.json   上游提交、LightX2V 提交和源码 SHA
```

不包含模型权重、Python 环境、用户上传素材、生成视频、历史实验、旧作业凭据或 SSH 配置。全部部署说明集中在本 README。

## 2. 硬件与基础工具

- Linux x86_64，Python **3.10**。
- 人物生成：同一节点的 **2 × H100 80GB 或 2 × H200**，建议 16 CPU、256 GiB 系统内存。预检会拒绝其他 GPU 配置。
- 语音网页：ASR、LLM、TTS 使用 CPU，建议 12 CPU、32 GiB；不调用付费 API。
- `git`、`sox`、`ffmpeg`、`ffprobe`；下载人物权重还需要 `aria2c`。
- 驱动必须能运行 CUDA 12.8 版 PyTorch。实际兼容性以双卡预检为准。
- 模型约 54 GiB，环境和构建缓存另需空间；建议预留至少 100 GiB。

可用 Conda 准备工具，不需要系统管理员修改 Python：

```bash
conda create -n liveact-tools -c conda-forge python=3.10 sox=14.4.2 ffmpeg=7.1.1 aria2 git -y
conda activate liveact-tools
git clone https://github.com/Kumoris/SoulX-LiveAct_CuDNN.git
cd SoulX-LiveAct_CuDNN
mkdir -p runs/slurm
```

编译 `llama-cpp-python` 还需要 C/C++ 编译器。HPC3 模板使用 `gcc/12.5`；其他服务器可使用自身编译工具链。

## 3. 建立环境

在独立工作站/服务器执行：

```bash
python -m tools.bootstrap --mode all
```

只运行语音网页可选 `--mode speech`，只生成人物视频可选 `--mode avatar`。语音模式使用 CPU PyTorch；人物/完整模式使用 CUDA 12.8 PyTorch。

**HPC3 上通过 Slurm 安装/编译，不在登录节点加载模型：**

```bash
env -u SLURM_CLUSTERS sbatch -p acd_u --export=ALL,LIVEACT_SETUP_MODE=all deploy/setup.sbatch
squeue -u "$USER"
```

环境目录：`.liveact-env` 为基础环境，`.hopper-env` 为 FP8 人物推理环境，`.dialogue-env` 为 CPU 问答环境。后两个环境只读继承基础包并安装自己的依赖。LightX2V 按 `source-snapshot.json` 中的固定提交获取，不把整个第三方仓库打进本包。SageAttention 不作为本版本的运行后端。

若**同一服务器已有当前项目的环境和权重**，可以先复用它们，避免重新安装或传输。把下方路径改成实际原项目路径：

```bash
liveactOriginal=/absolute/path/to/current/SoulX-LiveAct
ln -s "$liveactOriginal/.liveact-env" .liveact-env
ln -s "$liveactOriginal/.hopper-env" .hopper-env
ln -s "$liveactOriginal/.dialogue-env" .dialogue-env
ln -s "$liveactOriginal/third_party" third_party
ln -s "$liveactOriginal/models" models
```

复用环境时不要再对这些环境运行 bootstrap；部署程序会从新目录读取源码。迁移到另一台服务器时应重新安装环境，不能直接复制虚拟环境目录。

## 4. 准备与验证权重

推荐从原部署服务器传输已经验证的 `models/`，例如在目标服务器执行：

```bash
scp -rp YOUR_USER@YOUR_SOURCE_HOST:/absolute/project/models ./
.liveact-env/bin/python -m tools.models verify --group all
```

也可以显式下载清单中的文件，下载支持断点续传并在完成后校验：

```bash
.liveact-env/bin/python -m tools.models download --group avatar
.liveact-env/bin/python -m tools.models download --group speech
```

人物权重来自 ModelScope，按包内文件 SHA 校验；对话权重与十个音色来自固定 Hugging Face revision。远端文件若与清单不符，程序会拒绝使用；不要删除校验来绕过版本差异。推理时关闭在线下载。

必须形成以下目录：

```text
models/LiveAct/
models/chinese-wav2vec2-base/
models/dialogue/faster-whisper-small/
models/dialogue/Qwen2.5-3B-Instruct-GGUF/
models/dialogue/Kokoro-82M-v1.1-zh/voices/
```

语音识别为 faster-whisper-small CPU int8；回答模型为 Qwen2.5-3B-Instruct Q4_K_M；TTS 为 Kokoro-82M-v1.1-zh。下载操作不等于模型推理已经通过。

## 5. 图片、音频与动作输入

提供自己的图片和 **10–15 秒**音频，分别放到 `examples/reference.jpg` 和 `examples/audio.wav`，然后修改 `examples/input.json`。文件路径相对于本仓库根目录，也支持绝对路径。

当前配置沿用已认可 C2 的猫耳插画和手串手势描述。更换人物后应同步修改 `main_prompt`、`persistent_prompt` 和 `edits`，不要让猫耳、手串等描述继续约束真人图片。

启动器会将非标准音频转成 16 kHz、单声道、PCM16，并拒绝超出 10–15 秒的测试输入，不裁剪或补长。标准 WAV 不重新转换。原模型参数保持 416×720、媒体 20 FPS、三步、0.5 秒额外前瞻和 4 秒历史。

## 6. 双卡预检与生成视频

HPC3：

```bash
env -u SLURM_CLUSTERS sbatch -p acd_u deploy/preflight.sbatch
# 查看 runs/slurm/preflight-作业号.out 与 runs/hardware.json，确认 COMPLETED/0:0。
env -u SLURM_CLUSTERS sbatch -p acd_u deploy/video.sbatch
```

视频及运行报告写入 `runs/video-作业号/`。独立 GPU 服务器上可直接执行：

```bash
.hopper-env/bin/python -m tools.run preflight
.hopper-env/bin/python -m tools.run video --input examples/input.json
```

检查源文件和环境目录：

```bash
.liveact-env/bin/python -m tools.run doctor
```

预检验证两张实际 Hopper GPU、NCCL 和小规模 FP8 内核；不代替完整视频质量或持续速率验收。第一次运行会加载权重并编译内核，冷启动明显慢于后续轮次。

## 7. 双卡常驻 PCM 人物生成

```bash
env -u SLURM_CLUSTERS sbatch -p acd_u deploy/resident.sbatch
```

运行目录为 `runs/resident-作业号/`。等待其中 `service.json` 的 `status` 为 `ready`；内核与固定条件会在第一条完整任务时预热。控制服务仅监听计算节点的 `127.0.0.1`。

在**同一计算节点**的另一个终端发送音频：

```bash
.hopper-env/bin/python -m tools.resident_client status --root runs/resident-作业号
.hopper-env/bin/python -m tools.resident_client send \
  --root runs/resident-作业号 --audio /absolute/path/to/mono16k.wav --paced
.hopper-env/bin/python -m tools.resident_client shutdown --root runs/resident-作业号
```

HPC3 可以在作业运行期间进入已分配节点：

```bash
srun --jobid=作业号 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --pty bash
cd /absolute/path/to/SoulX-LiveAct_CuDNN
```

如果站点不允许这样的附加步骤，可在同一个 sbatch 脚本里启动 worker 并运行客户端。不要直接在登录节点请求计算节点的 `127.0.0.1`。

每轮输入切成 100 ms PCM 包，显式发送 EOS；`--paced` 模拟实时到达。完整视频、逐块可读 MP4 与状态写入 `runs/resident-作业号/tasks/轮次ID/`。取消命令为 `cancel --root ... --turn-id ...`。这是私有渲染协议，尚不是公开数字人对话网页，也不是 MSE 连续播放协议。

## 8. 本地语音问答网页

HPC3：

```bash
env -u SLURM_CLUSTERS sbatch -p acd_u deploy/speech.sbatch
```

CPU 问答在 HPC3 的 ACD 分区仍按平台要求申请最少 1 GPU，模型实际只使用 CPU。独立服务器无需 GPU：

```bash
.dialogue-env/bin/python -m tools.speech --host 127.0.0.1 --port 15001
```

服务在日志中输出运行目录、节点、端口及登录令牌文件的位置。查看该目录下 `service.json`：HTTP 启动不代表模型已 ready，应等待 `status=ready`。从自己的终端读取 `access-token` 登录，不把令牌提交到 Git。

本机访问 `http://127.0.0.1:15001/login`。HPC 计算节点服务需要 SSH 隧道，在自己电脑执行并替换节点、端口和用户名：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15001:COMPUTE_NODE:SERVICE_PORT YOUR_USER@YOUR_LOGIN_HOST
```

然后打开 `http://127.0.0.1:15001/login`，使用这次服务的令牌登录，进入 `/dialogue`。SSH 认证后终端不继续输出是正常的；保持窗口开启。浏览器麦克风需要 localhost 或 HTTPS。

页面提供录音、自动句末、文字、回答、中文语音、打断和服务恢复。默认 `zf_003`，可在页面选择十个女声；后端也支持环境变量 `LIVEACT_TTS_VOICE`。

公网接入可使用现有 Caddy/反向 SSH 网关。服务已有 `LIVEACT_PUBLIC_ORIGIN` 和 `LIVEACT_GATEWAY_KEY_FILE` 支持，公网域名、密钥文件和隧道需由部署方自行配置；本包不包含现有服务器凭据。直接公开内部计算节点端口不作为部署步骤。

## 9. 验证与故障排查

```bash
.liveact-env/bin/python -m unittest discover -s tests -v
squeue -u "$USER"
sacct -j 作业号 --format=JobID,State,ExitCode,Elapsed,NodeList -P
```

作业结束后从 `squeue` 消失是正常的，使用 `sacct` 和 `runs/slurm/` 查看结果。Connection refused 通常表示远端服务已停止或节点/端口过期，应重新读取本次运行信息。源码修改后 `doctor` 会提示与交接快照不符，保留原清单供对照，不伪造历史通过。

**本次交接复核（2026-10-14）：** 在另一个独立目录、复用已有已验证环境，二期与三期各 25 项 CPU 检查通过；双 H100/NCCL/FP8 预检、独立生成和两轮常驻任务通过。三条完整片的解码 RGB/PCM 与已认可 C2 精确一致，每 rank 单次加载、每轮 240 个 KV 张量重置、正常退出通过。此次没有重建全新环境，也没有重新做语音网页或持续实时验收；这些限制记录在 `source-snapshot.json`。

## 10. 接手开发的下一项

1. 稳定真实时钟输入下的完整输出；最终目标是持续 ≥20 FPS、完整块 p95≤1.6 秒、至少 10 分钟且无持续积压。文件标为 20 FPS 不等于生成达标。
2. 将 `dialogue/session.py` 的回答 `audio` 事件桥接到常驻人物协议：按轮次/epoch 发送 PCM，回答完成才发送 EOS，取消同时作废旧人物任务。
3. 将对应回答声音与人物帧按同一 PCM 样本时钟封装 fMP4/MSE，浏览器同步边生成边播放；当前逐块 MP4 不直接当作完成了 MSE。
4. 实测真实麦克风多轮、打断停止延迟、首个同步音视频、断线恢复和长时稳定，再冻结最终配置。

角色参考与动作文字是软控制；指定样片已认可，不承诺任意输入的手指、姿态或骨架几何都正确。对话单轮输入和回答保留 15 秒限制，长会话通过多轮实现。

本包沿用各上游源码中的版权声明与模型来源，不另行变更上游许可。精确来源和文件 SHA 见 `source-snapshot.json`。
