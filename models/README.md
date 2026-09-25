# models/

模型单独存放在这里。这个目录只有本说明入库，其余内容（权重、自定义模型的代码和说明）都被 `.gitignore` 挡在 git 之外，不会被误提交。录制出的数据不放这里，统一放 `runs/`。

## 下载的权重

按 Hugging Face 的 `<组织>/<名字>` 存放，运行时用 `--model` 传入：

```bash
hf download openai/clip-vit-base-patch32 \
  --local-dir models/openai/clip-vit-base-patch32 \
  --exclude "*.h5" --exclude "*.msgpack"
```

```bash
xray serve --model models/openai/clip-vit-base-patch32
```

权重已经在别处（比如共享的模型库）时不必复制，直接用 `--model` 传那个路径；也可以把这里的子目录建成指向那里的链接（Windows 用 `mklink /J`，macOS / Linux 用 `ln -s`）。

## 自定义模型

一个模型一个目录，和它有关的东西都放在里面：

```text
models/<名字>/
├── <名字>_recorder.py   录制入口：加载模型，挂上 xray 的录制器，把结果写到 runs/<名字>/<运行>/
├── test_<名字>.py       这个模型的测试
├── README.md            权重在哪、怎么录、验证记录
└── weights/             （可选）本地权重
```

- **按路径运行**：在项目根目录执行 `PYTHONPATH=. python models/<名字>/<名字>_recorder.py …`，测试用 `python -m pytest models/<名字>`。不要写 `import models.<名字>`：很多模型仓库自带同名的 `models` 包，混在一起会互相遮挡。
- **依赖只能由外向内**：录制脚本可以 import `xray`，`xray/` 里的代码不能反过来 import 这里的任何东西。
- **输出放 `runs/<名字>/`**：`xray serve` 会在 `runs/` 下找到每一次运行。
- **想让别人也能用**：把这个模型的适配器和录制器写进 `xray/` 并提 PR；只自己用的留在这里。
