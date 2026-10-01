# Public website

The homepage is the project title, curated example cards, then the repository's complete README. `examples.json` determines the cards and the examples included in a release.

## Build and preview

From the repository root, with Python 3.11 or newer:

```bash
python -m pip install -r site/requirements.txt
python site/build.py --output runs/tool/github-pages/preview/x-ray-nns
python -m http.server 8770 --directory runs/tool/github-pages/preview
```

Open `http://localhost:8770/x-ray-nns/`. The output directory must be new or empty. Building requires neither PyTorch nor model weights. All paths work under the GitHub Pages repository subpath.

The generated website snapshot is tracked at the repository root on `main`. To publish it from `main`, in repository Settings → Pages select **Deploy from a branch**, **main**, and **/ (root)**. The `.nojekyll` file preserves the generated HTML unchanged. After updating the project README, viewer, or sample list, rebuild into a new output directory, copy the reviewed website files into the repository root, and commit them. A source change alone does not refresh the snapshot.

## Public CLIP sample

`site/samples/clip-cat/` contains the recorded trace, precomputed scene data and input image for CLIP ViT-B/32, with prompts `a photo of a cat`, `a photo of a tiger`, and `a photo of a dog`. It is the same real inference illustrated in the project README. The scene retains attention and intermediate-value panels. Enable **Final output** (or click the final node) to show the compact score bars. Select the final node or its compact output to inspect it in the existing right-hand **Details** panel. The inspector shows normalized image and text matrices, their dot product, scale, logits and derived probabilities. Its **CLIP 决策过程 · 矩阵** section can be collapsed while the graph stays compact. Omitted embedding coordinates are marked with ellipses; scores use every dimension. `#similarity` opens the compact result. The browser only replays this sample; new inference requires the local application.

The public trace omits machine-local model paths, source locations, and unavailable raw-tensor storage references. Model weights and raw `.pt` files are excluded. The original local recording is unchanged. To add a case, prepare its reviewed trace, scene and assets in a new sample directory and add its card to `examples.json`.

The cat photograph is scikit-image's `chelsea.png`, photographed by Stéfan van der Walt and released under CC0. The viewer and project are MIT; the generated site includes the project's LICENSE and Three.js license.

## CNN samples

`alexnet-cat` and `resnet18-cat` use the same public cat image, official torchvision ImageNet-1K V1 weights and preprocessing, and real CPU evaluation. Each recording checks its logits against the unmodified model (maximum absolute error 0.0 in both published examples). The ResNet recording includes all eight actual residual additions and both identity and projection shortcuts.

The sample pages offer spatial PCA, individual feature channels and ImageNet top-five predictions. Channel maps use average pooling to at most 28×28 and individual min/max scaling; top-five probabilities are computed over all 1000 classes. Model weights, local paths and raw `.pt` files are excluded from the public samples.

Recreate the full local recordings with the `cnn-run` commands in the repository README, then curate their trace/scene/assets into these sample directories. `examples.json` includes all three cases in the normal static build.

CNN pages open with a structural guide. ResNet stages and all eight blocks are separate cards; selecting a block shows its main and shortcut paths, actual tensor shapes, and the addition before ReLU. Step controls explain identity versus projection. Diagram modules open the real recorded values in Details; the 3D action focuses the same block. The 2D/3D views use distinct module colours and bounded group cards.
