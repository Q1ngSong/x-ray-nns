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

Only the generated directory is published to the `gh-pages` branch. In repository Settings → Pages, select **Deploy from a branch**, **gh-pages**, and **/ (root)**. The `.nojekyll` file preserves the generated HTML unchanged. Rebuild and republish after updating the project README, viewer, or sample list.

## Public CLIP sample

`site/samples/clip-cat/` contains the recorded trace, precomputed scene data and input image for CLIP ViT-B/32, with prompts `a photo of a cat`, `a photo of a tiger`, and `a photo of a dog`. It is the same real inference illustrated in the project README. The scene retains attention and intermediate-value panels. Enable **Final output** (or click the final node) to show the compact score bars. Select the final node or its compact output to inspect it in the existing right-hand **Details** panel. The inspector shows normalized image and text matrices, their dot product, scale, logits and derived probabilities. Its **CLIP 决策过程 · 矩阵** section can be collapsed while the graph stays compact. Omitted embedding coordinates are marked with ellipses; scores use every dimension. `#similarity` opens the compact result. The browser only replays this sample; new inference requires the local application.

The public trace omits machine-local model paths, source locations, and unavailable raw-tensor storage references. Model weights and raw `.pt` files are excluded. The original local recording is unchanged. To add a case, prepare its reviewed trace, scene and assets in a new sample directory and add its card to `examples.json`.

The cat photograph is scikit-image's `chelsea.png`, photographed by Stéfan van der Walt and released under CC0. The viewer and project are MIT; the generated site includes the project's LICENSE and Three.js license.
