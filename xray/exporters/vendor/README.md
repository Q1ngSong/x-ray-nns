# Vendored Three.js

`three.xray.min.js` 是 3D 视图用的 Three.js，导出时原样内联进 `index.html`，页面因此不需要网络。它不是手写代码，不要直接编辑。

- 版本：`three@0.186.0`（含 `OrbitControls`），用 `esbuild@0.28.2` 打包成 IIFE，全局名 `XRAY_THREE`。
- 入口：`three-entry.js` 只导出 3D 视图实际用到的类，新增用法时先改这里再重建。
- 许可证：MIT，全文见 `three.LICENSE`；打包文件末尾保留了 Three.js 的版权声明。

在任意临时目录重建：

```bash
npm install --no-audit --no-fund three@0.186.0 esbuild@0.28.2
cp <仓库>/xray/exporters/vendor/three-entry.js entry.js
npx esbuild entry.js --bundle --minify --format=iife --global-name=XRAY_THREE \
  --legal-comments=eof --target=es2020 \
  --banner:js="/* three@0.186.0 with OrbitControls, bundled by esbuild@0.28.2 from xray/exporters/vendor/three-entry.js (see README.md). */" \
  --outfile=<仓库>/xray/exporters/vendor/three.xray.min.js
```

重建后确认打包文件里没有 `fetch(`、`XMLHttpRequest`、`import(`，URL 也只剩 XHTML 命名空间和一处算法出处，这样页面离线也能打开。
