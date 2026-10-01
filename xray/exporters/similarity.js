// CLIP decision matrices live in the existing Details inspector, separate from the compact graph.
(function () {
  'use strict';
  const outputs = new Map(Object.values(scene.views || {}).filter(value => value.output?.similarity)
    .map(({output}) => [output.tensor, output.similarity]));
  if (!outputs.size) return;
  const css = node('style');
  css.textContent = `
.sim-details{margin:12px 0;border:1px solid #d6e2e7;border-radius:9px;background:#fbfdff;overflow:hidden}.sim-details[hidden]{display:none}.sim-details>summary{padding:12px 10px;cursor:pointer;font-size:12px;color:#44697c}.sim-content{padding:0 10px 12px;min-width:0}.sim-stage{margin:14px 0}.sim-stage h4{font-size:11px;font-weight:600;margin:0 0 5px;color:#577990}.sim-stage.text h4{color:#847399}.sim-stage.result h4{color:#60866d}.sim-meta,.sim-note{font:10px/1.6 var(--mono);color:#7d919e;margin:4px 0 8px;overflow-wrap:anywhere}.sim-table-wrap{overflow:auto;padding:2px 0}.sim-matrix{width:100%;border-collapse:separate;border-spacing:2px;table-layout:fixed;font:9px/1.5 var(--mono);color:#36556a}.sim-matrix td,.sim-matrix th{text-align:center;padding:5px 1px;white-space:nowrap}.sim-matrix th{font-weight:400;color:#8796a6;background:transparent}.sim-matrix th:first-child{width:22px}.sim-matrix td{border-radius:2px}.sim-operation{white-space:pre-line;text-align:center;margin:12px 0;padding:9px 4px;border-top:1px solid #e1e9ee;border-bottom:1px solid #e1e9ee;color:#668272;font:11px/1.7 var(--mono)}.sim-winner{white-space:pre-line;padding:9px;background:#edf5ef;border:1px solid #d2e3d7;border-radius:6px;color:#54755f;font-size:11px;line-height:1.7;margin:8px 0}.sim-prompts{margin:10px 0;font:10px/1.7 var(--mono);color:#847399}.sim-prompts p{margin:3px 0;overflow-wrap:anywhere}
`;
  document.head.append(css);
  const section = node('details', undefined, 'sim-details'); section.id = 'similarity-details'; section.hidden = true;
  section.append(node('summary', 'CLIP 决策过程 · 矩阵'));
  const content = node('div', undefined, 'sim-content'); section.append(content);
  E('detail-content').insertBefore(section, E('state-strip'));
  const sample = (length, limit) => length <= limit ? Array.from({length}, (_, i) => i)
    : [...Array.from({length: limit - 2}, (_, i) => i), null, length - 1];
  const promptName = index => String(inputs.text?.[index] || 'Text ' + (index + 1));
  let current;
  section.ontoggle = () => {
    if (section.open && current && !content.childElementCount) render(current);
  };

  function matrix(title, shape, values, rows, cols, rowName, colName, kind, digits = 3, percent = false) {
    const stage = node('section', undefined, 'sim-stage ' + kind);
    stage.append(node('h4', title), node('p', shape, 'sim-meta'));
    const table = node('table', undefined, 'sim-matrix'); table.setAttribute('aria-label', title);
    const head = node('tr'); head.append(node('th', ''));
    cols.forEach(c => { const th = node('th', c === null ? '…' : colName(c)); th.scope = 'col'; head.append(th); });
    const thead = node('thead'); thead.append(head); table.append(thead);
    const peak = values.reduce((best, row) => row.reduce((p, v) => Math.max(p, Math.abs(v)), best), 0) || 1;
    const tbody = node('tbody');
    rows.forEach(r => {
      const tr = node('tr'), th = node('th', r === null ? '⋮' : rowName(r)); th.scope = 'row'; tr.append(th);
      cols.forEach(c => {
        const omitted = r === null || c === null, value = omitted ? 0 : values[r][c];
        const td = node('td', omitted ? '…' : (value * (percent ? 100 : 1)).toFixed(digits) + (percent ? '%' : ''));
        if (!omitted) {
          const rgb = value < 0 ? [186, 122, 106] : kind === 'text' ? [157, 137, 186] : kind === 'result' ? [114, 166, 135] : [105, 154, 184];
          const t = .12 + .55 * Math.abs(value) / peak;
          td.style.background = 'rgb(' + rgb.map(v => Math.round(250 + (v - 250) * t)).join(',') + ')';
          td.title = rowName(r) + ' · ' + colName(c) + ': ' + value;
        }
        tr.append(td);
      });
      tbody.append(tr);
    });
    table.append(tbody); const wrap = node('div', undefined, 'sim-table-wrap'); wrap.append(table); stage.append(wrap); content.append(stage);
  }

  function render(data) {
    const m = data.image.values.length, n = data.text.values.length, d = data.dimensions;
    const images = sample(m, 5), texts = sample(n, 4), dims = sample(d, 6);
    const imageName = i => 'I' + (i + 1), textName = j => 'T' + (j + 1), dimension = k => String(k + 1);
    content.append(node('p', '两路 embedding 归一化后，在同一特征空间中做矩阵乘法。', 'sim-note'));
    matrix('01 · 视觉 embedding V̂', m + ' × ' + d + ' · L2 归一化', data.image.values, images, dims, imageName, dimension, 'vision', 2);
    const transposed = Array.from({length: d}, (_, k) => data.text.values.map(row => row[k]));
    matrix('02 · 文本 embedding T̂ᵀ', d + ' × ' + n + ' · 归一化后转置', transposed, dims, texts, dimension, textName, 'text');
    content.append(node('p', 'V̂ [' + m + '×' + d + '] × T̂ᵀ [' + d + '×' + n + ']\n↓', 'sim-operation'));
    matrix('03 · 图文相似度 C', m + ' × ' + n + ' · Cᵢⱼ = Σₖ V̂ᵢₖ · T̂ⱼₖ', data.cosine, images, texts, imageName, textName, 'result', 4);
    content.append(node('p', 'C × exp(logit_scale)\n= C × ' + data.scale.toFixed(4), 'sim-operation'));
    matrix('04 · Logits', '模型实际输出 · ' + m + ' × ' + n, data.logits, images, texts, imageName, textName, 'vision');
    matrix('05 · Softmax', '对每行的候选文本换算概率', data.probs, images, texts, imageName, textName, 'result', 2, true);
    images.filter(i => i !== null).forEach(i => {
      const row = data.logits[i], best = row.indexOf(Math.max(...row));
      content.append(node('p', imageName(i) + ' 最高分 → ' + textName(best) + '\n' + promptName(best), 'sim-winner'));
    });
    const prompts = node('div', undefined, 'sim-prompts');
    texts.filter(j => j !== null).forEach(j => prompts.append(node('p', textName(j) + ' · ' + promptName(j)))); content.append(prompts);
    content.append(node('p', '矩阵中的 … 表示省略坐标；相似度使用全部 ' + d + ' 维计算。悬停数值可查看记录精度。', 'sim-note'));
    content.append(node('p', '向量、缩放参数与 logits 来自记录；C、概率及最高分候选为展示推导。语义对齐已在训练中学习。', 'sim-note'));
  }

  function sync() {
    const data = outputs.get(state.tensor);
    if (current === data) return;
    current = data;
    section.hidden = !data;
    section.open = false;
    content.replaceChildren();
  }

  window.XRAY_SIMILARITY = {sync};
})();
