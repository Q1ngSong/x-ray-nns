// Shared core of the 2D and 3D model views: the embedded scene model, the view state both views
// share, layout in world units, intermediate-value panels drawn on canvases, labels, actions and
// the toolbar. Loaded after the 2D player script and sharing its globals: state, tensors, inputs,
// scene, E, node, trimPadding, selectOperation, selectTensor and selectPatch. Renderers register
// themselves with XRAY_SCENE.register(mode, renderer) and draw the same layout their own way.
(function () {
  'use strict';
  const model = scene;
  const blocks = model.blocks || {}, groups = model.groups || {}, lanes = model.lanes || [], views = model.views || {};
  const PALETTE = {
    vision: {face: '#d9e7f3', edge: '#6f93b1'}, text: {face: '#e5e0f2', edge: '#8f86b0'},
    fusion: {face: '#f3e3d6', edge: '#b78f70'}, model: {face: '#dcefe9', edge: '#6f9e92'},
  };
  // Lanes the palette does not name take the colour of their place in the scene's lane order.
  const CYCLE = [PALETTE.text, PALETTE.model, PALETTE.vision, {face: '#f1e1e8', edge: '#a9829a'}, PALETTE.fusion];
  const TYPES = [['hidden', 'Layer output'], ['attention', 'Attention'], ['cross', 'Cross-attention'], ['mlp', 'MLP activation'],
    ['qkv', 'Q / K / V'], ['output', 'Final output']];
  // World units: slabs are thin along x (the flow axis); lanes stack along y.
  const GAP = 0.9, LAYER_GAP = 1.0, SLAB = 0.3, PLATE = 0.07, PLATE_GAP = 0.05, LANE_GAP = 2.4, TILE_GAP = 0.05;
  // Panels: canvas pixels map to world units through PX; they stack away from the other lane.
  const PX = 1 / 60, PANEL_GAP = 0.3, PANEL_LIFT = 0.45, PAD = 14, TITLE = 30;
  const FONT = 'ui-monospace, SFMono-Regular, Menlo, monospace';
  const ATTENTION_RAMP = [[247, 249, 252], [46, 84, 122]], MLP_RAMP = [[241, 246, 244], [47, 110, 96]];
  const view = {mode: '2d', expanded: new Set(), open: new Set(), types: new Set(['hidden', 'output']),
    head: 'avg', style: 'matrix', active: null, hover: null};
  const renderers = new Map(), decoded = new Map(), imageWaiters = [];
  const flowLanes = lanes.filter(lane => !lane.items.every(item => item.block && blocks[item.block].kind === 'result'));
  const mergeLanes = lanes.filter(lane => !flowLanes.includes(lane));
  let imageCanvas = null, imageReady = false, lastBranch = state.branch;

  const grow = n => 0.4 + 0.3 * Math.log2(1 + Math.max(1, n));
  const palette = lane => {
    const index = lanes.findIndex(item => item.id === lane);
    return PALETTE[lane] || (index < 0 ? PALETTE.model : CYCLE[index % CYCLE.length]);
  };
  // The first flow lane (and the fusion lane) stacks panels upwards, every other lane downwards.
  const panelDirection = lane => flowLanes.findIndex(item => item.id === lane) > 0 ? -1 : 1;
  const widthOf = object => Math.max(object.size.w, object.stack.width);
  // Numbered layers read as L3; named group members (resnet0, attn1) keep their names.
  const blockName = block => block.group && /^\d+$/.test(block.label) ? 'L' + block.label : block.label;
  const cleanToken = label => String(label).replace('</w>', '').replace('<|startoftext|>', 'BOS').replace('<|endoftext|>', 'EOS');

  function inputKind(block) {
    if (block.kind !== 'input') return null;
    if (block.semantic === 'image_pixels' || /image$/.test(block.stage)) return 'image';
    if (block.semantic === 'image_patch_tokens' || /patch/.test(block.stage)) return 'patches';
    if (/token/.test(block.stage)) return 'tokens';
    return null;
  }

  // Slab size from the recorded shape: height follows the token count, depth the feature width.
  // A [batch, channels, H, W] feature map counts its positions as tokens and its channels as features.
  // Results stay at least 1.4 units so the final logits remain visible next to wide encoders.
  function slabSize(block) {
    const dims = (block.shape || []).slice();
    if (dims.length === 4) dims.push(dims.splice(1, 1)[0]);
    if (dims.length > 1 && dims[0] === 1) dims.shift();
    const features = dims.length ? dims[dims.length - 1] : 1;
    const sequence = dims.slice(0, -1).reduce((product, value) => product * value, 1);
    const least = block.kind === 'result' ? 1.4 : 0;
    return {w: SLAB, h: Math.max(least, grow(sequence)), d: Math.max(least, grow(features))};
  }

  function patchGrid() {
    const grid = (inputs.image_patch_tokens && inputs.image_patch_tokens.grid_shape) || [7, 7];
    const rows = Number(grid[0]) || 7, cols = Number(grid[1]) || 7, tile = (2.6 - TILE_GAP * (cols - 1)) / cols;
    return {rows, cols, tile, size: {w: 2.6, h: rows * tile + TILE_GAP * (rows - 1), d: 0.04}};
  }

  // A collapsed group is a tight stack of thin plates, one per member layer.
  function groupSize(group, member) {
    const count = group.blocks.length;
    return {w: count * PLATE + (count - 1) * PLATE_GAP, h: member.h, d: member.d};
  }

  // WebGL refuses file:// images, so the scene embeds the centre crop as a data URL; both views
  // draw from this one canvas and hear about it through ``onImage`` once it has loaded.
  function sourceImage(onReady) {
    if (!model.image) return null;
    if (onReady) imageWaiters.push(onReady);
    if (!imageCanvas) {
      imageCanvas = document.createElement('canvas');
      imageCanvas.width = imageCanvas.height = 224;
      const image = new Image();
      image.onload = () => {
        imageCanvas.getContext('2d').drawImage(image, 0, 0, 224, 224);
        imageReady = true;
        imageWaiters.forEach(waiter => waiter());
        redrawPanels('attention');
        redrawPanels('cross');
      };
      image.src = model.image;
    } else if (imageReady && onReady) {
      onReady();
    }
    return imageCanvas;
  }

  // Token labels of a lane: the strings of its ``input.tokens`` block, one list per sequence.
  function tokenRows(lane) {
    const block = Object.values(blocks).find(item => item.lane === lane && inputKind(item) === 'tokens');
    const tensor = block && tensors.get(block.tensor), preview = tensor && tensor.metadata && tensor.metadata.preview;
    return preview && Array.isArray(preview.tokens) ? preview.tokens : null;
  }

  // Token chips for a ``input.tokens`` block, with their size in world units; padding after EOS folds into one chip.
  function tokenCanvas(block) {
    const tensor = tensors.get(block.tensor), preview = tensor && tensor.metadata && tensor.metadata.preview || {};
    const rows = (Array.isArray(preview.tokens) ? preview.tokens : []).map(trimPadding), ids = Array.isArray(preview.values) ? preview.values : [];
    const cols = Math.max(1, ...rows.map(row => row.length)), cell = 68, line = 44, pad = 8, scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = (cols * cell + pad * 2) * scale;
    canvas.height = (Math.max(1, rows.length) * line + pad * 2) * scale;
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    rows.forEach((row, r) => row.forEach((label, c) => {
      const x = pad + c * cell, y = pad + r * line, text = String(label);
      const special = /^<\|/.test(text);
      ctx.fillStyle = special ? '#f0edf7' : '#f8fafc';
      ctx.strokeStyle = special ? '#d9d1e9' : '#d4dee5';
      ctx.beginPath();
      ctx.roundRect(x + 2, y + 2, cell - 4, line - 6, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = special ? '#756a92' : '#4f6978';
      ctx.font = '600 13px ' + FONT;
      ctx.fillText(cleanToken(text).slice(0, 7), x + 8, y + 19);
      ctx.fillStyle = '#8a9ba5';
      ctx.font = '10px ' + FONT;
      ctx.fillText(String((ids[r] || [])[c] ?? ''), x + 8, y + 33);
    }));
    return {canvas, width: canvas.width / scale / 100, height: canvas.height / scale / 100};
  }

  // ---- Intermediate-value panels: each is a canvas drawn in 2D; renderers show it as they like.

  function bytesOf(text) {
    if (!decoded.has(text)) decoded.set(text, Uint8Array.from(atob(text), char => char.charCodeAt(0)));
    return decoded.get(text);
  }

  function ramp([from, to], t) {
    const k = Math.max(0, Math.min(1, t));
    return 'rgb(' + from.map((value, index) => Math.round(value + (to[index] - value) * k)).join(',') + ')';
  }

  // PCA colours are softened towards white so they sit with the pastel structure.
  function softColour(rgb, index) {
    return 'rgb(' + [0, 1, 2].map(offset => Math.round(rgb[index + offset] * 0.85 + 38)).join(',') + ')';
  }

  function diverging(t) {
    return t < 0 ? ramp([[247, 247, 247], [111, 147, 177]], -t) : ramp([[247, 247, 247], [196, 154, 122]], t);
  }

  // A ViT row (CLS + n² patches) draws as a CLS cell plus an n × n grid; other rows are token strips,
  // wrapped every 26 tokens once a sequence is longer than 40.
  function tokenArea(rows, cols, cell) {
    const side = Math.round(Math.sqrt(cols - 1));
    if (rows === 1 && side > 1 && side * side === cols - 1) {
      const grid = side * cell + (side - 1) * 2;
      return {side, cell, width: cell + 12 + grid, height: grid + 12};
    }
    const wrap = cols > 40 ? 26 : cols, lines = rows * Math.ceil(cols / wrap);
    return {side: 0, cell, wrap, width: wrap * (cell + 2) - 2, height: lines * (cell + 14) - 2};
  }

  function drawTokens(ctx, area, x, y, rows, cols, colourAt, labels) {
    const cell = area.cell;
    ctx.font = '8px ' + FONT;
    if (area.side) {
      ctx.fillStyle = colourAt(0, 0);
      ctx.fillRect(x, y, cell, cell);
      ctx.fillStyle = '#8a9ba5';
      ctx.fillText('CLS', x, y + cell + 10);
      for (let i = 1; i < cols; i++) {
        const r = Math.floor((i - 1) / area.side), c = (i - 1) % area.side;
        ctx.fillStyle = colourAt(0, i);
        ctx.fillRect(x + cell + 12 + c * (cell + 2), y + r * (cell + 2), cell, cell);
      }
      return;
    }
    const lines = Math.ceil(cols / area.wrap);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cx = x + (c % area.wrap) * (cell + 2), cy = y + (r * lines + Math.floor(c / area.wrap)) * (cell + 14);
        ctx.fillStyle = colourAt(r, c);
        ctx.fillRect(cx, cy, cell, cell);
        const label = labels && labels[r] && labels[r][c];
        if (label !== undefined && label !== null) {
          ctx.fillStyle = '#8a9ba5';
          ctx.fillText(cleanToken(label).slice(0, Math.max(2, Math.floor(cell / 5))), cx, cy + cell + 10);
        }
      }
    }
  }

  function frameCanvas(ctx, spec, title, caption) {
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#d9e2ea';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(0.5, 0.5, spec.width - 1, spec.height - 1, 8);
    ctx.fill();
    ctx.stroke();
    ctx.textAlign = 'left';
    ctx.fillStyle = '#3f5a69';
    ctx.font = '600 11px ' + FONT;
    ctx.fillText(title, PAD, 20);
    const room = spec.width - PAD * 2 - ctx.measureText(title).width - 12;
    ctx.font = '9px ' + FONT;
    if (caption && ctx.measureText(caption).width <= room) {
      ctx.textAlign = 'right';
      ctx.fillStyle = '#8a9ba5';
      ctx.fillText(caption, spec.width - PAD, 20);
      ctx.textAlign = 'left';
    }
  }

  function tokenSpec(key, data, block, title, caption, colourAt) {
    const area = tokenArea(data.rows, data.cols, data.rows === 1 ? 28 : 30);
    const spec = {key, tensor: data.tensor, width: area.width + PAD * 2, height: TITLE + area.height + PAD};
    spec.draw = ctx => {
      frameCanvas(ctx, spec, blockName(block) + ' · ' + title, caption);
      drawTokens(ctx, area, PAD, TITLE, data.rows, data.cols, colourAt, tokenRows(block.lane));
    };
    return spec;
  }

  // Name of batch row ``row`` of ``rows``: the trace's ``batch_labels`` when they fit.
  function batchLabel(rows, row) {
    const labels = Array.isArray(inputs.batch_labels) && inputs.batch_labels.length === rows ? inputs.batch_labels : null;
    return labels ? String(labels[row]) : 'row ' + (row + 1);
  }

  // A feature map (rows × H × W positions) draws each batch row as an H × W raster about 128 px wide.
  function gridSpec(key, data, block, title, caption, colourAt) {
    const [h, w] = data.grid, scale = Math.max(1, Math.floor(128 / Math.max(h, w))), top = TITLE + (data.rows > 1 ? 14 : 0);
    const spec = {key, tensor: data.tensor, width: PAD * 2 + data.rows * w * scale + (data.rows - 1) * 16, height: top + h * scale + PAD};
    spec.draw = ctx => {
      frameCanvas(ctx, spec, blockName(block) + ' · ' + title, caption);
      for (let r = 0; r < data.rows; r++) {
        const x = PAD + r * (w * scale + 16);
        if (data.rows > 1) {
          ctx.fillStyle = '#8a9ba5';
          ctx.font = '9px ' + FONT;
          ctx.fillText(batchLabel(data.rows, r), x, TITLE + 8);
        }
        for (let i = 0; i < h * w; i++) {
          ctx.fillStyle = colourAt(r, i);
          ctx.fillRect(x + (i % w) * scale, top + Math.floor(i / w) * scale, scale, scale);
        }
      }
    };
    return spec;
  }

  // The head chosen in the toolbar as an index below ``heads``, or null for the mean of all heads.
  function chosenHead(heads) {
    const chosen = view.head === 'avg' ? null : Number(view.head) - 1;
    return chosen !== null && chosen < heads ? chosen : null;
  }

  // Attention for one batch row: the selected head (or the mean of all heads), scaled to its maximum.
  function attentionMap(data, row, head) {
    const bytes = bytesOf(data.data), count = (data.queries || data.size) * data.size, out = new Float32Array(count);
    for (let h = 0; h < data.heads; h++) {
      if (head !== null && h !== head) continue;
      const base = (row * data.heads + h) * count, scale = data.scale[row * data.heads + h];
      for (let i = 0; i < count; i++) out[i] += bytes[base + i] * scale / 255;
    }
    const max = out.reduce((best, value) => Math.max(best, value), 0);
    return max > 0 ? out.map(value => value / max) : out;
  }

  // "On image": the CLS query's attention over patches; unattended patches are dimmed.
  function drawAttentionOnImage(ctx, map, size, x, y, extent) {
    const side = Math.round(Math.sqrt(size - 1)), cell = extent / side;
    ctx.drawImage(imageCanvas, x, y, extent, extent);
    let max = 0;
    for (let i = 1; i < size; i++) max = Math.max(max, map[i]);
    for (let i = 1; i < size; i++) {
      const t = max ? map[i] / max : 0, r = Math.floor((i - 1) / side), c = (i - 1) % side;
      ctx.fillStyle = 'rgba(46,84,122,' + (0.78 * (1 - t)).toFixed(3) + ')';
      ctx.fillRect(x + c * cell, y + r * cell, cell, cell);
    }
  }

  // Attention kept as the CLS query row only (long ViT sequences): CLS and its patch grid, or on the image.
  function clsAttentionSpec(data, block) {
    const area = tokenArea(1, data.size, 9), extent = Math.max(area.height, 160);
    const spec = {key: 'attention', tensor: data.tensor, width: PAD * 2 + Math.max(area.width, extent), height: TITLE + extent + PAD};
    spec.draw = ctx => {
      const head = chosenHead(data.heads), onImage = area.side && view.style === 'image' && imageReady;
      frameCanvas(ctx, spec, blockName(block) + ' · attention, CLS → patches',
        head === null ? 'mean of ' + data.heads + ' heads' : 'head ' + (head + 1) + ' of ' + data.heads);
      const map = attentionMap(data, 0, head), patches = Math.max(...map.slice(1)) || 1;
      if (onImage) drawAttentionOnImage(ctx, map, data.size, PAD, TITLE, extent);
      else drawTokens(ctx, area, PAD, TITLE, 1, data.size, (r, c) => ramp(ATTENTION_RAMP, c ? Math.pow(map[c] / patches, 0.35) : 1), null);
    };
    return spec;
  }

  // Cross-attention of one prompt token: its probability at every position for the chosen head (or the
  // mean) and that probability's range; values span the range, so near-uniform maps such as the
  // attention sink on BOS still show where they dip.
  function crossMap(data, token, head) {
    const bytes = bytesOf(data.data), count = data.grid[0] * data.grid[1], tokens = data.tokens.length, out = new Float32Array(count);
    let used = 0;
    for (let h = 0; h < data.heads; h++) {
      if (head !== null && h !== head) continue;
      const base = (h * tokens + token) * count, scale = data.scale[h * tokens + token];
      for (let i = 0; i < count; i++) out[i] += bytes[base + i] * scale / 255;
      used += 1;
    }
    const low = Math.min(...out), high = Math.max(...out), span = high - low;
    return {values: span > 0 ? out.map(value => (value - low) / span) : out.fill(0), low: low / Math.max(1, used), high: high / Math.max(1, used)};
  }

  const SPECS = {
    hidden: (data, block) => {
      const rgb = bytesOf(data.rgb), colourAt = (r, c) => softColour(rgb, (r * data.cols + c) * 3);
      return data.grid ? gridSpec('hidden', data, block, 'layer output', 'PCA colour', colourAt)
        : tokenSpec('hidden', data, block, 'layer output', 'PCA colour', colourAt);
    },
    mlp: (data, block) => {
      const share = bytesOf(data.active), top = Math.max(13, ...share);
      return tokenSpec('mlp', data, block, 'MLP', 'units > 0, max ' + Math.round(top / 2.55) + '%',
        (r, c) => ramp(MLP_RAMP, share[r * data.cols + c] / top));
    },
    qkv: (data, block) => {
      const kinds = ['q', 'k', 'v'].filter(key => data[key]), first = data[kinds[0]];
      const area = tokenArea(first.rows, first.cols, first.rows === 1 ? 11 : 16);
      const spec = {key: 'qkv', tensor: first.tensor, width: PAD * 2 + kinds.length * area.width + (kinds.length - 1) * 16,
        height: TITLE + 14 + area.height + PAD};
      spec.draw = ctx => {
        frameCanvas(ctx, spec, blockName(block) + ' · Q / K / V', 'PCA colour');
        kinds.forEach((key, index) => {
          const x = PAD + index * (area.width + 16), part = data[key], rgb = bytesOf(part.rgb);
          ctx.fillStyle = '#5f7481';
          ctx.font = '600 10px ' + FONT;
          ctx.fillText(key.toUpperCase(), x, TITLE + 8);
          drawTokens(ctx, area, x, TITLE + 14, part.rows, part.cols, (r, c) => softColour(rgb, (r * part.cols + c) * 3), null);
        });
      };
      return spec;
    },
    attention: (data, block) => {
      if ((data.queries || data.size) < data.size) return clsAttentionSpec(data, block);
      const grid = data.rows === 1 && tokenArea(1, data.size, 1).side > 0;
      const cell = data.rows === 1 ? Math.max(3, Math.floor(210 / data.size)) : Math.max(8, Math.floor(126 / data.size));
      const matrix = data.size * cell, prompts = Array.isArray(inputs.text) ? inputs.text : [];
      const spec = {key: 'attention', tensor: data.tensor, width: PAD * 2 + data.rows * matrix + (data.rows - 1) * 16,
        height: TITLE + matrix + PAD + (data.rows > 1 ? 14 : 0)};
      spec.draw = ctx => {
        const head = chosenHead(data.heads), onImage = grid && view.style === 'image' && imageReady;
        frameCanvas(ctx, spec, blockName(block) + ' · attention' + (onImage ? ', CLS → patches' : ''),
          head === null ? 'mean of ' + data.heads + ' heads' : 'head ' + (head + 1) + ' of ' + data.heads);
        for (let row = 0; row < data.rows; row++) {
          const map = attentionMap(data, row, head), x = PAD + row * (matrix + 16);
          if (onImage) {
            drawAttentionOnImage(ctx, map, data.size, x, TITLE, matrix);
          } else {
            for (let q = 0; q < data.size; q++) {
              for (let k = 0; k < data.size; k++) {
                ctx.fillStyle = ramp(ATTENTION_RAMP, Math.pow(map[q * data.size + k], 0.35));
                ctx.fillRect(x + k * cell, TITLE + q * cell, cell, cell);
              }
            }
          }
          if (data.rows > 1) {
            ctx.fillStyle = '#8a9ba5';
            ctx.font = '9px ' + FONT;
            ctx.fillText(String(prompts[row] || 'row ' + (row + 1)).slice(0, Math.floor(matrix / 5.5)), x, TITLE + matrix + 11);
          }
        }
      };
      return spec;
    },
    // One map per prompt token; "On image" dims the image where the token is not attended.
    cross: (data, block) => {
      const [h, w] = data.grid, count = data.tokens.length, scale = Math.max(1, Math.floor(64 / Math.max(h, w)));
      const size = {w: w * scale, h: h * scale};
      const spec = {key: 'cross', tensor: data.tensor, width: PAD * 2 + count * size.w + (count - 1) * 8, height: TITLE + size.h + 26 + PAD};
      spec.draw = ctx => {
        const head = chosenHead(data.heads), onImage = view.style === 'image' && imageReady;
        frameCanvas(ctx, spec, blockName(block) + ' · cross-attention',
          head === null ? 'mean of ' + data.heads + ' heads' : 'head ' + (head + 1) + ' of ' + data.heads);
        for (let t = 0; t < count; t++) {
          const {values, low, high} = crossMap(data, t, head), x = PAD + t * (size.w + 8);
          if (onImage) ctx.drawImage(imageCanvas, x, TITLE, size.w, size.h);
          for (let i = 0; i < h * w; i++) {
            ctx.fillStyle = onImage ? 'rgba(46,84,122,' + (0.78 * (1 - values[i])).toFixed(3) + ')' : ramp(ATTENTION_RAMP, Math.pow(values[i], 0.5));
            ctx.fillRect(x + (i % w) * scale, TITLE + Math.floor(i / w) * scale, scale, scale);
          }
          ctx.font = '9px ' + FONT;
          ctx.fillStyle = '#4f6978';
          ctx.fillText(cleanToken(data.tokens[t]).slice(0, Math.max(3, Math.floor(size.w / 5.5))), x, TITLE + size.h + 11);
          ctx.fillStyle = '#8a9ba5';
          ctx.fillText(low.toFixed(2) + '–' + high.toFixed(2), x, TITLE + size.h + 22);
        }
      };
      return spec;
    },
    output: (data, block) => {
      const prompts = Array.isArray(inputs.text) ? inputs.text : [];
      if (data.threshold !== undefined) {
        const score = data.values[0][0], limit = Math.max(0.5, score * 1.2, data.threshold * 2), bar = 280;
        const spec = {key: 'output', tensor: data.tensor, width: 330, height: TITLE + 66 + PAD};
        spec.draw = ctx => {
          frameCanvas(ctx, spec, blockName(block) + ' · score', 'score vs threshold');
          const y = TITLE + 6, at = value => PAD + bar * Math.max(0, Math.min(1, value / limit));
          ctx.fillStyle = '#f0e8e0';
          ctx.fillRect(PAD, y, bar, 12);
          ctx.fillStyle = data.detected ? '#c29b7d' : '#a9bccb';
          ctx.fillRect(PAD, y, at(score) - PAD, 12);
          ctx.fillStyle = '#3f5a69';
          ctx.fillRect(at(data.threshold) - 1, y - 4, 2, 20);
          ctx.font = '10px ' + FONT;
          ctx.fillStyle = '#4f6978';
          ctx.fillText((data.score_type || 'score') + ' ' + score.toFixed(4) + ' · threshold ' + Number(data.threshold).toFixed(4), PAD, y + 32);
          ctx.font = '600 11px ' + FONT;
          ctx.fillStyle = data.detected ? '#765b48' : '#5f7481';
          ctx.fillText(data.detected ? 'detected: score ≥ threshold' : 'not detected: score < threshold', PAD, y + 50);
        };
        return spec;
      }
      if (data.probs) {
        const probs = data.probs[0] || [], logits = data.values[0] || [];
        const spec = {key: 'output', tensor: data.tensor, width: 330, height: TITLE + probs.length * 24 + PAD};
        spec.draw = ctx => {
          frameCanvas(ctx, spec, blockName(block) + ' · similarity', 'softmax');
          probs.forEach((p, j) => {
            const y = TITLE + j * 24;
            ctx.fillStyle = '#4f6978';
            ctx.font = '10px ' + FONT;
            ctx.fillText(String(prompts[j] || 'prompt ' + (j + 1)).slice(0, 22), PAD, y + 12);
            ctx.fillStyle = '#f0e8e0';
            ctx.fillRect(150, y + 3, 90, 10);
            ctx.fillStyle = '#c29b7d';
            ctx.fillRect(150, y + 3, 90 * p, 10);
            ctx.fillStyle = '#765b48';
            ctx.fillText(Math.round(p * 100) + '% · ' + Number(logits[j]).toFixed(1), 248, y + 12);
          });
        };
        return spec;
      }
      const rows = data.values, strip = 302;
      const spec = {key: 'output', tensor: data.tensor, width: 330, height: TITLE + rows.length * 30 + PAD};
      spec.draw = ctx => {
        frameCanvas(ctx, spec, blockName(block) + ' · output', rows[0].length + ' values');
        rows.forEach((row, r) => {
          const magnitudes = row.map(Math.abs).sort((a, b) => a - b);
          const y = TITLE + r * 30, limit = magnitudes[Math.floor(magnitudes.length * 0.98)] || magnitudes[magnitudes.length - 1] || 1;
          ctx.fillStyle = '#8a9ba5';
          ctx.font = '9px ' + FONT;
          if (rows.length > 1) ctx.fillText(String(prompts[r] || 'row ' + (r + 1)).slice(0, 48), PAD, y + 8);
          for (let i = 0; i < strip; i++) {
            const from = Math.floor(i * row.length / strip), to = Math.max(from + 1, Math.floor((i + 1) * row.length / strip));
            const mean = row.slice(from, to).reduce((sum, value) => sum + value, 0) / (to - from);
            ctx.fillStyle = diverging(mean / limit);
            ctx.fillRect(PAD + i, y + 12, 1, 12);
          }
        });
      };
      return spec;
    },
  };

  function drawPanel(panel) {
    const ctx = panel.canvas.getContext('2d');
    ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.clearRect(0, 0, panel.spec.width, panel.spec.height);
    panel.spec.draw(ctx);
    if (panel.onDraw) panel.onDraw();
  }

  function redrawPanels(key) {
    renderers.forEach(renderer => {
      if (!renderer.objects) return;
      renderer.objects.forEach(object => object.panels.forEach(panel => { if (panel.key === key) drawPanel(panel); }));
      if (renderer.requestRender) renderer.requestRender();
    });
  }

  // The panels an opened block shows for the enabled types, each drawn on its own canvas;
  // ``x`` / ``y`` are offsets from the block centre in world units, set by ``arrangePanels``.
  function makePanels(object) {
    if (!view.open.has(object.id)) return [];
    const data = views[object.id] || {}, block = blocks[object.id];
    return TYPES.filter(([key]) => view.types.has(key) && data[key]).map(([key]) => {
      const spec = SPECS[key](data[key], block), canvas = document.createElement('canvas');
      canvas.width = spec.width * 2;
      canvas.height = spec.height * 2;
      const panel = {key, spec, canvas, width: spec.width * PX, height: spec.height * PX, x: 0, y: 0};
      drawPanel(panel);
      return panel;
    });
  }

  // Panels fill rows of at most two, stacked away from the block (and from the other lane).
  function arrangePanels(object) {
    const panels = object.panels, direction = object.stack.direction, columns = panels.length > 2 ? 2 : 1;
    let offset = object.size.h / 2 + PANEL_LIFT, width = 0;
    for (let start = 0; start < panels.length; start += columns) {
      const row = panels.slice(start, start + columns);
      const rowWidth = row.reduce((sum, panel) => sum + panel.width, 0) + (row.length - 1) * PANEL_GAP;
      const rowHeight = Math.max(...row.map(panel => panel.height));
      let x = -rowWidth / 2;
      row.forEach(panel => {
        panel.x = x + panel.width / 2;
        panel.y = direction * (offset + rowHeight / 2);
        x += panel.width + PANEL_GAP;
      });
      offset += rowHeight + PANEL_GAP;
      width = Math.max(width, rowWidth);
    }
    object.stack.width = width;
    object.stack.height = panels.length ? offset - PANEL_GAP - object.size.h / 2 : 0;
  }

  // ---- Layout in world units over a renderer's objects: {id, kind, lane, size, target, shown, stack}.

  // The player's Branch filter (``state.branch``) keeps one lane, or all of them.
  const laneShown = lane => state.branch === 'all' || lane.id === state.branch;
  const blockShown = id => state.branch === 'all' || blocks[id].lane === state.branch;

  // The object that stands for a block: itself, or its group while the group is collapsed.
  function unitOf(objects, id) {
    const block = blocks[id];
    return objects.get(block.group && !view.expanded.has(block.group) ? block.group : id);
  }

  // The visible objects of a lane in order: blocks, collapsed groups, or an expanded group's members.
  function laneUnits(objects, lane) {
    return lane.items.flatMap(item => {
      if (!item.group) return [objects.get(item.block)];
      return view.expanded.has(item.group) ? groups[item.group].blocks.map(id => objects.get(id)) : [objects.get(item.group)];
    });
  }

  // Flow lanes stack from the top, spaced so that open panels clear the next lane. A lane fed by an
  // earlier one (``model.links``) starts just after the block that feeds it, so features run on into
  // the lane that reads them; a lane nothing feeds starts under the lane above it. Result-only lanes
  // follow at the end, level with the lanes that feed them.
  function layout(objects) {
    const flow = flowLanes.filter(laneShown), merge = mergeLanes.filter(laneShown), links = model.links || [];
    lanes.filter(lane => !laneShown(lane)).forEach(lane => lane.items.forEach(item => {
      (item.group ? [item.group, ...groups[item.group].blocks] : [item.block]).forEach(id => { objects.get(id).shown = false; });
    }));
    // How far a lane reaches above (``up``) or below its centre line, panels included.
    const reach = (lane, up) => Math.max(0.25, ...laneUnits(objects, lane).map(unit =>
      unit.size.h / 2 + ((unit.stack.direction > 0) === up ? unit.stack.height : 0)));
    const laneY = [];
    flow.forEach((lane, index) => {
      const previous = laneY[index - 1], below = reach(lane, false);
      laneY.push({y: previous ? previous.y - previous.below - LANE_GAP - reach(lane, true) : 0, below});
    });
    const middle = laneY.length ? (laneY[0].y + laneY[laneY.length - 1].y) / 2 : 0;
    let end = 0, start = 0;
    const place = (lane, y, x) => {
      lane.items.forEach(item => {
        const width = extent(objects, item);
        placeItem(objects, item, x + width / 2, y);
        x += width + GAP;
      });
      return x;
    };
    flow.forEach((lane, index) => {
      const earlier = new Set(flow.slice(0, index).map(item => item.id));
      const feeds = links.filter(([from, to]) => blocks[to].lane === lane.id && earlier.has(blocks[from].lane))
        .map(([from]) => unitOf(objects, from)).map(unit => unit.target.x + widthOf(unit) / 2 + GAP);
      start = feeds.length ? Math.max(0, ...feeds) : start;
      end = Math.max(end, place(lane, laneY[index].y - middle, start));
    });
    merge.forEach(lane => {
      const sources = flow.map((item, index) => [item, index])
        .filter(([item]) => links.some(([from, to]) => blocks[from].lane === item.id && blocks[to].lane === lane.id));
      const y = sources.length ? sources.reduce((sum, [, index]) => sum + laneY[index].y - middle, 0) / sources.length : 0;
      place(lane, y, end);
    });
  }

  function extent(objects, item) {
    if (!item.group) return widthOf(objects.get(item.block));
    const group = groups[item.group], count = group.blocks.length;
    return view.expanded.has(group.id)
      ? group.blocks.reduce((sum, id) => sum + widthOf(objects.get(id)), 0) + (count - 1) * LAYER_GAP
      : objects.get(group.id).size.w;
  }

  function placeItem(objects, item, x, y) {
    if (!item.group) {
      const object = objects.get(item.block);
      object.target.x = x;
      object.target.y = y;
      object.shown = true;
      return;
    }
    const group = groups[item.group], box = objects.get(group.id), expanded = view.expanded.has(group.id);
    box.target.x = x;
    box.target.y = y;
    box.shown = !expanded;
    let cursor = x - extent(objects, item) / 2;
    group.blocks.forEach(id => {
      const member = objects.get(id), width = widthOf(member);
      member.shown = expanded;
      member.target.x = expanded ? cursor + width / 2 : x;
      member.target.y = y;
      cursor += width + LAYER_GAP;
    });
  }

  // Links join consecutive visible units in a lane, plus the recorded cross-lane Tensor links.
  function linkPairs(objects) {
    const pairs = [];
    lanes.filter(laneShown).forEach(lane => {
      const chain = laneUnits(objects, lane);
      chain.slice(1).forEach((unit, index) => pairs.push([chain[index], unit]));
    });
    (model.links || []).forEach(([from, to]) => {
      if (blockShown(from) && blockShown(to)) pairs.push([unitOf(objects, from), unitOf(objects, to)]);
    });
    return pairs;
  }

  function topOf(object) {
    return object.size.h / 2 + (object.stack.direction > 0 ? object.stack.height : 0);
  }

  // World-space corners of an object with its panel stack (and room above for its label).
  function cornersOf(object) {
    const t = object.target, s = object.size, half = widthOf(object) / 2, list = [];
    const top = topOf(object) + 1.0, bottom = -s.h / 2 - (object.stack.direction < 0 ? object.stack.height : 0);
    [-half, half].forEach(x => [bottom, top].forEach(y => [-s.d / 2, s.d / 2].forEach(z => list.push({x: t.x + x, y: t.y + y, z}))));
    return list;
  }

  // Objects a full-view fit frames: everything shown. Neither view zooms onto opened blocks.
  function framed(objects) {
    return [...objects.values()].filter(object => object.shown);
  }

  // Smallest shift that brings the span of ``values`` inside [low, high]; a span too long for it
  // keeps its high end (``keepHigh``) or its low end in view. Both views pan new panels into sight with it.
  function nudge(values, low, high, keepHigh) {
    const min = Math.min(...values), max = Math.max(...values);
    if (max - min > high - low) return keepHigh ? high - max : low - min;
    if (min < low) return low - min;
    return max > high ? high - max : 0;
  }

  // ---- Labels: DOM elements over each renderer's stage; anchors follow ``object.current``.

  function addLabel(list, layer, text, anchor, priority, className, onClick) {
    const element = onClick ? node('button', text, 'scene-label ' + className) : node('span', text, 'scene-label ' + className);
    if (onClick) {
      element.type = 'button';
      element.addEventListener('click', onClick);
    }
    layer.appendChild(element);
    list.push({element, anchor, priority});
    return element;
  }

  const above = (object, lift) => ({x: object.current.x, y: object.current.y + topOf(object) + lift, z: 0});
  // World y of an expanded group's label: above its tallest layer, panels included.
  const groupTop = members => members[0].current.y + Math.max(...members.map(topOf)) + 0.7;

  // Static labels name lanes, groups, inputs and results; other blocks are named only
  // while hovered or active, which keeps the default view to a handful of words.
  function buildLabels(objects, layer) {
    const list = [];
    lanes.forEach(lane => {
      const first = lane.items[0], firstId = first && (first.group || first.block);
      addLabel(list, layer, lane.id, () => {
        const object = objects.get(firstId);
        return object && object.visible() ? {x: object.current.x - widthOf(object) / 2, y: object.current.y + topOf(object) + 1.1, z: 0} : null;
      }, 0, 'lane');
    });
    Object.values(groups).forEach(group => {
      const element = addLabel(list, layer, '', () => {
        const members = group.blocks.map(id => objects.get(id)), box = objects.get(group.id);
        if (!view.expanded.has(group.id)) return box.visible() ? above(box, 0.7) : null;
        const first = members[0].current, last = members[members.length - 1].current;
        return {x: (first.x + last.x) / 2, y: groupTop(members), z: 0};
      }, 1, 'group', () => toggleGroup(group.id));
      element.dataset.group = group.id;
      objects.get(group.id).labelElement = element;
      group.blocks.forEach(id => addLabel(list, layer, blocks[id].label, () => {
        const object = objects.get(id);
        return object.visible() ? above(object, 0.25) : null;
      }, 3, 'minor'));
    });
    Object.values(blocks).filter(block => !block.group).forEach(block => {
      const always = block.kind !== 'module';
      addLabel(list, layer, block.label, () => {
        const object = objects.get(block.id), named = always || view.active === block.id || view.hover === block.id;
        return named && object.visible() ? above(object, 0.3) : null;
      }, always ? 2 : 1, 'block');
    });
    refreshGroupLabels(objects);
    return list;
  }

  function refreshGroupLabels(objects) {
    Object.values(groups).forEach(group => {
      const element = objects.get(group.id).labelElement;
      element.textContent = group.label + ' ×' + group.blocks.length + (view.expanded.has(group.id) ? '  −' : '  +');
      element.setAttribute('aria-expanded', String(view.expanded.has(group.id)));
    });
  }

  // ``project`` maps a world point to {x, y} inside ``rect``, or null when it is off screen.
  function placeLabels(list, rect, project) {
    const placed = [];
    list.slice().sort((a, b) => a.priority - b.priority).forEach(label => {
      const anchor = label.anchor(), element = label.element;
      const point = anchor && project(anchor, rect);
      if (!point) {
        element.hidden = true;
        return;
      }
      element.hidden = false;
      element.style.left = point.x + 'px';
      element.style.top = point.y + 'px';
      const width = element.offsetWidth, height = element.offsetHeight;
      const left = element.classList.contains('lane') ? point.x : point.x - width / 2;
      const box = {left, right: left + width, top: point.y - height, bottom: point.y};
      // Lower-priority labels give way instead of overlapping (lane > group > block > layer index).
      if (placed.some(other => box.left < other.right + 2 && box.right > other.left - 2 && box.top < other.bottom + 1 && box.bottom > other.top - 1)) {
        element.hidden = true;
        return;
      }
      placed.push(box);
    });
  }

  // ---- Actions: they change the shared view state, then every started renderer follows.

  const started = () => [...renderers.values()].filter(renderer => renderer.started && renderer.started());
  const current = () => renderers.get(view.mode);

  function toggleGroup(id) {
    if (view.expanded.has(id)) view.expanded.delete(id);
    else view.expanded.add(id);
    started().forEach(renderer => renderer.relayout(false));
    sync();
  }

  function toggleOpen(id) {
    if (view.open.has(id)) view.open.delete(id);
    else view.open.add(id);
    // Both views keep their zoom and hold the clicked block in place.
    started().forEach(renderer => {
      renderer.rebuildPanels(id);
      renderer.relayout(false, id);
    });
    sync();
  }

  function collapseAll() {
    view.expanded.clear();
    view.open.clear();
    started().forEach(renderer => {
      renderer.rebuildPanels();
      renderer.relayout(true);
    });
    sync();
  }

  function setType(key) {
    if (view.types.has(key)) view.types.delete(key);
    else view.types.add(key);
    refreshBar();
    started().forEach(renderer => {
      renderer.rebuildPanels();
      renderer.relayout(false, view.active);
    });
    sync();
  }

  function setHead(value) {
    view.head = value;
    refreshBar();
    redrawPanels('attention');
    redrawPanels('cross');
  }

  // "On image" draws from the shared image canvas; loading it redraws the attention panels.
  function setStyle(value) {
    view.style = value;
    if (value === 'image') sourceImage();
    refreshBar();
    redrawPanels('attention');
    redrawPanels('cross');
  }

  function activeId() {
    let id = state.operation && model.op_blocks[state.operation] || null;
    if (!id && state.tensor) id = (Object.values(blocks).find(block => block.tensor === state.tensor) || {}).id || null;
    const block = id && blocks[id];
    return block && block.group && !view.expanded.has(block.group) ? block.group : id;
  }

  // Called by the 2D player whenever its selection, timeline position or Branch filter changes.
  function sync() {
    if (state.branch !== lastBranch) {
      lastBranch = state.branch;
      started().forEach(renderer => renderer.relayout(true));
    }
    view.active = activeId();
    started().forEach(renderer => renderer.repaint());
  }

  // Bring a block into view for a list outside the graph (Full data path, Data flow): its group opens so
  // the block itself shows, then the visible view moves its camera there. Clicks inside the graph
  // never call this, so they keep the camera where it is.
  function focus(id) {
    const block = blocks[id];
    if (!block) return;
    if (block.group && !view.expanded.has(block.group)) {
      view.expanded.add(block.group);
      started().forEach(renderer => renderer.relayout(false));
    }
    sync();
    const renderer = current();
    if (renderer && renderer.focus) renderer.focus(id);
  }

  function hover(id) {
    if (id === view.hover) return;
    view.hover = id;
    started().forEach(renderer => renderer.repaint());
  }

  // A pick from either renderer: {kind: 'block'|'group'|'panel', id, patch?, key?, tensor?}.
  function activate(pick) {
    if (pick.kind === 'panel') {
      selectTensor(pick.tensor);
      return;
    }
    if (pick.patch !== undefined) {
      selectPatch(pick.patch);
      return;
    }
    if (pick.kind === 'group') {
      toggleGroup(pick.id);
      return;
    }
    const block = blocks[pick.id];
    if (block.kind === 'module' || !block.tensor) selectOperation(block.operations[0]);
    else selectTensor(block.tensor);
    if (views[block.id]) toggleOpen(block.id);
  }

  // The old view hides first so the new one measures its real size; a renderer's ``show``
  // returns false when it cannot start (for example without WebGL), and the old view returns.
  function setMode(mode) {
    const target = renderers.get(mode), previous = current();
    if (!target) return;
    if (previous && previous !== target) previous.hide();
    if (!target.show()) {
      if (previous && previous !== target) previous.show();
      return;
    }
    view.mode = mode;
    document.querySelector('.center').classList.toggle('mode-3d', mode === '3d');
    [['view-2d', mode === '2d'], ['view-3d', mode === '3d']].forEach(([id, on]) => {
      E(id).classList.toggle('active', on);
      E(id).setAttribute('aria-pressed', String(on));
    });
    sync();
  }

  function register(mode, renderer) {
    renderers.set(mode, renderer);
  }

  // ---- Toolbar: value switches, head and attention style, collapse and fit.

  const typeButtons = new Map(), styleButtons = new Map();
  let headSelect = null;

  function buildBar() {
    const bar = E('scene-bar');
    // A value type this trace never recorded gets no switch at all.
    TYPES.filter(([key]) => Object.values(views).some(item => item[key])).forEach(([key, label]) => {
      const button = node('button', label, 'scene-type');
      button.type = 'button';
      button.dataset.type = key;
      button.title = 'Show on opened blocks';
      button.addEventListener('click', () => setType(key));
      typeButtons.set(key, button);
      bar.appendChild(button);
    });
    const heads = Math.max(0, ...Object.values(views).flatMap(item => [item.attention, item.cross]).map(data => data ? data.heads : 0));
    const headWrap = node('label', 'Head', 'scene-head');
    headSelect = document.createElement('select');
    headSelect.id = 'scene-head';
    [['avg', 'mean']].concat(Array.from({length: heads}, (_, index) => [String(index + 1), String(index + 1)])).forEach(([value, text]) => {
      const option = node('option', text);
      option.value = value;
      headSelect.appendChild(option);
    });
    headSelect.addEventListener('change', () => setHead(headSelect.value));
    headWrap.appendChild(headSelect);
    const style = node('div', undefined, 'scene-style');
    [['matrix', 'Matrix'], ['image', 'On image']].forEach(([value, text]) => {
      const button = node('button', text);
      button.type = 'button';
      button.dataset.style = value;
      button.addEventListener('click', () => setStyle(value));
      styleButtons.set(value, button);
      style.appendChild(button);
    });
    // Head and style only steer attention panels, so a trace without them shows neither.
    headWrap.hidden = style.hidden = !heads;
    const collapse = node('button', 'Collapse all'), fit = node('button', 'Fit');
    collapse.type = fit.type = 'button';
    collapse.addEventListener('click', collapseAll);
    fit.addEventListener('click', () => { if (current()) current().fitAll(); });
    bar.append(headWrap, style, node('span', undefined, 'spacer'), collapse, fit);
    refreshBar();
  }

  function refreshBar() {
    typeButtons.forEach((button, key) => {
      const on = view.types.has(key);
      button.classList.toggle('active', on);
      button.setAttribute('aria-pressed', String(on));
    });
    const attention = ['attention', 'cross'].some(key => view.types.has(key) && typeButtons.has(key));
    headSelect.disabled = !attention;
    headSelect.value = view.head;
    styleButtons.forEach((button, value) => {
      button.disabled = !attention || (value === 'image' && !model.image);
      button.classList.toggle('active', view.style === value);
    });
  }

  // ---- Test hooks: the state plus whatever the visible renderer reports.

  function snapshot() {
    const visible = current();
    return Object.assign({
      mode: view.mode, expanded: [...view.expanded], open: [...view.open], types: [...view.types],
      head: view.head, style: view.style, active: view.active,
    }, visible && visible.snapshot ? visible.snapshot() : {});
  }

  // Scripts finish before DOMContentLoaded, so every renderer has registered by then.
  function start() {
    buildBar();
    E('view-2d').addEventListener('click', () => setMode('2d'));
    E('view-3d').addEventListener('click', () => setMode('3d'));
    renderers.forEach(renderer => { if (renderer.prepare) renderer.prepare(); });
    window.xrayScene = {
      sync, setMode, toggleGroup, toggleOpen, collapseAll, snapshot, focus,
      pointFor: (id, target) => current() && current().pointFor ? current().pointFor(id, target) : null,
      painted: () => renderers.get('3d') ? renderers.get('3d').painted() : 0,
    };
    const three = renderers.get('3d');
    setMode(location.hash === '#3d' && lanes.length && three && three.available() ? '3d' : '2d');
  }

  document.addEventListener('DOMContentLoaded', start);
  window.XRAY_SCENE = {
    model, blocks, groups, lanes, view, TILE_GAP, PLATE, PLATE_GAP,
    palette, inputKind, slabSize, patchGrid, groupSize, sourceImage, tokenCanvas,
    makePanels, arrangePanels, panelDirection, widthOf, topOf, groupTop, cornersOf, framed, nudge,
    layout, linkPairs, buildLabels, refreshGroupLabels, placeLabels,
    register, activate, hover,
  };
})();
