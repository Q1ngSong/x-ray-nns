// 2D view: the shared scene layout (XRAY_SCENE) seen from the front, drawn with DOM buttons,
// SVG links and the core's panel canvases on a plane that pans and zooms. One world unit is UNIT
// pixels at zoom 1, so intermediate-value panels appear at their native canvas resolution.
(function () {
  'use strict';
  const S = window.XRAY_SCENE;
  const {blocks, groups, view} = S;
  // Blocks are wider than the 3D slabs (0.3) so they stay easy to click when seen head-on.
  const UNIT = 60, BLOCK = 0.8, MIN_ZOOM = 0.1, MAX_ZOOM = 3, MAX_FIT = 1.5, FOCUS_ZOOM = 0.8;
  // An expanded group's dashed outline: world padding around its layers and panels, and a gap in
  // screen pixels between its top edge and the group label above it.
  const FRAME_PAD = 0.3, FRAME_GAP = 5;
  // Pixels kept clear at the plane's edges when a click pans new panels into view.
  const EDGE = 12;
  const stageEl = E('graph-stage'), plane = E('plane'), worldEl = E('plane-world'), linkLayer = E('plane-links'), labelLayer = E('plane-labels');
  const objects = new Map(), frames = new Map(), camera = {x: 0, y: 0, s: 1};
  let labels = [], linkPairs = [], links = [], ready = false, fitted = false, needsFit = false, frame = 0, tweens = [];

  const px = value => value * UNIT;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function button(className, pick, label) {
    const element = node('button', undefined, className);
    element.type = 'button';
    element.setAttribute('aria-label', label);
    element.addEventListener('click', () => S.activate(pick));
    return element;
  }

  function buildBlock(block) {
    const kind = S.inputKind(block), pick = {kind: 'block', id: block.id};
    const title = block.label + ' · [' + (block.shape || []).join(' × ') + ']';
    let element, size;
    if (kind === 'image') {
      size = {w: 2.6, h: 2.6, d: 0};
      element = button('b2d input ' + block.lane, pick, 'Input image');
      if (S.model.image) {
        const image = node('img');
        image.src = S.model.image;
        image.alt = '';
        image.draggable = false;
        element.appendChild(image);
      }
    } else if (kind === 'patches') {
      const {rows, cols, tile, size: gridSize} = S.patchGrid();
      size = gridSize;
      element = node('div', undefined, 'b2d patches ' + block.lane);
      for (let index = 0; index < rows * cols; index++) {
        const r = Math.floor(index / cols), c = index % cols;
        const cell = button('tile', {kind: 'block', id: block.id, patch: index}, 'Image patch ' + (index + 1));
        cell.dataset.patch = String(index);
        Object.assign(cell.style, {left: px(c * (tile + S.TILE_GAP)) + 'px', top: px(r * (tile + S.TILE_GAP)) + 'px',
          width: px(tile) + 'px', height: px(tile) + 'px'});
        if (S.model.image) {
          Object.assign(cell.style, {backgroundImage: 'url("' + S.model.image + '")', backgroundSize: px(tile) * cols + 'px ' + px(tile) * rows + 'px',
            backgroundPosition: -px(tile) * c + 'px ' + -px(tile) * r + 'px'});
        }
        element.appendChild(cell);
      }
    } else if (kind === 'tokens') {
      const tokens = S.tokenCanvas(block);
      size = {w: tokens.width, h: tokens.height, d: 0};
      element = button('b2d input ' + block.lane, pick, 'Tokens');
      tokens.canvas.style.width = px(size.w) + 'px';
      tokens.canvas.style.height = px(size.h) + 'px';
      element.appendChild(tokens.canvas);
    } else {
      const colours = S.palette(block.lane);
      size = Object.assign(S.slabSize(block), {w: BLOCK});
      element = button('b2d ' + block.lane, pick, title);
      Object.assign(element.style, {background: colours.face, borderColor: colours.edge});
    }
    element.dataset.block = block.id;
    element.title = title;
    element.style.width = px(size.w) + 'px';
    element.style.height = px(size.h) + 'px';
    element.addEventListener('pointerenter', () => S.hover(block.id));
    element.addEventListener('pointerleave', () => S.hover(null));
    worldEl.appendChild(element);
    return {id: block.id, kind: 'block', lane: block.lane, element, size, target: {x: 0, y: 0}, current: {x: 0, y: 0},
      shown: true, visible: () => !element.hidden, panels: [], stack: {width: 0, height: 0, direction: S.panelDirection(block.lane)}};
  }

  // A collapsed group is the same tight stack of thin plates as in 3D, seen head-on.
  function buildGroup(group) {
    const member = objects.get(group.blocks[0]).size, size = S.groupSize(group, member), colours = S.palette(group.lane);
    const element = button('g2d ' + group.lane, {kind: 'group', id: group.id}, group.label + ' ×' + group.blocks.length);
    element.dataset.group = group.id;
    group.blocks.forEach((_, index) => {
      const plate = node('span', undefined, 'plate');
      Object.assign(plate.style, {left: px(index * (S.PLATE + S.PLATE_GAP)) + 'px', width: px(S.PLATE) + 'px',
        background: colours.face, borderColor: colours.edge});
      element.appendChild(plate);
    });
    element.style.width = px(size.w) + 'px';
    element.style.height = px(size.h) + 'px';
    worldEl.appendChild(element);
    return {id: group.id, kind: 'group', lane: group.lane, element, size, target: {x: 0, y: 0}, current: {x: 0, y: 0},
      shown: true, visible: () => !element.hidden, panels: [], stack: {width: 0, height: 0, direction: 1}};
  }

  // Panels are the core's canvases, placed in the world at their native size.
  function rebuildPanels(id) {
    (id === undefined ? [...objects.values()] : [objects.get(id)]).forEach(object => {
      object.panels.forEach(panel => panel.canvas.remove());
      object.panels = S.makePanels(object);
      object.panels.forEach(panel => {
        const pick = {kind: 'panel', id: object.id, key: panel.key, tensor: panel.spec.tensor};
        Object.assign(panel.canvas, {className: 'p2d', tabIndex: 0, hidden: object.element.hidden});
        panel.canvas.dataset.panel = object.id + ':' + panel.key;
        panel.canvas.setAttribute('role', 'button');
        panel.canvas.setAttribute('aria-label', 'Inspect ' + panel.key + ' of ' + blocks[object.id].label);
        panel.canvas.style.width = px(panel.width) + 'px';
        panel.canvas.style.height = px(panel.height) + 'px';
        panel.canvas.addEventListener('click', () => S.activate(pick));
        panel.canvas.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            S.activate(pick);
          }
        });
        worldEl.appendChild(panel.canvas);
      });
      S.arrangePanels(object);
    });
  }

  function setHidden(object, hidden) {
    object.element.hidden = hidden;
    object.panels.forEach(panel => { panel.canvas.hidden = hidden; });
  }

  function rebuildLinks() {
    links.forEach(line => line.remove());
    linkPairs = S.linkPairs(objects);
    links = linkPairs.map(() => {
      const line = document.createElementNS(linkLayer.namespaceURI, 'line');
      linkLayer.appendChild(line);
      return line;
    });
  }

  // World units run up the y axis; screen pixels run down.
  function draw() {
    worldEl.style.transform = 'translate(' + camera.x + 'px,' + camera.y + 'px) scale(' + camera.s + ')';
    objects.forEach(object => {
      const {x, y} = object.current;
      object.element.style.left = px(x - object.size.w / 2) + 'px';
      object.element.style.top = px(-y - object.size.h / 2) + 'px';
      object.panels.forEach(panel => {
        panel.canvas.style.left = px(x + panel.x - panel.width / 2) + 'px';
        panel.canvas.style.top = px(-(y + panel.y) - panel.height / 2) + 'px';
      });
    });
    // Expanded groups keep a light dashed outline, one screen pixel wide, around their layers and
    // opened panels; its top edge stops just under the group label.
    frames.forEach((outline, id) => {
      const members = groups[id].blocks.map(member => objects.get(member)).filter(object => object.shown);
      outline.hidden = !view.expanded.has(id) || !members.length;
      if (outline.hidden) return;
      const left = Math.min(...members.map(object => object.current.x - S.widthOf(object) / 2)) - FRAME_PAD;
      const right = Math.max(...members.map(object => object.current.x + S.widthOf(object) / 2)) + FRAME_PAD;
      // Near the minimum zoom the label gap would cut into the tallest layer, so keep 2 pixels above it.
      const content = Math.max(...members.map(object => object.current.y + S.topOf(object)));
      const top = Math.max(content + 2 / (UNIT * camera.s), S.groupTop(members) - FRAME_GAP / (UNIT * camera.s));
      const bottom = Math.min(...members.map(object => object.current.y - object.size.h / 2
        - (object.stack.direction < 0 ? object.stack.height : 0))) - FRAME_PAD;
      Object.assign(outline.style, {left: px(left) + 'px', top: px(-top) + 'px', width: px(right - left) + 'px',
        height: px(top - bottom) + 'px', borderWidth: 1 / camera.s + 'px'});
    });
    linkPairs.forEach(([from, to], index) => {
      const line = links[index];
      line.setAttribute('x1', px(from.current.x + from.size.w / 2));
      line.setAttribute('y1', px(-from.current.y));
      line.setAttribute('x2', px(to.current.x - to.size.w / 2));
      line.setAttribute('y2', px(-to.current.y));
    });
    S.placeLabels(labels, plane.getBoundingClientRect(), project);
  }

  function project(point, rect) {
    const x = px(point.x) * camera.s + camera.x, y = px(-point.y) * camera.s + camera.y;
    return x < -20 || y < -20 || x > rect.width + 20 || y > rect.height + 20 ? null : {x, y};
  }

  function requestDraw() {
    if (!frame) frame = requestAnimationFrame(loop);
  }

  function loop() {
    frame = 0;
    const now = performance.now();
    tweens = tweens.filter(item => {
      const progress = Math.min(1, (now - item.start) / item.duration);
      item.step(1 - Math.pow(1 - progress, 3));
      if (progress < 1) return true;
      if (item.done) item.done();
      return false;
    });
    draw();
    if (tweens.length) requestDraw();
  }

  function tween(duration, step, done, kind) {
    tweens.push({start: performance.now(), duration, step, done, kind});
    requestDraw();
  }

  // Groups appear at the end of a move and vanish at its start; members do the reverse,
  // so expanding layers slide out of the plate stack and collapsing ones slide back in.
  function applyLayout(animate) {
    S.layout(objects);
    const moves = [];
    objects.forEach(object => {
      if (object.shown && object.element.hidden && object.kind === 'block' && blocks[object.id].group) {
        Object.assign(object.current, objects.get(blocks[object.id].group).current);
        setHidden(object, false);
      }
      if (!object.shown && object.kind === 'group') setHidden(object, true);
      moves.push({object, from: Object.assign({}, object.current), to: Object.assign({}, object.target)});
    });
    rebuildLinks();
    S.refreshGroupLabels(objects);
    const finish = () => moves.forEach(({object}) => {
      Object.assign(object.current, object.target);
      setHidden(object, !object.shown);
    });
    if (!animate || !fitted) {
      finish();
      requestDraw();
      return;
    }
    tween(380, eased => moves.forEach(({object, from, to}) => {
      object.current.x = from.x + (to.x - from.x) * eased;
      object.current.y = from.y + (to.y - from.y) * eased;
    }), finish);
  }

  // Frame everything shown: fit its bounds into the plane, centred. Clicks never call this; they
  // keep the zoom (see ``hold``).
  function fit(animate) {
    const corners = S.framed(objects).flatMap(S.cornersOf);
    if (!corners.length) return;
    const xs = corners.map(corner => px(corner.x)), ys = corners.map(corner => px(-corner.y));
    const [left, right, top, bottom] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const width = Math.max(1, plane.clientWidth), height = Math.max(1, plane.clientHeight);
    const s = clamp(Math.min(width / (right - left), height / (bottom - top)) * 0.92, MIN_ZOOM, MAX_FIT);
    const goal = {s, x: width / 2 - (left + right) / 2 * s, y: height / 2 - (top + bottom) / 2 * s};
    tweens = tweens.filter(item => item.kind !== 'camera');
    if (!animate || !fitted) {
      Object.assign(camera, goal);
      fitted = true;
      requestDraw();
      return;
    }
    const from = Object.assign({}, camera);
    tween(450, eased => {
      camera.x = from.x + (goal.x - from.x) * eased;
      camera.y = from.y + (goal.y - from.y) * eased;
      camera.s = from.s + (goal.s - from.s) * eased;
    }, null, 'camera');
  }

  // While the 3D view is showing, layout changes land without animation and refit on show.
  // ``everything`` (Collapse all, a Branch change) refits the whole scene; otherwise the zoom and
  // the camera stay, except that ``anchor`` (the clicked block) keeps its place on screen.
  function relayout(everything, anchor) {
    if (stageEl.hidden) {
      applyLayout(false);
      needsFit = true;
      return;
    }
    const before = anchor ? onScreen(anchor) : null;
    applyLayout(true);
    if (everything) fit(true);
    else if (before) hold(objects.get(anchor), before);
  }

  // Where an object is drawn now, in plane pixels; null when it is hidden.
  function onScreen(id) {
    const object = objects.get(id);
    return object && object.shown ? {x: px(object.current.x) * camera.s + camera.x, y: px(-object.current.y) * camera.s + camera.y} : null;
  }

  // Keep the zoom: pan so ``object`` lands where it was drawn, then only as far as needed to bring
  // its panels into the plane. A stack too big for the plane keeps the block's end in view.
  function hold(object, before) {
    if (!object.shown) return;
    const s = camera.s, goal = {x: before.x - px(object.target.x) * s, y: before.y - px(-object.target.y) * s};
    if (object.panels.length) {
      const corners = S.cornersOf(object);
      goal.x += S.nudge(corners.map(corner => px(corner.x) * s + goal.x), EDGE, plane.clientWidth - EDGE, false);
      goal.y += S.nudge(corners.map(corner => px(-corner.y) * s + goal.y), EDGE, plane.clientHeight - EDGE, object.stack.direction > 0);
    }
    tweens = tweens.filter(item => item.kind !== 'camera');
    const from = Object.assign({}, camera);
    tween(450, eased => {
      camera.x = from.x + (goal.x - from.x) * eased;
      camera.y = from.y + (goal.y - from.y) * eased;
    }, null, 'camera');
  }

  // Centre ``id`` in the plane for a list outside the graph (Full data path, Data flow); the zoom only
  // grows, to FOCUS_ZOOM, when the block would otherwise be too small to read.
  function focus(id) {
    const object = objects.get(id);
    if (!ready || stageEl.hidden || !object || !object.shown) return;
    const s = Math.max(camera.s, FOCUS_ZOOM);
    const goal = {s, x: plane.clientWidth / 2 - px(object.target.x) * s, y: plane.clientHeight / 2 - px(-object.target.y) * s};
    tweens = tweens.filter(item => item.kind !== 'camera');
    const from = Object.assign({}, camera);
    tween(450, eased => {
      camera.x = from.x + (goal.x - from.x) * eased;
      camera.y = from.y + (goal.y - from.y) * eased;
      camera.s = from.s + (goal.s - from.s) * eased;
    }, null, 'camera');
  }

  function paint() {
    objects.forEach(object => {
      const active = object.id === view.active;
      object.element.classList.toggle('active', active);
      object.element.classList.toggle('hover', !active && object.id === view.hover);
    });
    worldEl.querySelectorAll('.tile').forEach(cell => cell.classList.toggle('selected', Number(cell.dataset.patch) === state.patchIndex));
    requestDraw();
  }

  // Drag the empty plane to pan; the wheel (or a pinch) zooms towards the cursor.
  function bindPlane() {
    let drag = null;
    plane.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('button, .p2d')) return;
      tweens = tweens.filter(item => item.kind !== 'camera');
      drag = {x: event.clientX, y: event.clientY, from: Object.assign({}, camera)};
      plane.setPointerCapture(event.pointerId);
      plane.classList.add('panning');
    });
    plane.addEventListener('pointermove', event => {
      if (!drag) return;
      camera.x = drag.from.x + event.clientX - drag.x;
      camera.y = drag.from.y + event.clientY - drag.y;
      requestDraw();
    });
    const release = () => {
      drag = null;
      plane.classList.remove('panning');
    };
    plane.addEventListener('pointerup', release);
    plane.addEventListener('pointercancel', release);
    plane.addEventListener('wheel', event => {
      event.preventDefault();
      tweens = tweens.filter(item => item.kind !== 'camera');
      const rect = plane.getBoundingClientRect(), cx = event.clientX - rect.left, cy = event.clientY - rect.top;
      const s = clamp(camera.s * Math.exp(-event.deltaY * 0.0015), MIN_ZOOM, MAX_ZOOM);
      camera.x = cx - (cx - camera.x) * s / camera.s;
      camera.y = cy - (cy - camera.y) * s / camera.s;
      camera.s = s;
      requestDraw();
    }, {passive: false});
    // Folding a side panel resizes the plane; the camera shifts by half the change so the graph
    // keeps its place in the middle instead of sliding with the plane's left edge.
    let size = null;
    new ResizeObserver(() => {
      const next = {w: plane.clientWidth, h: plane.clientHeight};
      if (!next.w || !next.h) return;
      if (size && fitted) {
        camera.x += (next.w - size.w) / 2;
        camera.y += (next.h - size.h) / 2;
      }
      size = next;
      if (!stageEl.hidden && !fitted) fit(false);
      requestDraw();
    }).observe(plane);
  }

  function init() {
    if (ready) return;
    ready = true;
    Object.values(blocks).forEach(block => objects.set(block.id, buildBlock(block)));
    Object.values(groups).forEach(group => objects.set(group.id, buildGroup(group)));
    // Outlines sit behind the links and blocks and never take clicks.
    Object.values(groups).forEach(group => {
      const outline = node('div', undefined, 'frame2d ' + group.lane);
      outline.style.borderColor = S.palette(group.lane).edge + '73';
      outline.dataset.frame = group.id;
      outline.hidden = true;
      worldEl.insertBefore(outline, linkLayer);
      frames.set(group.id, outline);
    });
    labels = S.buildLabels(objects, labelLayer);
    bindPlane();
    rebuildPanels();
    applyLayout(false);
  }

  function show() {
    stageEl.hidden = false;
    init();
    if (!fitted || needsFit) fit(false);
    needsFit = false;
    return true;
  }

  // ---- Test hooks

  function box(element) {
    const r = element.getBoundingClientRect();
    return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, center: {x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2}};
  }

  function snapshot() {
    return {
      objects: [...objects.values()].map(object => ({id: object.id, kind: object.kind, lane: object.lane, shown: !object.element.hidden,
        rect: box(object.element), center: box(object.element).center})),
      panels: [...objects.values()].filter(object => !object.element.hidden).flatMap(object => object.panels.map(panel => ({
        id: object.id + ':' + panel.key, block: object.id, key: panel.key, rect: box(panel.canvas), center: box(panel.canvas).center}))),
      labels: labels.filter(label => !label.element.hidden).map(label => {
        const r = label.element.getBoundingClientRect();
        return {text: label.element.textContent, left: r.left, right: r.right, top: r.top, bottom: r.bottom};
      }),
      frames: [...frames].filter(([, outline]) => !outline.hidden).map(([group, outline]) => ({group, rect: box(outline)})),
      zoom: camera.s,
      settled: tweens.length === 0,
    };
  }

  // Centre of the element for ``id`` (or its patch tile / panel), when it is inside the plane.
  function pointFor(id, target) {
    const object = objects.get(id);
    if (!object || object.element.hidden) return null;
    const element = typeof target === 'string' ? (object.panels.find(panel => panel.key === target) || {}).canvas
      : target === undefined ? object.element : object.element.querySelector('[data-patch="' + target + '"]');
    if (!element) return null;
    const r = element.getBoundingClientRect(), outer = plane.getBoundingClientRect();
    const x = (r.left + r.right) / 2, y = (r.top + r.bottom) / 2;
    return x > outer.left && x < outer.right && y > outer.top && y < outer.bottom ? {x, y} : null;
  }

  S.register('2d', {
    objects, show, relayout, rebuildPanels, snapshot, pointFor, focus,
    requestRender: requestDraw,
    started: () => ready,
    hide: () => { stageEl.hidden = true; },
    repaint: () => { if (ready) paint(); },
    fitAll: () => fit(true),
  });
})();
