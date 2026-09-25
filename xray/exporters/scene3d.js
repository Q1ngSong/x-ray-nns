// 3D view: draws the shared scene layout (XRAY_SCENE) with the vendored Three.js bundle
// (XRAY_THREE) as slabs, plate stacks and panel planes, under an orbiting camera. State, layout,
// panels, labels and actions live in scene_core.js; this file owns meshes, camera and picking.
(function () {
  'use strict';
  const THREE = window.XRAY_THREE, S = window.XRAY_SCENE;
  const {blocks, groups, lanes, view} = S;
  const DEFAULT_DIRECTION = [-0.5, 0.42, 1], FOCUS_DISTANCE = 14;
  const stageEl = E('scene-stage'), holder = E('scene-canvas'), labelLayer = E('scene-labels'), note = E('scene-note');
  const objects = new Map();
  let labels = [], renderer = null, camera = null, controls = null, world = null, links = null, linkPairs = [];
  let frame = 0, tweens = [], imageTexture = null, fitted = false, needsFit = false;

  function hasWebGL() {
    try {
      const probe = document.createElement('canvas');
      return Boolean(probe.getContext('webgl2') || probe.getContext('webgl'));
    } catch (error) {
      return false;
    }
  }

  function pickable(mesh, pick) {
    mesh.userData.pick = pick;
    return mesh;
  }

  function slabParts(size, lane, pick) {
    const geometry = new THREE.BoxGeometry(size.w, size.h, size.d);
    const mesh = pickable(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: S.palette(lane).face, roughness: 0.85, metalness: 0})), pick);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({color: S.palette(lane).edge, transparent: true, opacity: 0.6}));
    return {meshes: [mesh], edges: [edges]};
  }

  // A plane facing +z; ``uv`` = [u0, v0, u1, v1] selects part of the texture.
  function plane(width, height, texture, uv, pick) {
    const geometry = new THREE.PlaneGeometry(width, height);
    if (uv) {
      const attribute = geometry.attributes.uv;
      attribute.setXY(0, uv[0], uv[3]); attribute.setXY(1, uv[2], uv[3]);
      attribute.setXY(2, uv[0], uv[1]); attribute.setXY(3, uv[2], uv[1]);
    }
    const material = new THREE.MeshBasicMaterial({map: texture || null, color: texture ? '#ffffff' : '#eef3f7', side: THREE.DoubleSide});
    return pickable(new THREE.Mesh(geometry, material), pick);
  }

  function frameLines(width, height, lane) {
    return new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(width, height)),
      new THREE.LineBasicMaterial({color: S.palette(lane).edge, transparent: true, opacity: 0.6}));
  }

  function canvasTexture(canvas) {
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  // The texture wraps the shared image canvas, which may still be loading.
  function sourceTexture() {
    if (!imageTexture && S.model.image) {
      imageTexture = canvasTexture(S.sourceImage());
      S.sourceImage(() => {
        imageTexture.needsUpdate = true;
        requestRender();
      });
    }
    return imageTexture;
  }

  function buildBlock(block) {
    const kind = S.inputKind(block), node3 = new THREE.Group(), pick = {kind: 'block', id: block.id};
    let parts, size;
    if (kind === 'image') {
      size = {w: 2.6, h: 2.6, d: 0.04};
      parts = {meshes: [plane(size.w, size.h, sourceTexture(), null, pick)], edges: [frameLines(size.w, size.h, block.lane)]};
    } else if (kind === 'patches') {
      const {rows, cols, tile, size: gridSize} = S.patchGrid();
      size = gridSize;
      parts = {meshes: [], edges: []};
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const mesh = plane(tile, tile, sourceTexture(), [c / cols, 1 - (r + 1) / rows, (c + 1) / cols, 1 - r / rows],
            {kind: 'block', id: block.id, patch: r * cols + c});
          mesh.position.set((c - (cols - 1) / 2) * (tile + S.TILE_GAP), ((rows - 1) / 2 - r) * (tile + S.TILE_GAP), 0);
          parts.meshes.push(mesh);
        }
      }
    } else if (kind === 'tokens') {
      const tokens = S.tokenCanvas(block);
      size = {w: tokens.width, h: tokens.height, d: 0.04};
      parts = {meshes: [plane(size.w, size.h, canvasTexture(tokens.canvas), null, pick)], edges: [frameLines(size.w, size.h, block.lane)]};
    } else {
      size = S.slabSize(block);
      parts = slabParts(size, block.lane, pick);
    }
    [...parts.meshes, ...parts.edges].forEach(part => node3.add(part));
    return {id: block.id, kind: 'block', lane: block.lane, node: node3, current: node3.position, visible: () => node3.visible,
      size, target: new THREE.Vector3(), shown: true, panels: [], stack: {width: 0, height: 0, direction: S.panelDirection(block.lane)}, ...parts};
  }

  // A collapsed group is a tight stack of thin plates, one per member layer.
  function buildGroup(group) {
    const member = objects.get(group.blocks[0]).size, size = S.groupSize(group, member), node3 = new THREE.Group();
    const parts = {meshes: [], edges: []};
    group.blocks.forEach((_, index) => {
      const plate = slabParts({w: S.PLATE, h: member.h, d: member.d}, group.lane, {kind: 'group', id: group.id});
      [...plate.meshes, ...plate.edges].forEach(part => {
        part.position.x = -size.w / 2 + S.PLATE / 2 + index * (S.PLATE + S.PLATE_GAP);
        node3.add(part);
      });
      parts.meshes.push(...plate.meshes);
      parts.edges.push(...plate.edges);
    });
    return {id: group.id, kind: 'group', lane: group.lane, node: node3, current: node3.position, visible: () => node3.visible,
      size, target: new THREE.Vector3(), shown: true, panels: [], stack: {width: 0, height: 0, direction: 1}, ...parts};
  }

  // Panel planes wear the canvases drawn by the core; redraws flag the texture for upload.
  function rebuildPanels(id) {
    (id === undefined ? [...objects.values()] : [objects.get(id)]).forEach(object => {
      object.panels.forEach(panel => {
        object.node.remove(panel.mesh);
        panel.mesh.geometry.dispose();
        panel.mesh.material.dispose();
        panel.texture.dispose();
      });
      object.panels = S.makePanels(object);
      object.panels.forEach(panel => {
        panel.texture = canvasTexture(panel.canvas);
        panel.onDraw = () => { panel.texture.needsUpdate = true; };
        panel.mesh = plane(panel.width, panel.height, panel.texture, null, {kind: 'panel', id: object.id, key: panel.key, tensor: panel.spec.tensor});
        panel.mesh.material.transparent = true;
        object.node.add(panel.mesh);
      });
      S.arrangePanels(object);
      object.panels.forEach(panel => panel.mesh.position.set(panel.x, panel.y, 0));
    });
  }

  // ---- Layout: the core places objects; this file animates them and draws the links.

  function rebuildLinks() {
    linkPairs = S.linkPairs(objects);
    if (links) {
      world.remove(links);
      links.geometry.dispose();
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(linkPairs.length * 6), 3));
    links = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({color: '#a9bccb', transparent: true, opacity: 0.8}));
    world.add(links);
    updateLinks();
  }

  function updateLinks() {
    if (!links) return;
    const array = links.geometry.attributes.position.array;
    linkPairs.forEach(([from, to], index) => {
      const a = from.node.position, b = to.node.position;
      array.set([a.x + from.size.w / 2, a.y, 0, b.x - to.size.w / 2, b.y, 0], index * 6);
    });
    links.geometry.attributes.position.needsUpdate = true;
    links.geometry.computeBoundingSphere();
  }

  // Groups appear at the end of a move and vanish at its start; members do the reverse,
  // so expanding layers fly out of the plate stack and collapsing ones fly back in.
  function applyLayout(animate) {
    S.layout(objects);
    const moves = [];
    objects.forEach(object => {
      const appearing = object.shown && !object.node.visible;
      if (appearing && object.kind === 'block' && blocks[object.id].group) {
        object.node.position.copy(objects.get(blocks[object.id].group).node.position);
        object.node.visible = true;
      }
      if (!object.shown && object.kind === 'group') object.node.visible = false;
      moves.push({object, from: object.node.position.clone(), to: object.target.clone()});
    });
    rebuildLinks();
    S.refreshGroupLabels(objects);
    const finish = () => moves.forEach(({object}) => {
      object.node.position.copy(object.target);
      object.node.visible = object.shown;
    });
    if (!animate || !fitted) {
      finish();
      updateLinks();
      requestRender();
      return;
    }
    tween(380, eased => {
      moves.forEach(move => move.object.node.position.lerpVectors(move.from, move.to, eased));
      updateLinks();
    }, () => { finish(); updateLinks(); });
  }

  function tween(duration, step, done) {
    tweens.push({start: performance.now(), duration, step, done});
    requestRender();
  }

  // Frame everything shown along the current viewing direction (or the default one): centre the
  // content across the view, then back off until each corner fits the frustum. Clicks never call
  // this; they keep the zoom (see ``hold``).
  function fit(animate, resetDirection) {
    const corners = S.framed(objects).flatMap(S.cornersOf).map(corner => new THREE.Vector3(corner.x, corner.y, corner.z));
    if (!corners.length) return;
    const back = camera.position.clone().sub(controls.target);
    if (resetDirection || !fitted || back.lengthSq() < 1e-6) back.set(...DEFAULT_DIRECTION);
    back.normalize();
    const right = new THREE.Vector3(0, 1, 0).cross(back).normalize(), up = back.clone().cross(right).normalize();
    const origin = corners.reduce((sum, corner) => sum.add(corner), new THREE.Vector3()).divideScalar(corners.length);
    const local = corners.map(corner => corner.clone().sub(origin)).map(v => [v.dot(right), v.dot(up), v.dot(back)]);
    const middle = axis => (Math.min(...local.map(v => v[axis])) + Math.max(...local.map(v => v[axis]))) / 2;
    const [mx, my] = [middle(0), middle(1)];
    const center = origin.clone().add(right.clone().multiplyScalar(mx)).add(up.clone().multiplyScalar(my));
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * 0.92, tanH = tanV * camera.aspect;
    const distance = Math.max(...local.map(([x, y, z]) => z + Math.max(Math.abs(x - mx) / tanH, Math.abs(y - my) / tanV)));
    const position = center.clone().add(back.multiplyScalar(distance));
    if (!animate || !fitted) {
      controls.target.copy(center);
      camera.position.copy(position);
      fitted = true;
      controls.update();
      requestRender();
      return;
    }
    const fromTarget = controls.target.clone(), fromPosition = camera.position.clone();
    tween(450, eased => {
      controls.target.lerpVectors(fromTarget, center, eased);
      camera.position.lerpVectors(fromPosition, position, eased);
    });
  }

  // While another view is showing, layout changes land without animation and refit on show.
  // ``everything`` (Collapse all, a Branch change) refits the whole scene; otherwise the camera keeps
  // its distance and direction, and only follows ``anchor`` (the clicked block).
  function relayout(everything, anchor) {
    if (stageEl.hidden) {
      applyLayout(false);
      needsFit = true;
      return;
    }
    const object = anchor && objects.get(anchor), before = object && object.node.visible ? object.node.position.clone() : null;
    applyLayout(true);
    if (everything) fit(true);
    else if (before) hold(object, before);
  }

  // Keep the zoom: move the camera and its target with ``object`` so it stays where it was drawn,
  // then pan across the view only as far as needed to bring its panels inside the fitted frustum.
  function hold(object, before) {
    if (!object.shown) return;
    const shift = object.target.clone().sub(before);
    if (object.panels.length) {
      const back = camera.position.clone().sub(controls.target).normalize();
      const right = new THREE.Vector3(0, 1, 0).cross(back).normalize(), up = back.clone().cross(right).normalize();
      const eye = camera.position.clone().add(shift), depth = Math.max(0.1, -object.target.clone().sub(eye).dot(back));
      const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * 0.92, tanH = tanV * camera.aspect;
      // Each corner's place on screen, where ±1 is the edge of the fitted frustum.
      const points = S.cornersOf(object).map(corner => new THREE.Vector3(corner.x, corner.y, corner.z).sub(eye))
        .map(v => ({x: v.dot(right) / Math.max(0.1, -v.dot(back)) / tanH, y: v.dot(up) / Math.max(0.1, -v.dot(back)) / tanV}));
      // Moving the camera one way slides the view the other way, by about depth · tan per unit.
      shift.addScaledVector(right, -S.nudge(points.map(point => point.x), -1, 1, false) * tanH * depth);
      shift.addScaledVector(up, -S.nudge(points.map(point => point.y), -1, 1, object.stack.direction < 0) * tanV * depth);
    }
    const fromTarget = controls.target.clone(), fromPosition = camera.position.clone();
    const toTarget = fromTarget.clone().add(shift), toPosition = fromPosition.clone().add(shift);
    tween(450, eased => {
      controls.target.lerpVectors(fromTarget, toTarget, eased);
      camera.position.lerpVectors(fromPosition, toPosition, eased);
    });
  }

  // Aim at ``id`` for a list outside the graph (Full data path, Data flow), keeping the viewing
  // direction; the camera only comes closer, to FOCUS_DISTANCE, when it is farther than that.
  function focus(id) {
    const object = objects.get(id);
    if (!renderer || stageEl.hidden || !object || !object.shown) return;
    const back = camera.position.clone().sub(controls.target), distance = Math.min(back.length(), FOCUS_DISTANCE);
    const fromTarget = controls.target.clone(), fromPosition = camera.position.clone();
    const toTarget = object.target.clone(), toPosition = toTarget.clone().add(back.normalize().multiplyScalar(distance));
    tween(450, eased => {
      controls.target.lerpVectors(fromTarget, toTarget, eased);
      camera.position.lerpVectors(fromPosition, toPosition, eased);
    });
  }

  function paint() {
    objects.forEach(object => {
      const active = object.id === view.active, hover = object.id === view.hover;
      object.meshes.forEach(mesh => {
        const patch = mesh.userData.pick && mesh.userData.pick.patch;
        if (mesh.material.emissive) {
          mesh.material.emissive.set(active ? S.palette(object.lane).edge : '#000000');
          mesh.material.emissiveIntensity = active ? 0.32 : 0;
          mesh.material.color.set(hover && !active ? '#eef5fa' : S.palette(object.lane).face);
        }
        if (patch !== undefined) mesh.position.z = state.patchIndex === patch ? 0.18 : 0;
      });
      object.edges.forEach(edge => { edge.material.opacity = active ? 1 : hover ? 0.85 : 0.6; });
    });
    requestRender();
  }

  // ---- Rendering and input

  function project(point, rect) {
    const p = new THREE.Vector3(point.x, point.y, point.z || 0).project(camera);
    if (p.z > 1 || Math.abs(p.x) > 1.05 || Math.abs(p.y) > 1.05) return null;
    return {x: (p.x + 1) / 2 * rect.width, y: (1 - p.y) / 2 * rect.height};
  }

  function requestRender() {
    if (renderer && !frame) frame = requestAnimationFrame(loop);
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
    const moving = controls.update();
    renderer.render(world, camera);
    S.placeLabels(labels, holder.getBoundingClientRect(), project);
    if (tweens.length || moving) requestRender();
  }

  function pointerHit(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    const caster = new THREE.Raycaster();
    caster.setFromCamera(pointer, camera);
    const meshes = [];
    objects.forEach(object => { if (object.node.visible) meshes.push(...object.meshes, ...object.panels.map(panel => panel.mesh)); });
    const hit = caster.intersectObjects(meshes, false)[0];
    return hit ? hit.object.userData.pick : null;
  }

  function bindPointer(canvas) {
    let down = null;
    canvas.addEventListener('pointerdown', event => { down = {x: event.clientX, y: event.clientY}; });
    canvas.addEventListener('pointerup', event => {
      const click = down && Math.hypot(event.clientX - down.x, event.clientY - down.y) < 5;
      down = null;
      const pick = click && pointerHit(event);
      if (pick) S.activate(pick);
    });
    canvas.addEventListener('pointermove', event => {
      if (down) return;
      const pick = pointerHit(event);
      canvas.style.cursor = pick ? 'pointer' : '';
      S.hover(pick && pick.id || null);
    });
    canvas.addEventListener('pointerleave', () => S.hover(null));
  }

  function resize() {
    if (!renderer) return;
    const width = Math.max(1, holder.clientWidth), height = Math.max(1, holder.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    requestRender();
  }

  function init() {
    if (renderer) return true;
    try {
      renderer = new THREE.WebGLRenderer({antialias: true});
    } catch (error) {
      E('view-3d').disabled = true;
      note.textContent = 'WebGL could not start in this browser; the 2D view is unaffected.';
      return false;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor('#fbfcfe');
    holder.insertBefore(renderer.domElement, labelLayer);
    camera = new THREE.PerspectiveCamera(26, 1, 0.1, 500);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    // Azimuth stays within ±90° of the front view so the left-to-right flow is never seen from behind.
    // Panning is OrbitControls' default: right-drag, or Ctrl/⌘/Shift + left-drag.
    Object.assign(controls, {enableDamping: true, dampingFactor: 0.12, screenSpacePanning: true, minDistance: 3, maxDistance: 220,
      minPolarAngle: 0.35, maxPolarAngle: 1.5, minAzimuthAngle: -Math.PI / 2, maxAzimuthAngle: Math.PI / 2});
    controls.addEventListener('change', requestRender);
    world = new THREE.Scene();
    world.add(new THREE.HemisphereLight('#ffffff', '#dfe6ee', 2.2));
    const sun = new THREE.DirectionalLight('#ffffff', 1.6);
    sun.position.set(-4, 9, 7);
    world.add(sun);
    Object.values(blocks).forEach(block => objects.set(block.id, buildBlock(block)));
    Object.values(groups).forEach(group => objects.set(group.id, buildGroup(group)));
    objects.forEach(object => world.add(object.node));
    labels = S.buildLabels(objects, labelLayer);
    bindPointer(renderer.domElement);
    new ResizeObserver(resize).observe(holder);
    rebuildPanels();
    applyLayout(false);
    return true;
  }

  // The stage is shown before the renderer starts so the first fit uses the real canvas size.
  function show() {
    stageEl.hidden = false;
    if (!init()) {
      stageEl.hidden = true;
      return false;
    }
    resize();
    if (!fitted || needsFit) fit(false);
    needsFit = false;
    return true;
  }

  // ---- Test hooks

  function snapshot() {
    const canvasRect = renderer ? renderer.domElement.getBoundingClientRect() : null;
    const toScreen = point => {
      const p = point.clone().project(camera);
      return {x: canvasRect.left + (p.x + 1) / 2 * canvasRect.width, y: canvasRect.top + (1 - p.y) / 2 * canvasRect.height};
    };
    const all = [...objects.values()];
    return {
      objects: all.map(object => {
        const p = object.node.position, s = object.size;
        return {id: object.id, kind: object.kind, lane: object.lane, shown: object.node.visible,
          world: {min: [p.x - s.w / 2, p.y - s.h / 2, -s.d / 2], max: [p.x + s.w / 2, p.y + s.h / 2, s.d / 2]},
          center: renderer ? toScreen(p) : null};
      }),
      panels: all.filter(object => object.node.visible).flatMap(object => object.panels.map(panel => {
        const p = object.node.position, x = p.x + panel.x, y = p.y + panel.y;
        return {id: object.id + ':' + panel.key, block: object.id, key: panel.key,
          world: {min: [x - panel.width / 2, y - panel.height / 2, -0.01], max: [x + panel.width / 2, y + panel.height / 2, 0.01]},
          center: renderer ? toScreen(new THREE.Vector3(x, y, 0)) : null};
      })),
      labels: labels.filter(label => !label.element.hidden).map(label => {
        const r = label.element.getBoundingClientRect();
        return {text: label.element.textContent, left: r.left, right: r.right, top: r.top, bottom: r.bottom};
      }),
      distance: renderer ? camera.position.distanceTo(controls.target) : null,
      settled: tweens.length === 0,
    };
  }

  // Screen point whose first ray hit is ``id`` (or its ``target`` patch index / panel key).
  function pointFor(id, target) {
    const object = objects.get(id);
    if (!object || !object.node.visible) return null;
    const panel = typeof target === 'string' ? object.panels.find(item => item.key === target) : null;
    const rect = renderer.domElement.getBoundingClientRect(), box = new THREE.Box3();
    (panel ? [panel.mesh] : object.meshes).forEach(mesh => box.expandByObject(mesh));
    for (let i = 1; i < 12; i++) {
      for (let j = 1; j < 12; j++) {
        const point = new THREE.Vector3(box.min.x + (box.max.x - box.min.x) * i / 12, box.min.y + (box.max.y - box.min.y) * j / 12, box.max.z).project(camera);
        const x = rect.left + (point.x + 1) / 2 * rect.width, y = rect.top + (1 - point.y) / 2 * rect.height;
        if (Math.abs(point.x) > 0.98 || Math.abs(point.y) > 0.98) continue;
        const pick = pointerHit({clientX: x, clientY: y});
        const matches = pick && pick.id === id && (panel ? pick.kind === 'panel' && pick.key === target
          : pick.kind !== 'panel' && (target === undefined || pick.patch === target));
        if (matches) return {x, y};
      }
    }
    return null;
  }

  // Fraction of the frame that differs from the background; tests use it to prove WebGL drew.
  function painted() {
    if (!renderer) return 0;
    renderer.render(world, camera);
    const gl = renderer.getContext(), width = gl.drawingBufferWidth, height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let count = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (Math.abs(pixels[i] - 251) + Math.abs(pixels[i + 1] - 252) + Math.abs(pixels[i + 2] - 254) > 12) count++;
    }
    return count / (width * height);
  }

  function prepare() {
    if (!lanes.length) {
      note.textContent = 'This trace has no recorded module hierarchy to draw in 3D.';
    } else if (!hasWebGL()) {
      E('view-3d').disabled = true;
      E('view-3d').title = 'WebGL is not available in this browser';
    } else {
      note.textContent = 'Drag to orbit · Ctrl + drag (or right-drag) to pan · scroll to zoom · click a group label to expand it · click a block to open its values; the switches choose which values appear.';
    }
  }

  S.register('3d', {
    objects, prepare, show, relayout, rebuildPanels, snapshot, pointFor, painted, requestRender, focus,
    started: () => Boolean(renderer),
    available: () => lanes.length > 0 && !E('view-3d').disabled,
    hide: () => { stageEl.hidden = true; },
    repaint: () => { if (renderer) paint(); },
    fitAll: () => fit(true, true),
  });
})();
