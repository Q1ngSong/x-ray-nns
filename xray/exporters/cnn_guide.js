// A reading view over the same recorded CNN operations: overview, residual paths and real values.
(function () {
  'use strict';
  const S = window.XRAY_SCENE, backend = metadata.backend;
  if (!['torchvision.resnet18', 'torchvision.alexnet'].includes(backend)) return;
  const resnet = backend.endsWith('resnet18'), blocks = S.blocks, groups = S.groups;
  const title = resnet ? 'ResNet-18' : 'AlexNet';
  const root = node('section', undefined, 'cnn-guide'); root.id = 'cnn-guide'; root.hidden = true;
  root.setAttribute('aria-label', title + ' 结构导览');
  E('graph-stage').before(root);
  const switcher = node('button', '结构导览'); switcher.id = 'view-guide'; switcher.type = 'button';
  switcher.setAttribute('aria-pressed', 'false'); E('view-2d').before(switcher);
  switcher.onclick = () => window.xrayScene.setMode('guide');
  let selected = resnet ? 'layer1.0' : 'features.stage1', step = 'all';
  const ids = Object.keys(blocks), opOf = id => operations.get(blocks[id].operations[0]);
  const shape = id => (tensors.get(id)?.shape || []).slice(1).join(' × ');
  const outputShape = id => shape(blocks[id].tensor);
  const stages = resnet ? [
    {name:'Stem', note:'初步提取特征', ids:ids.filter(id => ['conv1','bn1','relu','maxpool'].includes(id)), key:'stem'},
    ...[1,2,3,4].map(i => ({name:'Stage ' + i, note: i === 1 ? '保持分辨率' : '首块下采样', groups:Object.keys(groups).filter(id => id.startsWith('layer'+i+'.')), key:'stage'+i})),
    {name:'分类头',note:'空间 → 类别',ids:ids.filter(id => ['avgpool','fx_flatten','fc'].includes(id) || blocks[id].kind === 'result'),key:'head'}
  ] : [
    ...Object.values(groups).map((g,i) => ({name:g.id === 'classifier' ? '分类头' : '卷积阶段 '+(i+1),note:g.id === 'classifier' ? '4096 → 4096 → 1000' : '卷积 / 激活 / 池化',groups:[g.id],key:g.id})),
    {name:'最终预测',note:'ImageNet · 1000 类',ids:ids.filter(id => blocks[id].kind === 'result'),key:'head'}
  ];
  const stageMembers = stage => stage.ids || stage.groups.flatMap(id => groups[id].blocks);
  const members = key => key === 'classifier' ? [...ids.filter(id=>['avgpool','fx_flatten'].includes(id)),...groups[key].blocks] : groups[key]?.blocks || stageMembers(stages.find(stage => stage.key === key));
  const isProjection = key => members(key).some(id => blocks[id].stage.includes('.downsample.'));
  const section = node('div', undefined, 'guide-detail');
  root.style.setProperty('--stage-count', String(stages.length));
  const bar = node('div', undefined, 'guide-stages'); bar.setAttribute('aria-label','按计算顺序排列的网络阶段');
  const intro = node('div',undefined,'guide-intro');
  intro.append(node('div','01 / NETWORK AT A GLANCE','guide-kicker'),node('h2',title+'，从整体到一个块'));
  intro.append(node('p',resnet ? '先看 4 个 Stage，再点击其中一个残差块。蓝色是主分支，橙色是捷径，两路在 + 处逐元素相加。' : '卷积提取局部特征，池化缩小空间尺寸，分类头把特征转成类别分数。点击阶段查看内部模块。'));
  const input = node('div', '输入 RGB  ·  3 × 224 × 224    →    计算按编号从左到右进行；尺寸记为 C × H × W。', 'guide-input-note');
  root.append(intro,input,bar,section);
  const legend = node('div',undefined,'guide-legend');
  [['Conv2d','卷积'],['BatchNorm2d','归一化'],['ReLU','激活'],['MaxPool2d','池化'],['_FunctionCall','相加'],['Linear','全连接']].filter(([type])=>type === '_FunctionCall' ? resnet : [...operations.values()].some(op=>op.type===type)).forEach(([type,label])=>{
    const item=node('span',label); item.style.setProperty('--role',S.moduleStyle(type).edge); legend.appendChild(item);
  });
  root.appendChild(legend);
  const reading=node('details',undefined,'guide-reading');
  reading.appendChild(node('summary',resnet ? '为什么叫残差？18 层又是怎么数的？' : '怎样把特征图与真实数据对应起来？'));
  reading.appendChild(node('p',resnet ? '普通堆叠直接学习变换 H(x)。残差块把输出写成 F(x) + x，让主分支学习相对输入的修正；相加之后再经过 ReLU。这种连接为信息和梯度提供较直接的通路，但不保证所有层都不会丢失信息。当前页面记录的是前向推理，没有展示训练梯度。' : '每个模块下面的尺寸来自这次真实运行。卷积与池化输出保留空间位置；切换到 2D / 3D，可查看 PCA 摘要与单个特征通道。通道预览平均池化至多 28 × 28，颜色按各通道自身范围缩放。'));
  if (resnet) reading.appendChild(node('p','ResNet-18 的常见层数计法：Stem 的 1 个卷积 + 8 个 BasicBlock × 2 个主分支卷积 + 1 个全连接 = 18。BN、ReLU、池化和捷径投影不计入这个名称。这里是 BasicBlock；ResNet-50 使用 1×1 → 3×3 → 1×1 的 Bottleneck，不能直接把它当成同一种块。'));
  const source=node('a','架构来源：torchvision 官方实现 ↗');
  source.href='https://docs.pytorch.org/vision/stable/_modules/torchvision/models/'+(resnet?'resnet':'alexnet')+'.html';source.target='_blank';source.rel='noopener';reading.appendChild(source);
  root.appendChild(reading);

  function inspectOperation(id) { if(E('toggle-inspector').getAttribute('aria-pressed')!=='true') E('toggle-inspector').click(); selectOperation(id); }
  function inspectTensor(id) { if(E('toggle-inspector').getAttribute('aria-pressed')!=='true') E('toggle-inspector').click(); selectTensor(id); }
  function choose(key) { selected=key; step='all'; renderStages(); renderDetail(); section.scrollIntoView({block:'start',behavior:'smooth'}); }
  function renderStages() {
    bar.replaceChildren();
    stages.forEach((stage,index)=>{
      const card=node('div',undefined,'guide-stage'); card.style.setProperty('--stage', ['#64889c','#447eae','#458d83','#8879a8','#b08653','#9377a3'][index%6]);
      card.append(node('div',String(index+1).padStart(2,'0')+' / '+stage.name,'guide-stage-name'),node('p',stage.note));
      (stage.groups || [stage.key]).forEach(key=>{
        const button=node('button',undefined,'guide-module');button.type='button'; button.dataset.guideGroup=key;
        button.setAttribute('aria-label','查看 '+key);button.setAttribute('aria-pressed',String(selected===key));
        button.append(node('strong',groups[key]?.label || stage.name));
        const residual=resnet && key.startsWith('layer');
        button.append(node('span',residual ? (isProjection(key)?'F(x) + P(x)':'F(x) + x') : '展开内部模块','guide-module-formula'));
        button.append(node('small',outputShape(members(key).at(-1))));button.onclick=()=>choose(key);card.appendChild(button);
      });
      bar.appendChild(card);
    });
  }
  function svgNode(tag,attrs,text) {
    const element=document.createElementNS('http://www.w3.org/2000/svg',tag);
    Object.entries(attrs||{}).forEach(([key,value])=>element.setAttribute(key,String(value)));
    if(text!==undefined)element.textContent=text;
    return element;
  }
  function renderDetail() {
    section.replaceChildren();
    const residual=resnet && selected.startsWith('layer'), list=members(selected), projection=residual&&isProjection(selected);
    section.append(node('div','02 / '+(residual?'BASIC BLOCK':'INSIDE THE STAGE'),'guide-kicker'));
    const head=node('div',undefined,'guide-detail-head');
    head.append(node('h3',selected+' · '+(residual?(projection?'投影与下采样':'恒等捷径'): '内部模块')));
    const explore=node('button','在 3D 中定位 ↗');explore.type='button';explore.onclick=()=>{
      window.xrayScene.setMode('3d'); const id=residual?list.find(id=>blocks[id].semantic==='residual_add'):list[0];
      inspectOperation(blocks[id].operations[0]);window.xrayScene.focus(id);
    };const back=node('button','网络总览 ↑');back.type='button';back.onclick=()=>root.scrollTo({top:0,behavior:'smooth'});const actions=node('div',undefined,'guide-detail-actions');actions.append(back,explore);head.appendChild(actions);section.appendChild(head);
    if(residual) {
      const add=list.find(id=>blocks[id].semantic==='residual_add'), addOp=opOf(add), first=opOf(list[0]);
      section.append(node('p',projection?'输入与主分支的形状不同，捷径先经过 1×1 卷积和 BN，把尺寸对齐后才能相加。':'输入与主分支输出的形状相同，捷径直接传递 x，不额外做卷积。','guide-description'));
      const chips=node('div',undefined,'guide-steps');
      [['all','完整路径'],['input','① 输入 x'],['main','② 主分支 F(x)'],['shortcut','③ 捷径'],['add','④ 相加与激活']].forEach(([key,label])=>{
        const button=node('button',label);button.type='button';button.setAttribute('aria-pressed',String(step===key));button.onclick=()=>{step=key;renderDetail();};chips.appendChild(button);
      });section.appendChild(chips);
      const wrap=node('div',undefined,'guide-diagram');wrap.dataset.step=step;
      const svg=svgNode('svg',{viewBox:'0 0 1000 340',role:'group','aria-label':selected+' 残差计算图'});
      const defs=svgNode('defs');
      [['main','#5c89a8'],['skip','#c18442']].forEach(([id,color])=>{
        const marker=svgNode('marker',{id:'guide-arrow-'+id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});marker.appendChild(svgNode('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.appendChild(marker);
      });svg.appendChild(defs);
      const path=(d,kind)=>{const line=svgNode('path',{d,fill:'none',stroke:kind==='skip'?'#c18442':'#5c89a8','stroke-width':3,'marker-end':'url(#guide-arrow-'+kind+')',class:kind==='skip'?'guide-skip-path':'guide-main-path'});svg.appendChild(line);};
      path('M 120 142 H 154','main');path('M 266 142 H 288','main');path('M 362 142 H 382','main');path('M 454 142 H 474','main');path('M 586 142 H 608','main');path('M 682 142 H 722','main');path('M 778 142 H 820','main');
      path('M 135 142 V 282 H '+(projection?'340':'750'), 'skip');
      if(projection){path('M 474 282 H 510','skip');path('M 596 282 H 750 V 174','skip');}
      else path('M 750 282 V 174','skip');
      svg.append(svgNode('text',{x:160,y:64,class:'guide-path-title'},'主分支 F(x)'),svgNode('text',{x:160,y:242,class:'guide-path-title skip'},projection?'捷径 P(x) · 投影匹配形状':'捷径 x · 原样传递'));
      const tile=(id,x,y,w,label,subtitle,part,tensorId)=>{
        const block=id&&blocks[id], operation=id&&opOf(id), type=operation?.type||'input', color=S.moduleStyle(type);
        const g=svgNode('g',{class:'guide-op '+part,role:'button',tabindex:0,'aria-label':(id?blocks[id].stage:'输入 x')+' · '+label,'data-guide-op':id||'input'});
        g.append(svgNode('rect',{x,y,width:w,height:76,rx:type==='ReLU'?24:10,fill:color.face,stroke:color.edge,'stroke-width':1.5}));
        g.append(svgNode('text',{x:x+w/2,y:y+25,'text-anchor':'middle',class:'guide-op-title'},label));
        if(subtitle)g.appendChild(svgNode('text',{x:x+w/2,y:y+46,'text-anchor':'middle',class:'guide-op-sub'},subtitle));
        g.appendChild(svgNode('text',{x:x+w/2,y:y+64,'text-anchor':'middle',class:'guide-op-shape'},(tensorId?shape(tensorId):outputShape(id)).replaceAll(' ','')));
        const pick=()=>tensorId?inspectTensor(tensorId):inspectOperation(block.operations[0]);g.onclick=pick;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}};svg.appendChild(g);
      };
      tile(null,20,104,100,'x','块输入','input',first.inputs[0]);
      const main=list.filter(id=>!blocks[id].stage.includes('.downsample.')&&id!==add).slice(0,5);
      const positions=[[160,106],[292,70],[386,68],[480,106],[612,70]];
      main.forEach((id,i)=>{
        const type=opOf(id).type, stride=projection&&i===0?'s=2':'s=1';
        tile(id,positions[i][0],104,positions[i][1],type==='Conv2d'?'Conv 3×3':type==='BatchNorm2d'?'BN':'ReLU',type==='Conv2d'?stride:'', 'main');
      });
      if(projection){const shortcuts=list.filter(id=>blocks[id].stage.includes('.downsample.'));tile(shortcuts[0],340,244,134,'Conv 1×1','s=2','shortcut');tile(shortcuts[1],510,244,86,'BN','','shortcut');}
      else svg.appendChild(svgNode('text',{x:440,y:271,'text-anchor':'middle',class:'guide-identity'},'identity · '+shape(addOp.inputs[1])));
      const plus=svgNode('g',{class:'guide-op add',role:'button',tabindex:0,'aria-label':'Add 逐元素相加','data-guide-op':add});
      plus.append(svgNode('circle',{cx:750,cy:142,r:26,fill:'#fff2de',stroke:'#c18442','stroke-width':2}),svgNode('text',{x:750,y:151,'text-anchor':'middle',class:'guide-plus'},'+'));
      plus.onclick=()=>inspectOperation(addOp.id);plus.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();plus.onclick();}};svg.appendChild(plus);
      tile(list.at(-1),826,104,124,'ReLU','输出 y','add');
      wrap.appendChild(svg);section.appendChild(wrap);
      const descriptions={all:'同一个 x 分成两路，先各自计算，再在 + 处汇合。点击图中任一模块，可查看这次运行的真实数值。',input:'输入 x：'+shape(first.inputs[0])+'。同一份张量同时送入主分支与捷径。',main:'F(x) = BN₂(Conv₂(ReLU(BN₁(Conv₁(x)))))。第二个 BN 后先相加，再做 ReLU。',shortcut:projection?'P(x) = BN(Conv 1×1, stride 2)。空间减半、通道翻倍，输出为 '+shape(addOp.inputs[1])+'。':'捷径直接使用 x，输出仍是 '+shape(addOp.inputs[1])+'。这里没有新的可学习层。',add:'相加的两路都是 '+shape(addOp.inputs[0])+'，逐元素相加，不是拼接；然后 y = ReLU(F(x) + '+(projection?'P(x)':'x')+')。'};
      section.appendChild(node('p',descriptions[step],'guide-explanation'));
      const equality=node('div',undefined,'guide-equation');equality.append(node('span','F(x)  '+shape(addOp.inputs[0])),node('b','+'),node('span',(projection?'P(x)  ':'x  ')+shape(addOp.inputs[1])),node('b','→'),node('span','y  '+outputShape(list.at(-1))));section.appendChild(equality);
    } else {
      const chain=node('div',undefined,'guide-chain');
      list.forEach(id=>{const op=opOf(id), style=S.blockStyle(blocks[id]),button=node('button',undefined,'guide-chain-op');button.type='button';button.dataset.guideOp=id;button.style.setProperty('--role',style.edge);button.style.background=style.face;
        button.append(node('strong',op.type==='Dropout'?'Dropout (eval)':op.type==='checkpoint'?'输入':style.title),node('small',blocks[id].stage),node('span',outputShape(id)));button.onclick=()=>inspectOperation(op.id);chain.appendChild(button);});section.appendChild(chain);
      section.appendChild(node('p','每个卡片对应一次实际模块调用；点击后，右侧展示该模块的输入、输出和统计量。箭头表示计算先后。','guide-description'));
    }
    repaint();
  }
  function repaint() {
    root.querySelectorAll('[data-guide-op]').forEach(element=>element.classList.toggle('selected', blocks[element.dataset.guideOp]?.operations.includes(state.operation)));
  }
  renderStages();renderDetail();
  S.register('guide',{started:()=>true,show:()=>{root.hidden=false;return true;},hide:()=>{root.hidden=true;},repaint,relayout:()=>{},rebuildPanels:()=>{},fitAll:()=>{},focus:id=>{const group=blocks[id]?.group;if(group)choose(group);repaint();}});
})();
