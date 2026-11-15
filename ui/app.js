(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const ACTIVE = ['queued', 'planning', 'running', 'evaluating'];
  const state = { bridge: null, image: null, job: null, candidate: null, keepEmpty: false, maskStrokes: [], regionStrokes: [], strokeOrder: [], currentStroke: null,
    baseMask: null, maskDirty: false, tool: 'rect', brushSize: 28, zoom: 1, promptPoints: [], promptLabels: [], promptBox: null,
    proposal: null, references: [], selectedReference: null, registration: null, lastSnapshot: null, toastTimer: null,
    auxVersion: -1, damageItems: [], damageSummary: '', selectedDamage: null, activeJobId: null, running: false, autoBusy: false,
    damageDrawMode: false, damageDraw: null, damageDrawTarget: null, damageJobKey: null, maskEditTarget: null,
    rectDraw: null,
    aiAssessment: null, finalRepair: null,
    batchJobIds: [], defectSubmitting: false, triple: { largeJobId: null, fullJobId: null } };
  const canvas = $('#mask-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const regionCanvas = $('#region-canvas');
  const regionCtx = regionCanvas ? regionCanvas.getContext('2d', { willReadFrequently: true }) : null;
  // Opaque category labels are separate from the translucent display colours.
  // Red encodes unrestorable; blue encodes uncertain (including its alias).
  const regionClasses = document.createElement('canvas');
  const regionClassCtx = regionClasses.getContext('2d', { willReadFrequently: true });
  const image = $('#source-image');
  const imageWrap = $('#canvas-wrap');
  const REGION_COLORS = {
    uncertain: 'rgba(220,0,60,0.72)',
    uncertainty: 'rgba(220,0,60,0.72)', // alias
    unrestorable: 'rgba(180,0,40,0.85)',
  };

  const invoke = (method, ...args) => new Promise((resolve, reject) => {
    if (!state.bridge || !state.bridge[method]) return reject(new Error('桌面服务尚未连接。'));
    state.bridge[method](...args, result => {
      try {
        const parsed = typeof result === 'string' ? JSON.parse(result) : result;
        if (parsed?.error) reject(new Error(parsed.error)); else resolve(parsed);
      } catch (error) { reject(error); }
    });
  });
  const toast = (text) => {
    const node = $('#toast'); node.textContent = text; node.classList.remove('hidden');
    clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => node.classList.add('hidden'), 3200);
  };
  const setBusy = (value) => {
    state.running = value;
    document.body.classList.toggle('busy', value);
    $('[data-action="run-repair"]').disabled = value || !state.image || !maskHasPixels();
    $$('[data-action="open-image"], [data-action="load-demo"]').forEach(button => { button.disabled = value; });
    $$('.history-item').forEach(item => { item.disabled = value; });
  };
  const maskHasPixels = () => {
    if (!canvas.width || !canvas.height) return false;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
    return false;
  };
  const notifyError = (error) => toast(error?.message || String(error));
  const AUTO_HINT = '分析结果须人工核对。';
  const isProfessional = () => Boolean($('#professional-step')?.checked);
  const currentWorkflow = () => (isProfessional() ? 'professional_step' : 'standard');
  const isRegionTool = (tool) => tool === 'uncertain' || tool === 'uncertainty' || tool === 'unrestorable';
  const regionPixelStats = () => {
    const stats = { uncertain: false, unrestorable: false };
    if (!regionClasses.width || !regionClasses.height) return stats;
    const data = regionClassCtx.getImageData(0, 0, regionClasses.width, regionClasses.height).data;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3]; if (a < 40) continue;
      if (data[i] > data[i + 2]) stats.unrestorable = true;
      else stats.uncertain = true;
      if (stats.uncertain && stats.unrestorable) break;
    }
    return stats;
  };
  const hasUnrestorableRegions = () => regionPixelStats().unrestorable;
  const hasUncertainRegions = () => regionPixelStats().uncertain;
  const selectedDecision = () => {
    const node = document.querySelector('input[name="review-decision"]:checked');
    return node ? node.value : '';
  };

  const setAutoStatus = (text) => { $('#auto-status').textContent = text || AUTO_HINT; };
  const renderVisionProviderNote = (models = state.lastSnapshot?.models || {}) => {
    const note = $('#vision-provider-note');
    const setup = models.local_detection_setup || {};
    if ($('#vision-provider').value === 'local') {
      if (setup.qwen_ready === false) {
        note.textContent = `${setup.qwen_hint || '本地分析需要已安装 Qwen3-VL。'} 仍可手动圈选缺损区域。`;
      } else {
        note.textContent = '使用本机 Qwen3-VL 识别病害；'
          + (setup.sam_ready === false ? 'SAM 2.1 未安装，仅保留候选框，修复范围需手动修正。' : 'SAM 2.1 生成修复范围候选，需人工确认。');
      }
    } else {
      note.textContent = '点击分析会向 DeepSeek 发送当前图像的缩小副本。'
        + (models.deepseek_vision === false ? '请先在“API 设置”保存 DeepSeek 密钥。' : '');
    }
  };
  const syncRegionUi = () => {
    const on = isProfessional();
    const regions = regionPixelStats();
    const force = $('#force-override')?.closest('label');
    const showForce = on && regions.uncertain;
    if (force) force.style.display = showForce ? '' : 'none';
    const reason = $('#force-override-reason');
    if (reason) reason.classList.toggle('hidden', !(showForce && $('#force-override')?.checked));
    const runBtn = $('[data-action="run-repair"]');
    if (runBtn) runBtn.disabled = state.running || state.autoBusy || !state.image || !maskHasPixels() || (on && regions.unrestorable);
  };
  const syncProfessionalUi = () => {
    const on = isProfessional();
    const autoBtn = $('[data-action="auto-restore"]');
    if (autoBtn) autoBtn.textContent = 'AI 分析缺损与文化线索';
    const note = $('#professional-note');
    if (note) note.style.opacity = on ? '1' : '.72';
    const mode = $('#repair-mode')?.value || 'conservative';
    const generative = Boolean($('#allow-generative-attempt')?.checked);
    // Enhanced and ordinary speculative both re-check these flags at runtime.
    // Hiding them leaves a job that cannot be confirmed, or one that is created
    // and then rejected before it runs.
    const needsRisk = generative || mode === 'enhanced' || mode === 'speculative';
    const needsEnhanced = mode === 'enhanced';
    const needsMaskGate = on || needsRisk;
    const riskAck = $('#risk-ack')?.closest('label');
    if (riskAck) riskAck.style.display = needsRisk ? '' : 'none';
    const allowEnhanced = $('#allow-enhanced')?.closest('label');
    if (allowEnhanced) allowEnhanced.style.display = needsEnhanced ? '' : 'none';
    const maskGate = $('#mask-gate-confirmed')?.closest('label');
    if (maskGate) maskGate.style.display = needsMaskGate ? '' : 'none';
    syncRegionUi();
  };
  const fileUrl = (value) => value || '';

  function showPage(page) {
    $$('.page').forEach(item => {
      const active = item.id === `page-${page}`;
      item.classList.toggle('active', active);
      item.classList.toggle('hidden', !active);
    });
    $$('.nav-item[data-page]').forEach(item => item.classList.toggle('active', item.dataset.page === page));
    const title = { workbench: '修复工作台', tools: '工具与评测', history: '历史记录' }[page];
    $('#page-title').textContent = title;
    if (page === 'history') renderHistory();
  }

  function showView(view) {
    $$('.view-tab').forEach(item => item.classList.toggle('active', item.dataset.view === view));
    $('#edit-view').classList.toggle('active', view === 'edit');
    $('#results-view').classList.toggle('active', view === 'results');
    if (view === 'results') requestAnimationFrame(layoutComparison);
  }

  function setTool(tool) {
    state.tool = tool;
    $$('.tool-button').forEach(item => item.classList.toggle('active', item.dataset.tool === tool));
    canvas.classList.toggle('sam-point', tool === 'sam-point');
    canvas.classList.toggle('sam-box', tool === 'sam-box');
    canvas.classList.toggle('eraser', tool === 'eraser');
    canvas.classList.toggle('rect-select', tool === 'rect');
    $('#mask-status').textContent = tool === 'rect' ? '红色矩形：在图上拖动框出要修复的区域' :
      tool === 'sam-point' ? '左键添加前景点，右键添加背景点' :
      tool === 'sam-box' ? '在图像上拖动框选目标' :
      tool === 'uncertain' || tool === 'uncertainty' ? '红色：不确定区（默认禁修，须强制放行才可保守）' :
      tool === 'unrestorable' ? '深红：不可复原区（禁止任何像素修复）' :
      tool === 'eraser' ? '橡皮：同时清除可修区域与标红区' :
      '琥珀/红：可修露底选区（仅保守）';
  }

  function updateCanvasSize() {
    if (!state.image || !image.naturalWidth) return;
    imageWrap.style.transform = `scale(${state.zoom})`;
    requestAnimationFrame(() => {
      canvas.style.width = `${image.clientWidth}px`;
      canvas.style.height = `${image.clientHeight}px`;
      if (regionCanvas) {
        regionCanvas.style.width = `${image.clientWidth}px`;
        regionCanvas.style.height = `${image.clientHeight}px`;
      }
      $('#mask-status').textContent = `修复范围 ${Math.round(maskFraction() * 10000) / 100}% · ${state.image.width} × ${state.image.height}`;
    });
  }

  function maskFraction() {
    if (!canvas.width || !canvas.height) return 0;
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) count++;
    return count / (canvas.width * canvas.height);
  }

  function drawStroke(context, stroke, color, operation = 'source-over') {
    if (!stroke.points.length) return;
    context.save();
    context.globalCompositeOperation = operation;
    context.strokeStyle = color; context.fillStyle = color;
    context.lineCap = 'round'; context.lineJoin = 'round'; context.lineWidth = stroke.size;
    if (stroke.points.length === 1) {
      context.beginPath(); context.arc(stroke.points[0][0], stroke.points[0][1], stroke.size / 2, 0, Math.PI * 2); context.fill();
    } else {
      context.beginPath(); context.moveTo(...stroke.points[0]);
      stroke.points.slice(1).forEach(point => context.lineTo(...point)); context.stroke();
    }
    context.restore();
  }

  function paintStroke(stroke) {
    if (isRegionTool(stroke.tool)) {
      if (regionCtx) drawStroke(regionCtx, stroke, REGION_COLORS[stroke.tool]);
      drawStroke(regionClassCtx, stroke, stroke.tool === 'unrestorable' ? '#ff0000' : '#0000ff');
      return;
    }
    const operation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
    drawStroke(ctx, stroke, '#f5a623', operation);
    if (stroke.tool === 'eraser') {
      if (regionCtx) drawStroke(regionCtx, stroke, '#000000', operation);
      drawStroke(regionClassCtx, stroke, '#000000', operation);
    }
  }

  function redrawMask() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (regionCtx) regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
    regionClassCtx.clearRect(0, 0, regionClasses.width, regionClasses.height);
    if (state.baseMask) ctx.drawImage(state.baseMask, 0, 0, canvas.width, canvas.height);
    // Replay both kinds in creation order so erasing and undo affect all layers.
    let maskIndex = 0, regionIndex = 0;
    state.strokeOrder.forEach(kind => paintStroke(kind === 'region'
      ? state.regionStrokes[regionIndex++] : state.maskStrokes[maskIndex++]));
    if (state.currentStroke) paintStroke(state.currentStroke);
    state.maskDirty = true;
    $('#mask-fraction').textContent = `${Math.round(maskFraction() * 10000) / 100}%`;
    setBusy(state.running);
    syncRegionUi();
  }

  function pointAt(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height];
  }


  function loadImage(record) {
    clearDamageHighlight(); hideModal('damage-modal'); hideModal('triple-modal'); resetAiReview();
    state.finalRepair = null; renderFinal(null);
    state.triple = { largeJobId: null, fullJobId: null };
    state.batchJobIds = [];
    renderBatchJobs([]);
    regionClassCtx.clearRect(0, 0, regionClasses.width, regionClasses.height);
    state.damageDrawMode = false; state.damageDraw = null; state.damageDrawTarget = null;
    state.damageSummary = ''; state.maskEditTarget = null;
    clearDamageOverlay();
    $('#canvas-wrap')?.classList.remove('damage-drawing');
    if (!record) {
      state.keepEmpty = true;
      state.image = state.job = state.candidate = state.proposal = null;
      state.registration = null; state.references = []; state.selectedReference = null;
      state.damageItems = []; state.selectedDamage = null;
      $('#mask-gate-confirmed').checked = false;
      $('#culture-summary').classList.add('hidden');
      state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
      state.proposalIds = [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (regionCtx) regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
      image.removeAttribute('src'); imageWrap.classList.add('hidden'); $('#empty-state').classList.remove('hidden');
      $('#image-name').textContent = '导入壁画图像'; $('#image-meta').textContent = '选择本地图片开始';
      $('#candidate-count').textContent = '0'; $('#mask-fraction').textContent = '';
      $('#job-status').textContent = '尚未开始'; $('#job-progress').style.width = '0%';
      $('#job-percent').textContent = '0%';
      $('#progress-estimate').textContent = '等待任务';
      $('[data-action="run-repair"]').disabled = true;
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      renderDamage();
      state.autoBusy = false; setAutoStatus(AUTO_HINT);
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      $('#event-log').innerHTML = '<div class="empty-log">完成修复范围后生成候选。</div>';
      return Promise.resolve();
    }
    state.keepEmpty = false;
    state.image = record; state.job = null; state.candidate = null; state.proposal = null;
    state.registration = null; state.references = []; state.selectedReference = null;
    state.damageItems = []; state.selectedDamage = null;
    $('#mask-gate-confirmed').checked = false;
    $('#culture-summary').classList.add('hidden');
    state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
    if (regionCtx) regionCtx.clearRect(0, 0, regionCanvas.width || 0, regionCanvas.height || 0);
    state.zoom = 1; state.promptPoints = []; state.promptLabels = []; state.promptBox = null;
    $('#image-name').textContent = record.name;
    $('#image-meta').textContent = `${record.width} × ${record.height} · 原始文件已归档`;
    $('#empty-state').classList.add('hidden'); imageWrap.classList.remove('hidden');
    state.autoBusy = false; setAutoStatus(AUTO_HINT);
    renderDamage();
    $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
    $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
    $('#candidate-count').textContent = '0';
    $('#job-status').textContent = '尚未开始'; $('#job-progress').style.width = '0%';
    $('#job-percent').textContent = '0%';
    $('#progress-estimate').textContent = '等待任务';
    $('#event-log').innerHTML = '<div class="empty-log">完成修复范围后生成候选。</div>';
    $('#mask-fraction').textContent = '';
    $('[data-action="save-result"]').disabled = true;
    $('[data-action="select-candidate"]').disabled = true;
    $('[data-action="export-archive"]').disabled = true;
    state.proposalIds = [];
    showView('edit'); showPage('workbench');
    invoke('setCurrentImage', record.id).catch(() => {});
    return new Promise(resolve => {
      image.onload = () => {
        canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
        regionClasses.width = image.naturalWidth; regionClasses.height = image.naturalHeight;
        if (regionCanvas) { regionCanvas.width = image.naturalWidth; regionCanvas.height = image.naturalHeight; regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height); }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        updateCanvasSize();
        setBusy(state.running);
        resolve();
      };
      image.onerror = () => { toast('图像无法载入。'); resolve(); };
      image.src = fileUrl(record.file_url);
    });
  }

  // Backend masks are 8-bit grayscale PNGs (white = editable). Canvas masks are
  // alpha-composited red overlays, so a plain drawImage() would make every pixel
  // opaque and turn the editable area into the whole image. Rebuild the alpha
  // channel from the luminance before it becomes baseMask.
  function maskLayerFrom(source, width, height) {
    const layer = document.createElement('canvas');
    layer.width = width; layer.height = height;
    const layerContext = layer.getContext('2d', { willReadFrequently: true });
    layerContext.drawImage(source, 0, 0, width, height);
    const sourceData = layerContext.getImageData(0, 0, width, height);
    const target = layerContext.createImageData(width, height);
    for (let i = 0; i < sourceData.data.length; i += 4) {
      const alpha = sourceData.data[i + 3];
      const luminance = (sourceData.data[i] + sourceData.data[i + 1] + sourceData.data[i + 2]) / 3;
      const editable = alpha > 127 && luminance > 127;
      target.data[i] = 216; target.data[i + 1] = 77; target.data[i + 2] = 69;
      target.data[i + 3] = editable ? 255 : 0;
    }
    layerContext.putImageData(target, 0, 0);
    return layer;
  }

  function installMaskFromSource(source) {
    const width = canvas.width || source.naturalWidth || source.width;
    const height = canvas.height || source.naturalHeight || source.height;
    if (!width || !height) { toast('选区载入时画布尺寸未知，请重新载入图像。'); return; }
    state.baseMask = maskLayerFrom(source, width, height);
    $('#mask-gate-confirmed').checked = false;
    state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.maskDirty = false;
    redrawMask(); state.maskDirty = false;
  }

  function installMask(url, expectedJobId = null) {
    const expectedImageId = state.image?.id;
    return new Promise((resolve, reject) => {
      const maskImage = new Image();
      maskImage.onload = () => {
        try {
          if (state.image?.id === expectedImageId && (!expectedJobId || state.job?.id === expectedJobId)) installMaskFromSource(maskImage);
          resolve();
        }
        catch (error) { reject(error); }
      };
      maskImage.onerror = () => { toast('选区读取失败。'); reject(new Error('选区读取失败。')); };
      maskImage.src = url;
    });
  }

  function showMaskFromBase64(data) {
    const expectedImageId = state.image?.id;
    return new Promise((resolve, reject) => {
      const maskImage = new Image();
      maskImage.onload = () => {
        try {
          if (state.image?.id === expectedImageId) installMaskFromSource(maskImage);
          resolve();
        }
        catch (error) { reject(error); }
      };
      maskImage.onerror = () => { toast('选区读取失败。'); reject(new Error('选区读取失败。')); };
      maskImage.src = `data:image/png;base64,${data}`;
    });
  }

  function exportMaskBase64() {
    const source = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const output = document.createElement('canvas'); output.width = canvas.width; output.height = canvas.height;
    const target = output.getContext('2d'); const data = target.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < source.data.length; i += 4) {
      const value = source.data[i + 3] > 0 ? 255 : 0;
      data.data[i] = data.data[i + 1] = data.data[i + 2] = value; data.data[i + 3] = 255;
    }
    target.putImageData(data, 0, 0);
    return output.toDataURL('image/png').split(',')[1];
  }

  async function openImage() {
    if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再打开新图像。');
    try { const record = await invoke('openImage'); if (record?.id) await loadImage(record); }
    catch (error) { notifyError(error); }
  }

  async function createJob() {
    if (!state.image || !maskHasPixels()) return toast('请先圈选需要修复的区域。');
    if (state.autoBusy) return toast('AI 分析仍在运行，请完成后再启动修复。');
    if (state.running) return toast('已有任务在执行，请等待完成或取消。');
    const generative = Boolean($('#allow-generative-attempt')?.checked);
    if (generative && !state.lastSnapshot?.models?.diffusion) {
      return toast('AI 尝试性补全需要已安装扩散模型（models/sd-inpainting）。');
    }
    if (isProfessional()) {
      if (!$('#mask-gate-confirmed')?.checked) return toast('专业分步：请勾选「已核对修复范围与依据」后再进入下一档。');
      if ($('#evidence-level').value === 'observation') return toast('专业分步：observation 只可记录观察，不能执行修复。');
      if (!generative && hasUnrestorableRegions()) return toast('存在不可复原红区：禁止任何像素修复，请只导出标注图与报告。');
      if (!generative && hasUncertainRegions()) {
        if ($('#evidence-level').value !== 'conservative' || $('#repair-mode').value !== 'conservative') {
          return toast('不确定红区仅可在强制放行后做保守修补。');
        }
        if (!$('#force-override')?.checked) return toast('不确定红区须勾选「专家强制放行」并填写理由。');
        if (!($('#force-override-reason')?.value || '').trim()) return toast('请填写强制放行理由。');
      }
      if (!generative && ($('#repair-mode').value === 'speculative' || $('#evidence-level').value === 'speculative')) {
        if ($('#repair-mode').value !== 'speculative' || $('#evidence-level').value !== 'speculative') {
          return toast('专业分步：推测须同时将修复模式与证据等级设为 speculative。');
        }
        if (!$('#risk-ack')?.checked) return toast('专业分步：推测性补全须勾选风险提示已知悉。');
      }
      if (!generative && ($('#repair-mode').value === 'enhanced' || $('#evidence-level').value === 'enhanced')) {
        if ($('#repair-mode').value !== 'enhanced' || $('#evidence-level').value !== 'enhanced') {
          return toast('专业分步：加强修复须同时将修复模式与证据等级设为 enhanced。');
        }
        if (!$('#allow-enhanced')?.checked) {
          return toast('专业分步：加强修复须勾选「允许加强修复」。');
        }
        if (!$('#risk-ack')?.checked) {
          return toast('专业分步：加强修复须勾选风险提示（文化边界）已知悉。');
        }
      }
    }
    if (generative && !$('#risk-ack')?.checked) {
      return toast('AI 尝试性补全须勾选风险提示：结果是推测，不是历史原貌。');
    }
    if (!isProfessional() && !generative) {
      const requestedMode = $('#repair-mode')?.value || 'conservative';
      if (requestedMode === 'enhanced' || requestedMode === 'speculative') {
        if (!$('#mask-gate-confirmed')?.checked) return toast('请勾选「已核对修复范围与依据」后再执行加强或推测修复。');
        if (!$('#risk-ack')?.checked) {
          return toast(requestedMode === 'speculative'
            ? '推测性补全须勾选风险提示已知悉。'
            : '加强修复须勾选风险提示（文化边界 / 非历史定稿）已知悉。');
        }
        if (requestedMode === 'enhanced' && !$('#allow-enhanced')?.checked) return toast('加强修复须勾选「允许加强」。');
      }
    }
    setBusy(true);
    try {
      const rationale = ($('#rationale')?.value || '').trim() ||
        `依据：仅处理经人工确认的当前选区，修复范围之外的像素保持不变。${generative ? ' AI 尝试性补全结果是推测，不是历史原貌，只能按假设归档。' : ($('#repair-mode').value === 'speculative' ? ' 推测结果仅作假设性建议，需人工复核。' : ' AI 补全仅为待复核候选，不代表历史原貌。')}`;
      const mask = await invoke('saveMask', state.image.id, exportMaskBase64(), JSON.stringify(state.proposalIds || []));
      const request = { image_id: state.image.id, mask_id: mask.id,
        backend: generative ? 'diffusion' : $('#repair-backend').value, planner: $('#planner').value,
        candidate_count: Number($('#candidate-number').value), instructions: $('#instructions').value,
        mode: generative ? 'speculative' : $('#repair-mode').value, registration_id: state.registration?.id || null,
        structure_control: $('#controlnet').checked,
        budget: { candidates: Number($('#candidate-number').value), revisions: Number($('#revisions').value), timeout_seconds: Number($('#timeout').value), max_api_calls: 1 },
        quality_thresholds: { boundary_pair_delta: 65 },
        workflow: currentWorkflow(),
        rationale,
        evidence_level: generative ? 'speculative' : ($('#evidence-level')?.value || null),
        expert_notes: ($('#expert-notes')?.value || '').trim(),
        mask_source: (state.proposalIds || []).length ? 'ai_candidate' : 'manual',
        allow_enhanced: Boolean($('#allow-enhanced')?.checked),
        risk_acknowledged: generative ? true : Boolean($('#risk-ack')?.checked),
        mask_gate_confirmed: Boolean($('#mask-gate-confirmed')?.checked),
        has_unrestorable: hasUnrestorableRegions(),
        has_uncertain: hasUncertainRegions(),
        force_override: Boolean($('#force-override')?.checked),
        force_override_reason: ($('#force-override-reason')?.value || '').trim(),
        allow_generative_attempt: generative };
      state.job = await invoke('createJob', JSON.stringify(request));
      state.candidate = null; state.maskDirty = false;
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      showView('results'); toast(isProfessional() ? '专业分步任务已加入队列（单块执行）。' : '修复任务已加入工作队列。');
      await refresh();
    } catch (error) { setBusy(false); notifyError(error); }
  }

  async function refresh() {
    if (!state.bridge) return;
    try {
      const hadImage = Boolean(state.image);
      const snapshot = await invoke('snapshot'); state.lastSnapshot = snapshot;
      renderModels(snapshot.models || {});
      if (!state.image && !state.keepEmpty && snapshot.images?.length) {
        const latestJob = snapshot.jobs?.[0];
        const record = snapshot.images.find(item => item.id === latestJob?.image_id) || snapshot.images[0];
        await loadImage(record);
      }
      if (state.job) state.job = snapshot.jobs.find(item => item.id === state.job.id) || state.job;
      else if (!hadImage && !state.keepEmpty && snapshot.jobs?.length) state.job = snapshot.jobs.find(item => item.image_id === state.image?.id) || null;
      const activeJob = ACTIVE.includes(state.job?.status) ? state.job : (snapshot.jobs || []).find(job => ACTIVE.includes(job.status));
      state.activeJobId = activeJob?.id || null;
      setBusy(Boolean(activeJob) || state.defectSubmitting);
      const visionBusy = snapshot.aux?.status === 'running' && (snapshot.aux.label || '').includes('AI 正在分析');
      $('[data-action="auto-restore"]').disabled = state.running || state.autoBusy || visionBusy || !state.image;
      $('#vision-provider').disabled = state.autoBusy || visionBusy;
      renderJob(state.job);
      syncRegionUi();
      renderBatchJobs(snapshot.jobs || []);
      renderHistory(snapshot.jobs || []);
      if (!$('#triple-modal').classList.contains('hidden')) renderTriple(snapshot.jobs || []);
      const damageJobKey = (snapshot.jobs || []).map(job => `${job.id}:${job.status}:${job.defect_proposal_id || ''}`).join('|');
      if (state.damageItems.length && damageJobKey !== state.damageJobKey) {
        state.damageJobKey = damageJobKey;
        renderDamage();
      }
      if (snapshot.aux_version !== state.auxVersion) {
        state.auxVersion = snapshot.aux_version;
        renderAux(snapshot.aux);
      }
      if (snapshot.registration) state.registration = snapshot.registration;
    } catch (error) { console.warn(error); }
  }

  function renderModels(models) {
    renderVisionProviderNote(models);
    renderDeepSeekKeyStatus(models.deepseek_vision);
    ['allow-generative-attempt', 'damage-generative'].forEach(id => {
      const node = $('#' + id);
      if (node) {
        node.disabled = !models.diffusion;
        node.closest('label')?.classList.toggle('disabled', !models.diffusion);
      }
    });
    const labels = [['LaMa', models.lama], ['SAM 2', models.sam2], ['Qwen-VL', models.vision], ['SD', models.diffusion]];
    $('#model-list').innerHTML = labels.map(([name, available]) => `<span class="model-tag ${available ? '' : 'off'}">${name}</span>`).join('');
    const ready = models.lama ? 'LaMa 可用' : '传统修补可用';
    $('#engine-status').innerHTML = `<i></i>${ready}`;
    const backend = $('#repair-backend'); const prior = backend.value;
    const items = [['opencv', '传统修补 · Telea / NS', true], ['lama', 'LaMa · 纹理补全', models.lama],
      ['telea', 'Telea · 细裂缝', true], ['ns', 'Navier–Stokes · 细裂缝', true],
      ['reference', '参考证据重建', models.reference], ['diffusion', '扩散推测补全', models.diffusion],
      ['denoise', '局部降噪', true], ['color', '周边色调匹配', true], ['deblur', '局部锐化', true],
      ['research', '壁画研究模型', models.research]];
    backend.innerHTML = items.filter(([, , available]) => available).map(([value, title]) => `<option value="${value}">${title}</option>`).join('');
    backend.value = items.some(([value, , available]) => value === prior && available) ? prior : (models.lama ? 'lama' : 'opencv');
    $('#planner option[value="local"]').disabled = !models.vision;
    $('#planner option[value="api"]').disabled = !models.api_configured;
  }

  function renderJob(job) {
    if (!job) return;
    if (job.mode === 'speculative') $('#archive-hypothesis').checked = true;
    const running = ACTIVE.includes(job.status);
    $('#job-status').textContent = ({ queued: '排队中', planning: '规划中', running: '修复中', evaluating: '质量检查', review: '等待复核', interrupted: '已中断', failed: '执行失败', cancelled: '已取消' })[job.status] || job.status;
    $('#job-progress').style.width = `${Math.max(0, Math.min(100, job.progress || 0))}%`;
    $('#job-percent').textContent = `${Math.max(0, Math.min(100, Math.round(job.progress || 0)))}%`;
    const startedAt = Date.parse(job.started_at || '');
    const liveElapsed = running && Number.isFinite(startedAt) ? Math.max(0, (Date.now() - startedAt) / 1000) : 0;
    const elapsed = Math.max(0, Math.round((job.elapsed_seconds || 0) + liveElapsed));
    let progressCopy = '等待任务';
    if (running && job.progress > 18 && job.progress < 100) {
      const fraction = Math.min(.98, Math.max(.02, (job.progress - 18) / 75));
      const remaining = Math.max(1, Math.round(elapsed * (1 - fraction) / fraction));
      progressCopy = `预计还需 ${remaining} 秒 · 已用 ${elapsed} 秒`;
    } else if (running) {
      const limit = Math.max(0, Number(job.budget?.timeout_seconds) || 0);
      progressCopy = limit
        ? `正在准备模型与修复方案 · 已用 ${elapsed} 秒 · 时间上限 ${limit} 秒`
        : `正在准备模型与修复方案 · 已用 ${elapsed} 秒`;
    } else if (job.status === 'review') {
      progressCopy = `已完成 · 共用 ${Math.round(job.elapsed_seconds || elapsed)} 秒`;
    } else if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
      progressCopy = `任务${job.status === 'failed' ? '失败' : job.status === 'cancelled' ? '已取消' : '已中断'} · 已用 ${elapsed} 秒`;
    }
    $('#progress-estimate').textContent = progressCopy;
    $('[data-action="run-repair"]').disabled = state.running || !state.image || !maskHasPixels() || (isProfessional() && hasUnrestorableRegions());
    $('[data-action="run-repair"]').classList.toggle('hidden', running);
    $('[data-action="cancel-job"]').classList.toggle('hidden', !state.activeJobId);
    $('#resume-button').classList.toggle('hidden', running || !['interrupted', 'failed', 'cancelled'].includes(job.status));
    $('#candidate-count').textContent = job.candidates?.length || 0;
    const lines = [];
    if (job.plan) lines.push(`<div class="event-line"><span class="event-time">修复计划</span>${escapeHtml(job.plan.summary || '')}</div>`);
    (job.events || []).slice(-40).forEach(event => lines.push(`<div class="event-line"><span class="event-time">${escapeHtml((event.time || '').slice(11, 19))}</span>${escapeHtml(event.message)}</div>`));
    if (job.error) lines.push(`<div class="event-line"><span class="event-time">错误</span>${escapeHtml(job.error)}</div>`);
    $('#event-log').innerHTML = lines.join('') || '<div class="empty-log">生成任务后显示规划、工具调用与检查结果。</div>';
    if (running) $('#event-log').scrollTop = $('#event-log').scrollHeight;
    renderCandidates(job);
  }

  function renderBatchJobs(jobs = state.lastSnapshot?.jobs || []) {
    const panel = $('#batch-jobs');
    panel.classList.toggle('hidden', !state.batchJobIds.length);
    const batch = state.batchJobIds.map(id => jobs.find(job => job.id === id)).filter(Boolean);
    const finished = batch.filter(job => !ACTIVE.includes(job.status)).length;
    $('#batch-summary').textContent = `本批次 ${state.batchJobIds.length} 项 · ${finished} 项已结束`;
    const labels = { queued: '排队中', planning: '规划中', running: '修复中', evaluating: '质量检查',
      review: '等待复核', failed: '失败', cancelled: '已取消', interrupted: '已中断' };
    $('#batch-job-list').innerHTML = batch.map((job, index) =>
      `<button class="button batch-job ${job.id === state.job?.id ? 'active' : ''}" data-batch-job="${job.id}" aria-pressed="${job.id === state.job?.id}">第 ${index + 1} 项 · ${escapeHtml(labels[job.status] || job.status)}</button>`).join('');
    $$('[data-batch-job]').forEach(button => button.addEventListener('click', () => selectBatchJob(button.dataset.batchJob).catch(notifyError)));
  }

  async function selectBatchJob(identifier) {
    const job = (state.lastSnapshot?.jobs || []).find(item => item.id === identifier);
    if (!job || job.image_id !== state.image?.id) return;
    state.job = job; state.candidate = null;
    resetAiReview();
    $('#review-note').value = job.review?.note || '';
    $('#archive-hypothesis').checked = Boolean(job.review?.archive_as_hypothesis || job.mode === 'speculative');
    $$('input[name="review-decision"]').forEach(input => { input.checked = input.value === job.review?.decision; });
    showPage('workbench'); showView('results');
    renderJob(job); renderBatchJobs();
    await installMask(job.mask_file_url, job.id);
    await refresh();
  }

  function renderCandidates(job) {
    const candidates = job.candidates || [];
    if (!candidates.length) {
      state.candidate = null;
      $('#candidate-strip').innerHTML = '<div class="empty-inline">此任务还没有修复候选</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">任务完成后，候选结果会显示在这里</div>';
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      return;
    }
    const selected = state.candidate?.id || job.selected_id || candidates[0].id;
    if (!state.candidate) state.candidate = candidates.find(item => item.id === selected) || candidates[0];
    const methodNames = { lama: 'LaMa 纹理补全', telea: 'Telea 快速行进', ns: 'Navier–Stokes 边缘传播',
      reference: '已核验壁画参考', diffusion: '扩散补全', imagine: 'Grok Imagine 全图重绘',
      denoise: '局部降噪', color: '局部色调匹配', deblur: '局部锐化', research: '研究模型' };
    const reviewLabels = { accepted: '已采纳', accepted_hypothesis: '假设归档', rejected: '已驳回',
      expert_review: '待专家复核', experimental_preview: '试验性预览', pending: '待审核' };
    $('#candidate-strip').innerHTML = candidates.map((candidate, index) => {
      const stateLabel = candidate.stale
        ? '<em class="candidate-state stale">已过期</em>'
        : (candidate.review?.status ? `<em class="candidate-state">${escapeHtml(reviewLabels[candidate.review.status] || candidate.review.status)}</em>` : '');
      return `<button class="candidate-card ${candidate.id === state.candidate?.id ? 'active' : ''}" data-candidate="${candidate.id}"><img src="${fileUrl(candidate.file_url)}"><span>${index + 1}. ${escapeHtml(methodNames[candidate.tool] || candidate.tool)}</span>${stateLabel}</button>`;
    }).join('');
    $$('.candidate-card').forEach(item => item.addEventListener('click', () => {
      state.candidate = candidates.find(candidate => candidate.id === item.dataset.candidate); renderCandidates(job); renderComparison();
      if (job.selected_id === state.candidate?.id && job.review?.note && !$('#review-note').value) $('#review-note').value = job.review.note;
    }));
    renderComparison();
    renderRegionPanel(job);
  }

  function renderAiReview(record) {
    const panel = $('#ai-review-panel');
    if (!panel) return;
    const regions = record?.regions || [];
    if (!record || !regions.length) {
      panel.classList.add('hidden');
      panel.innerHTML = '';
      return;
    }
    const risks = { low: '低', medium: '中', high: '高', unknown: '待判断' };
    panel.classList.remove('hidden');
    panel.innerHTML = `<div class="ai-review-head"><strong>AI 修复评估（建议，不改变本地结果）</strong>`
      + `<small>${escapeHtml(record.model_source || 'deepseek-flash')} · ${escapeHtml((record.created_at || '').slice(0, 16).replace('T', ' '))}</small></div>`
      + regions.map(region => `<article class="ai-region ${region.human_review_required ? 'needs-review' : ''}" data-ai-region="${escapeHtml(region.region_id)}">`
        + `<header><b>区域 ${region.index} · ${escapeHtml(region.type_label)}</b>`
        + `<span class="ai-risk">风险 ${escapeHtml(risks[region.risk_level] || region.risk_level)}</span>`
        + `${region.human_review_required ? '<span class="ai-flag">需要人工复核</span>' : '<span class="ai-flag ok">未标记必须复核</span>'}</header>`
        + `<p><b>本地检测</b> ${escapeHtml(region.local_evidence || '未记录可观察依据')}</p>`
        + `<p><b>本地修复</b> ${region.candidate
          ? `${escapeHtml(region.candidate.method || region.candidate.tool)}${region.candidate.parameters?.tool ? ` · ${escapeHtml(region.candidate.parameters.tool)}` : ''}`
          : '还没有本地修复候选'} · 修复范围占比 ${(Number(region.local_mask_fraction || 0) * 100).toFixed(2)}%</p>`
        + `<p><b>DeepSeek 观察</b> ${escapeHtml(region.observation)}</p>`
        + `<p><b>修复评价</b> ${escapeHtml(region.repair_assessment)}</p>`
        + `<p><b>修复建议</b> ${escapeHtml(region.repair_suggestion)}</p>`
        + `<p><b>判断依据</b> ${escapeHtml(region.basis)}</p>`
        + `<p><b>不确定性</b> ${escapeHtml(region.uncertainty)}</p>`
        + `<p class="ai-reason"><b>复核原因</b> ${escapeHtml(region.review_reason || '模型未给出原因')}</p>`
        + `</article>`).join('')
      + `<div class="notice">AI 评估仅为建议：不会修改修复范围、像素或审核状态。`
      + `${record.limitations ? ` ${escapeHtml(record.limitations)}` : ''}</div>`;
  }

  async function exportPdf(method, payload) {
    if (method === 'exportObservationCard' && !state.image) return toast('请先打开一幅壁画。');
    if (method === 'exportRepairReport' && !state.job) return toast('请先选择一个修复任务。');
    try {
      const result = await invoke(method, JSON.stringify(payload));
      if (!result?.saved) return;
      toast(`已导出 PDF（${result.pages} 页）。`);
    } catch (error) { notifyError(error); }
  }

  async function openDemoGuide() {
    if (!state.image) return toast('请先打开一幅壁画。');
    const status = await invoke('demoStatus', state.image.id);
    const steps = status.steps || [];
    $('#demo-progress').textContent = `${status.done}/${status.total}`;
    $('#demo-steps').innerHTML = steps.map((step, index) =>
      `<div class="demo-step ${step.done ? 'done' : ''}"><span class="demo-mark">${step.done ? '✓' : index + 1}</span>`
      + `<div><strong>${escapeHtml(step.label)}</strong><small>${escapeHtml(step.hint || '')}</small></div></div>`).join('');
    showModal('demo-modal');
  }

  async function reviewCandidateAction(action) {
    if (!state.job || !state.candidate) return toast('请先选择一个修复候选。');
    if (state.running || state.autoBusy) return toast('请等待当前任务完成。');
    const result = await invoke('reviewCandidate', JSON.stringify({
      job_id: state.job.id, candidate_id: state.candidate.id, action,
      note: $('#review-note').value || '',
      archive_as_hypothesis: Boolean($('#archive-hypothesis')?.checked),
    }));
    state.job = result.job;
    state.candidate = (result.job.candidates || []).find(item => item.id === result.candidate.id) || result.candidate;
    renderJob(state.job); renderCandidates(state.job);
    await refresh();
    toast(action === 'expert_review'
      ? '已标记需要专家复核；正式导出会被阻止。'
      : '已记录试验性预览：仅为推测性参考，不会作为正式审核结果导出。');
  }

  async function assessRepair() {
    if (!state.image || !state.job || !state.candidate) return toast('请先选择一个本地修复候选。');
    if (!state.lastSnapshot?.models?.deepseek_vision) return toast('请先在 API 设置保存 DeepSeek 密钥。');
    if (state.running || state.autoBusy) return toast('请等待当前任务完成后再评估。');
    const panel = $('#ai-review-panel');
    panel.classList.remove('hidden');
    panel.innerHTML = '<div class="empty-inline">DeepSeek 正在评估本次修复结果…</div>';
    try {
      await invoke('assessRepair', JSON.stringify({
        image_id: state.image.id, job_id: state.job.id, candidate_id: state.candidate.id,
        context: $('#ai-review-context')?.value || '',
      }));
    } catch (error) {
      panel.innerHTML = `<div class="ai-review-error">${escapeHtml(error.message || String(error))}</div>`
        + '<div class="notice">本地修复结果与人工审核不受影响，可继续操作。</div>';
      notifyError(error);
    }
  }

  async function acceptDetectedRegion(index) {
    const item = state.damageItems[index];
    if (!item || !state.image) return;
    const result = await invoke('acceptDetectedRegion', JSON.stringify({
      image_id: state.image.id, bbox: item.bbox,
      type: item.type === 'manual' ? 'other' : item.type,
      evidence: item.evidence || item.observation || '',
      recommendation: item.recommendation || undefined,
      model_source: item.model_source || 'AI 检测框',
    }));
    const defect = result.defect;
    const currentIndex = state.damageItems.indexOf(item);
    if (currentIndex >= 0) state.damageItems.splice(currentIndex, 1, defect);
    else state.damageItems.push(defect);
    state.selectedDamage = defect;
    showDamageHighlight(defect);
    renderDamage();
    renderReviewHistory();
    toast(result.generated === 'sam'
      ? '已直接确认为待修复区域（按 SAM 自动生成范围），现在可生成修复候选。'
      : '已直接确认为待修复区域（按 AI 框生成矩形范围），可先生成候选或再用画笔修整。');
  }

  async function autoConfirmDefect(index) {
    const item = state.damageItems[index];
    if (!item?.proposal_id || !state.image) return;
    const result = await invoke('autoConfirmDefect', JSON.stringify({
      image_id: state.image.id, proposal_ids: [item.proposal_id],
    }));
    if (result.skipped?.length) return toast(result.skipped[0].reason || '未能确认该区域。');
    const defect = result.defects?.[0];
    if (!defect) return toast('未能确认该区域。');
    Object.assign(item, defect);
    renderDamage();
    renderReviewHistory();
    const kind = result.generated?.[0];
    toast(kind === 'sam' ? '已确认并生成 SAM 修复范围，请核对后使用。'
      : kind === 'kept' ? '已确认该区域，沿用已有的修复范围。'
      : '已确认并按矩形生成修复范围，可再用画笔修正。');
  }

  async function autoConfirmAll() {
    if (!state.image) return toast('请先打开一幅壁画。');
    if (!state.damageItems.length) return toast('清单里还没有可确认的区域。');
    // AI boxes that were never stored as local regions are confirmed one by one first.
    for (const item of [...state.damageItems]) {
      if (item.proposal_id) continue;
      const index = state.damageItems.indexOf(item);
      if (index >= 0) await acceptDetectedRegion(index);
    }
    const ids = state.damageItems.filter(item => item.proposal_id).map(item => item.proposal_id);
    if (!ids.length) return toast('清单里还没有可确认的区域。');
    const result = await invoke('autoConfirmDefect', JSON.stringify({
      image_id: state.image.id, proposal_ids: ids,
    }));
    const byId = new Map((result.defects || []).map(defect => [defect.proposal_id, defect]));
    state.damageItems.forEach(item => {
      if (item.proposal_id && byId.has(item.proposal_id)) Object.assign(item, byId.get(item.proposal_id));
    });
    renderDamage();
    renderReviewHistory();
    const count = result.defects?.length || 0;
    toast(`已确认 ${count} 个区域${result.skipped?.length ? `，跳过 ${result.skipped.length} 个：${result.skipped[0].reason}` : ''}。`);
  }

  async function saveRecommendation(index) {
    const item = state.damageItems[index];
    if (!item?.proposal_id || !state.image) return;
    const methodNames = { lama: 'LaMa 纹理补全', telea: 'Telea 局部纹理传播',
      ns: 'Navier–Stokes 细线修补', color: '局部色调匹配' };
    const method = item.selectedMethod && item.selectedMethod !== 'recommended'
      ? (methodNames[item.selectedMethod] || item.selectedMethod)
      : (item.recommendation?.method || '');
    if (!method) return toast('请先在“修补方案”中选择一种方案，再保存为修复建议。');
    const { defect } = await invoke('editLocalDefect', state.image.id, JSON.stringify({
      action: 'recommendation', proposal_id: item.proposal_id, method,
    }));
    Object.assign(item, defect);
    renderDamage();
    renderReviewHistory();
    toast('修复建议已保存；旧修复候选与 AI 评估已标记为待更新。');
  }

  async function markRegionExpert(index) {
    const item = state.damageItems[index];
    if (!item?.proposal_id || !state.image) return;
    const { defect } = await invoke('editLocalDefect', state.image.id, JSON.stringify({
      action: 'expert_review', proposal_id: item.proposal_id, note: '标记需要专家复核',
    }));
    Object.assign(item, defect);
    renderDamage();
    renderReviewHistory();
    toast('已把该区域标记为需要专家复核；正式导出会被阻止，直到重新确认。');
  }

  async function renderReviewHistory() {
    const box = $('#review-history');
    if (!box || !state.image) return;
    try {
      const { events } = await invoke('listReviewEvents', state.image.id);
      if (!events?.length) { box.classList.add('hidden'); box.innerHTML = ''; return; }
      const labels = { identification_confirm: '识别确认', identification_expert_review: '标记专家复核',
        region_update: '修改区域/修复范围', recommendation_update: '修改修复建议', region_remove: '删除区域',
        candidate_accept: '接受候选', candidate_reject: '驳回候选', candidate_expert_review: '候选标记专家复核',
        candidate_experimental_preview: '试验性预览', candidate_select: '选定候选' };
      box.classList.remove('hidden');
      box.innerHTML = '<strong>审核记录</strong>' + events.slice(0, 12).map(event =>
        `<div class="review-event"><span>${escapeHtml((event.created_at || '').slice(5, 16).replace('T', ' '))}</span>`
        + `${escapeHtml(labels[event.action] || event.action)}${event.note ? ` · ${escapeHtml(event.note)}` : ''}</div>`).join('');
    } catch (error) { console.warn(error); }
  }

  function renderFinal(record) {
    const panel = $('#final-panel');
    if (!panel) return;
    if (!record) {
      panel.classList.add('hidden');
      $('#final-images').innerHTML = '';
      $('#final-note').textContent = '';
      $('[data-action="save-final"]').disabled = true;
      return;
    }
    panel.classList.remove('hidden');
    const regions = record.regions || [];
    const skipped = record.skipped || [];
    $('#final-meta').textContent = `${regions.length} 个区域已合并 · 已采纳 ${record.accepted_regions || 0} 个 · `
      + `尺寸检查${record.checks?.size_match ? '通过' : '未通过'} · 范围外${record.checks?.outside_unchanged ? '未改动' : '有改动'}`;
    const stamp = Date.now();
    $('#final-images').innerHTML =
      `<figure><img src="${fileUrl(state.image.file_url)}?v=${stamp}" alt="修复前"><figcaption>修复前</figcaption></figure>`
      + `<figure><img src="${fileUrl(record.file_url)}?v=${stamp}" alt="修复后总图"><figcaption>修复后总图</figcaption></figure>`
      + `<figure><img src="${fileUrl(record.difference_url)}?v=${stamp}" alt="差异图"><figcaption>差异图</figcaption></figure>`;
    $('#final-note').textContent = skipped.length
      ? `未合并 ${skipped.length} 个区域：${skipped[0].reason}${skipped.length > 1 ? ' 等' : ''}。`
      : '所有已完成的区域修复已合并；修复范围之外的像素保持原图不变。';
    $('[data-action="save-final"]').disabled = false;
  }

  async function composeFinalRepair() {
    if (!state.image) return toast('请先打开一幅壁画。');
    try {
      const record = await invoke('composeFinalRepair', JSON.stringify({ image_id: state.image.id }));
      state.finalRepair = record;
      renderFinal(record);
      toast('已生成所有区域合并后的修复后总图。');
    } catch (error) { notifyError(error); }
  }

  async function saveFinalRepair() {
    if (!state.image || !state.finalRepair) return toast('请先生成修复后总图。');
    try {
      const result = await invoke('saveFinalRepair', JSON.stringify({ image_id: state.image.id }));
      if (result.saved) toast('修复后总图已保存。');
    } catch (error) { notifyError(error); }
  }

  function renderRegionPanel(job = state.job) {    const panel = $('#region-panel');
    if (!panel) return;
    const candidate = state.candidate;
    const region = candidate?.region;
    if (!candidate?.region_url || !region) {
      panel.classList.add('hidden');
      $('#region-images').innerHTML = '';
      return;
    }
    const index = (job?.candidates || []).indexOf(candidate) + 1;
    panel.classList.remove('hidden');
    $('#region-meta').textContent = `候选 ${index} · 区域 ${region.width}×${region.height}px · ${candidate.tool}`;
    const stamp = Date.now();
    $('#region-images').innerHTML =
      `<figure><img src="${fileUrl(candidate.region_url)}?v=${stamp}" alt="区域修复结果"><figcaption>区域修复</figcaption></figure>` +
      (candidate.region_difference_url
        ? `<figure><img src="${fileUrl(candidate.region_difference_url)}?v=${stamp}" alt="区域差异"><figcaption>区域差异</figcaption></figure>` : '');
    $('[data-action="regenerate-region"]').disabled = ACTIVE.includes(job?.status);
  }

  function applySwipe() {
    const layer = $('.comparison-original');
    if (!layer) return;
    const ratio = Math.min(100, Math.max(0, Number($('#compare-range').value) || 0));
    layer.style.width = `${ratio}%`;
    const inner = layer.querySelector('img');
    if (inner) inner.style.width = ratio > 0 ? `${10000 / ratio}%` : '0';
  }

  function layoutComparison() {
    const frame = $('.comparison-frame');
    if (!frame || !state.image) return;
    const stage = $('#comparison-stage');
    const width = state.image.width, height = state.image.height;
    if (!width || !height) return;
    const availableWidth = stage.clientWidth - 2;
    const availableHeight = stage.clientHeight - 2;
    if (availableWidth <= 0 || availableHeight <= 0) return;
    const scale = Math.min(availableWidth / width, availableHeight / height);
    frame.style.width = `${Math.max(1, Math.round(width * scale))}px`;
    frame.style.height = `${Math.max(1, Math.round(height * scale))}px`;
    applySwipe();
  }

  function renderComparison() {
    const candidate = state.candidate; if (!candidate || !state.image) return;
    const stage = $('#comparison-stage'); const mode = $('#comparison-mode').value;
    const sourceUrl = fileUrl(state.image.file_url);
    const url = mode === 'difference' ? fileUrl(candidate.difference_url) : fileUrl(candidate.file_url);
    // One frame sized to the working image keeps the candidate and the original
    // on exactly the same rectangle; previously the overlay was sized to the
    // stage height while the candidate was letterboxed, so the swipe did not line up.
    stage.innerHTML = '';
    const frame = document.createElement('div');
    frame.className = 'comparison-frame';
    const base = document.createElement('img');
    base.className = 'comparison-image';
    base.src = url;
    frame.appendChild(base);
    if (mode === 'compare') {
      const layer = document.createElement('div');
      layer.className = 'comparison-original';
      const original = document.createElement('img');
      original.src = sourceUrl;
      layer.appendChild(original);
      frame.appendChild(layer);
      $('#compare-range').oninput = applySwipe;
    }
    stage.appendChild(frame);
    $('#compare-range').disabled = mode !== 'compare';
    layoutComparison();
    const metrics = candidate.metrics || {};
    const maskLabel = metrics.evaluated_mask === 'enhanced_work_mask'
      ? '扩大后的修复范围（enhanced_work_mask）外'
      : (metrics.evaluated_mask === 'original_mask' ? '原始修复范围（mask.png）外' : '修复范围外');
    $('#candidate-device').textContent = `${candidate.versions?.device || ''} · ${candidate.elapsed_seconds || 0}s`;
    $('#candidate-metrics').textContent = `${maskLabel}最大差异 ${metrics.outside_max_difference ?? '-'} · 边界色差 ${metrics.boundary_color_gap ?? '-'} · ${candidate.speculative ? '推测性内容需单独复核' : (candidate.enhanced || state.job?.mode === 'enhanced') ? '加强修复·选区附近纹理补全·非历史定稿' : '工程指标不代表历史真实性'}`;
    $('[data-action="select-candidate"]').disabled = ACTIVE.includes(state.job?.status);
    $('[data-action="select-candidate"]').textContent = jobSelected(candidate.id) ? '已选定候选' : '选定候选';
    const assessButton = $('#assess-repair-button');
    if (assessButton) {
      assessButton.disabled = !state.lastSnapshot?.models?.deepseek_vision
        || ACTIVE.includes(state.job?.status) || Boolean(state.autoBusy);
      assessButton.title = state.lastSnapshot?.models?.deepseek_vision
        ? '发送原图与本地修复图给 DeepSeek 评估，结果仅为建议' : '请先在 API 设置保存 DeepSeek 密钥';
    }
    $('[data-action="save-result"]').disabled = false;
    $('[data-action="export-archive"]').disabled = !state.job?.selected_id;
  }

  const jobSelected = id => state.job?.selected_id === id;

  function renderHistory(jobs = state.lastSnapshot?.jobs || []) {
    const query = ($('#history-filter').value || '').trim().toLocaleLowerCase();
    const images = state.lastSnapshot?.images || [];
    const items = jobs.filter(job => {
      const img = images.find(record => record.id === job.image_id);
      return !query || (img?.name || '').toLocaleLowerCase().includes(query);
    });
    $('#history-list').innerHTML = items.map(job => {
      const img = images.find(record => record.id === job.image_id);
      const status = ({ review: '等待复核', failed: '失败', cancelled: '已取消', interrupted: '已中断' })[job.status] || job.status;
      return `<button class="history-item" data-job="${job.id}" ${state.running ? 'disabled' : ''}><span><span class="history-name">${escapeHtml(img?.name || '壁画任务')}</span><span class="history-detail">${(job.created_at || '').slice(0, 16).replace('T', ' ')} · ${(job.candidates || []).length} 个候选</span></span><span class="history-state">${escapeHtml(status)}</span></button>`;
    }).join('') || '<div class="empty-inline">暂无修复记录</div>';
    $$('.history-item').forEach(item => item.addEventListener('click', async () => {
      if (state.running || state.autoBusy) { toast('任务或 AI 分析进行中，完成后再打开历史任务。'); return; }
      try {
        const job = items.find(record => record.id === item.dataset.job);
        const record = images.find(imageRecord => imageRecord.id === job.image_id);
        await loadImage(record);
        state.job = job; state.candidate = null;
        await installMask(job.mask_file_url);
        $('#repair-mode').value = job.mode || 'conservative'; $('#repair-backend').value = job.backend;
        $('#planner').value = ['rules', 'local', 'api'].includes(job.planner) ? job.planner : 'rules';
        $('#instructions').value = job.instructions || '';
        $('#review-note').value = job.review?.note || '';
        state.autoBusy = false;
        setAutoStatus(job.auto?.summary || AUTO_HINT);
        showPage('workbench'); showView(job.candidates?.length ? 'results' : 'edit'); await refresh();
      } catch (error) { notifyError(error); }
    }));
  }

  async function renderAux(aux) {
    if (!aux || aux.status === 'idle') return;
    if (aux.image_id && aux.image_id !== state.image?.id) return;
    if (aux.status === 'running') {
      $('#sam-status').textContent = aux.label || '处理中';
      if (/缺损/.test(aux.label || '')) $('#damage-status').textContent = '识别中';
      if ((aux.label || '').includes('AI 正在分析')) setAutoStatus(aux.label);
      return;
    }
    if (aux.status === 'error') {
      $('#sam-status').textContent = '出错';
      $('#proposal-note').textContent = aux.error || '工具执行失败。';
      if ((aux.label || '').includes('评估')) {
        const panel = $('#ai-review-panel');
        if (panel) {
          panel.classList.remove('hidden');
          panel.innerHTML = `<div class="ai-review-error">${escapeHtml(aux.error || 'AI 修复评估失败。')}</div>`
            + '<div class="notice">本地修复结果与人工审核不受影响，可继续操作。</div>';
        }
      }
      if (state.autoBusy) {
        state.autoBusy = false;
        setAutoStatus(`AI 缺损分析失败：${aux.error || '未知错误'} 可切换来源或手动圈选。`);
      }
      toast(aux.error || '辅助工具执行失败。');
      return;
    }
    const result = aux.result || {};
    if (result.assessment) {
      state.aiAssessment = result.assessment;
      renderAiReview(result.assessment);
      return;
    }
    if (result.auto) {
      const auto = result.auto;
      if (auto.image_id && auto.image_id !== state.image?.id) return;
      state.autoBusy = false;
      clearDamageHighlight();
      state.damageItems = auto.defects || [];
      const hasDefects = state.damageItems.length > 0;
      state.damageSummary = auto.summary || (hasDefects ? 'AI 分析完成，请人工核对缺损清单。' : '未发现明确缺损，可手动圈选。');
      state.selectedDamage = null;
      state.job = null;
      state.candidate = null;
      state.proposalIds = [];
      renderDamage();
      if (hasDefects) showModal('damage-modal');
      setAutoStatus(state.damageSummary);
      $('#repair-mode').value = 'conservative';
      if (auto.backend) $('#repair-backend').value = auto.backend;
      $('#planner').value = state.lastSnapshot?.models?.vision ? 'local' : 'rules';
      $('#rationale').value = auto.summary || 'AI 提出的缺损候选，已人工检查选区。';
      $('#instructions').value = auto.planning_notes || '文化背景未核实；只做最小保守修补，不臆造壁画内容。';
      renderCulturalContext(auto.cultural_context || {});
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      $('#candidate-count').textContent = '0';
      $('#job-status').textContent = '尚未开始';
      $('#job-progress').style.width = '0%';
      $('#job-percent').textContent = '0%';
      $('#progress-estimate').textContent = '等待任务';
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      showPage('workbench'); showView('edit');
      toast(hasDefects ? 'AI 已列出疑似缺损，请核对后再修复。' : '未发现明确缺损，可手动圈选。');
      await refresh();
      return;
    }
    if (result.items) {
      state.autoBusy = false;
      state.damageItems = result.items;
      state.damageSummary = result.limitations || '';
      state.selectedDamage = null;
      renderDamage();
      $('#damage-status').textContent = result.items.length ? `待确认 ${result.items.length} 项` : '未发现候选';
      toast(result.items.length ? `AI 提出 ${result.items.length} 个疑似缺损，请人工核验。` : 'AI 未发现候选，请人工检查原图。');
      return;
    }
    if (result.proposal) {
      state.proposal = result.proposal;
      if (result.proposal.file_url) {
        $('#proposal-image').src = `${fileUrl(result.proposal.file_url)}?v=${Date.now()}`;
        $('#proposal-image').classList.remove('hidden'); $('#proposal-empty').classList.add('hidden');
      }
      $('#proposal-note').textContent = `${result.proposal.metadata?.notice || '辅助选区待人工确认。'} 候选面积 ${((result.proposal.fraction || 0) * 100).toFixed(2)}%。`;
      $('[data-action="use-proposal"]').disabled = false; $('[data-action="merge-proposal"]').disabled = false;
      $('#sam-status').textContent = '已生成待确认候选'; showPage('tools');
      return;
    }
    if (result.references) { renderReferences(result.references); return; }
    if (result.registration) {
      state.registration = result.registration;
      $('#registration-note').textContent = `内点 ${state.registration.inliers} · 中位误差 ${state.registration.median_error_px.toFixed(2)} px · 请人工核验覆盖区域。`;
      $('#registration-preview').innerHTML = `<img src="${fileUrl(state.registration.overlay_url)}">`;
      $('[data-action="approve-reference"]').disabled = false;
      return;
    }
    if (result.notice) toast(result.notice);
  }

  const DAMAGE_LABELS = { crack: '裂隙', paint_loss: '剥落 / 颜料层缺失', plaster_loss: '地仗/灰泥缺失', flaking: '起甲/剥落', fading: '褪色', stain: '污染 / 污渍', other: '其他疑似缺损', manual: '手动标注缺损' };
  const DAMAGE_RISKS = { low: '低', medium: '中', high: '高', unknown: '待判断' };
  const needsManualSelection = (item) => item.requires_manual_selection && !item.proposal_id;

  function renderCulturalContext(context) {
    const panel = $('#culture-summary');
    if (!panel) return;
    const lines = [];
    if (context.visible_subject) lines.push(`<p><b>可见题材</b> ${escapeHtml(context.visible_subject)}</p>`);
    if (context.possible_period_or_tradition) lines.push(`<p><b>可能年代 / 传统</b> ${escapeHtml(context.possible_period_or_tradition)} <small>待核实</small></p>`);
    if (context.visual_clues?.length) lines.push(`<p><b>图像线索</b> ${escapeHtml(context.visual_clues.join('；'))}</p>`);
    if (context.preservation_constraints?.length) lines.push(`<p><b>保护约束</b> ${escapeHtml(context.preservation_constraints.join('；'))}</p>`);
    if (context.verification_needed?.length) lines.push(`<p><b>待查来源</b> ${escapeHtml(context.verification_needed.join('；'))}</p>`);
    panel.innerHTML = `<strong>文化背景观察 · 未经考证</strong>${lines.join('') || '<p>AI 未能可靠判断。</p>'}<small>如需作为修复依据，请先结合馆藏记录或专家意见核实。</small>`;
    panel.classList.remove('hidden');
  }

  function damageRecommendation(item) {
    if (item.recommendation?.method) return item.recommendation.method;
    if (item.requires_manual_selection) return '先人工确认，再选择保守修补';
    const size = item.size || 'uncertain';
    if (item.type === 'stain') return '局部色调匹配';
    if (item.type === 'crack') return size === 'large' ? 'LaMa 候选；长裂缝需人工复核' : 'Navier–Stokes 细线修补';
    if (['paint_loss', 'plaster_loss'].includes(item.type) && size === 'large') return 'LaMa 纹理补全';
    if (['paint_loss', 'plaster_loss', 'flaking'].includes(item.type)) return 'Telea 局部纹理传播';
    return '先人工确认，再选择保守修补';
  }

  function bboxPercent(bbox) {
    const values = (bbox || [0, 0, 0, 0]).map(Number);
    const x0 = Math.min(values[0], values[2]), y0 = Math.min(values[1], values[3]);
    const width = Math.abs(values[2] - values[0]), height = Math.abs(values[3] - values[1]);
    return { left: `${x0 / 10}%`, top: `${y0 / 10}%`, width: `${width / 10}%`, height: `${height / 10}%` };
  }

  function showDamageHighlight(item) {
    const node = $('#damage-highlight');
    if (!node || !item || !Array.isArray(item.bbox) || item.bbox.length !== 4) return;
    Object.assign(node.style, bboxPercent(item.bbox));
    node.classList.add('visible');
  }

  function clearDamageHighlight() {
    const node = $('#damage-highlight');
    if (!node) return;
    node.classList.remove('visible');
    node.style.left = node.style.top = node.style.width = node.style.height = '';
  }

  // Candidate boxes, translucent masks and numbered type labels share one
  // overlay so the canvas shows box and mask as different things.
  function renderDamageOverlay() {
    const overlay = $('#damage-overlay');
    if (!overlay) return;
    const items = state.damageItems || [];
    if (!state.image || !items.length) {
      overlay.classList.add('hidden');
      overlay.innerHTML = '';
      return;
    }
    overlay.innerHTML = items.map((item, index) => {
      const label = `${index + 1} · ${DAMAGE_LABELS[item.type] || '疑似缺损'} · ${item.model_source || 'Qwen3-VL'}`;
      return `<div class="damage-region" data-region-index="${index}" data-mask-state="${escapeHtml(item.mask_status || '')}">`
        + `<span class="damage-region-label">${escapeHtml(label)}</span></div>`;
    }).join('') + '<canvas class="damage-mask-overlay"></canvas>';
    // The page's Content-Security-Policy blocks inline style attributes, so the
    // box geometry is applied through CSSOM after the nodes exist.
    $$('.damage-region', overlay).forEach((node, index) => {
      Object.assign(node.style, bboxPercent(items[index].bbox));
    });
    overlay.classList.remove('hidden');
    const maskCanvas = $('.damage-mask-overlay', overlay);
    if (!maskCanvas) return;
    maskCanvas.width = state.image.width;
    maskCanvas.height = state.image.height;
    const maskContext = maskCanvas.getContext('2d');
    items.filter(item => item.mask_overlay_data).forEach(item => {
      const picture = new Image();
      picture.onload = () => maskContext.drawImage(picture, 0, 0, maskCanvas.width, maskCanvas.height);
      picture.src = `data:image/png;base64,${item.mask_overlay_data}`;
    });
  }

  function clearDamageOverlay() {
    const overlay = $('#damage-overlay');
    if (!overlay) return;
    overlay.classList.add('hidden');
    overlay.innerHTML = '';
  }

  function resetAiReview() {
    state.aiAssessment = null;
    const panel = $('#ai-review-panel');
    if (panel) { panel.classList.add('hidden'); panel.innerHTML = ''; }
  }

  function renderTriple(jobs = state.lastSnapshot?.jobs || []) {
    if (!state.image) return;
    const original = $('#triple-original');
    if (original) original.src = `${fileUrl(state.image.file_url)}?v=${state.image.id}`;
    const update = (jobId, imageId, statusId) => {
      const job = jobs.find(item => item.id === jobId);
      const image = $('#' + imageId);
      const status = $('#' + statusId);
      if (!job) {
        if (status) status.textContent = '未生成';
        if (image) { image.classList.add('hidden'); image.removeAttribute('src'); }
        return;
      }
      if (ACTIVE.includes(job.status)) {
        if (status) status.textContent = `生成中…（${job.status}）`;
        return;
      }
      if (job.status === 'failed') {
        if (status) status.textContent = `失败：${job.error || '未知错误'}`;
        return;
      }
      const candidate = (job.candidates || [])[0];
      if (candidate?.file_url && image) {
        image.src = `${fileUrl(candidate.file_url)}?v=${candidate.id}`;
        image.classList.remove('hidden');
      }
      if (status) {
        status.textContent = job.mode === 'speculative'
          ? '已完成 · 推测性结果（只能按假设归档）'
          : '已完成';
      }
    };
    update(state.triple.largeJobId, 'triple-large', 'triple-large-status');
    update(state.triple.fullJobId, 'triple-full', 'triple-full-status');
    const canDiffusion = Boolean(state.lastSnapshot?.models?.diffusion);
    const canCloud = Boolean(state.lastSnapshot?.imagine?.configured);
    const busy = state.running || state.autoBusy;
    const largeButton = $('[data-action="generate-large-panel"]');
    const fullButton = $('[data-action="generate-full-panel"]');
    if (largeButton) {
      largeButton.disabled = !canDiffusion || busy;
      largeButton.title = canDiffusion ? '' : '需要安装扩散模型 models/sd-inpainting';
    }
    if (fullButton) {
      fullButton.disabled = !(canDiffusion || canCloud) || busy;
      fullButton.textContent = canCloud ? '生成全图（Grok）' : '生成全图';
      fullButton.title = canCloud ? '使用完整原图调用 Grok Imagine 重绘' : '使用本地扩散全图生成';
    }
  }

  function openTripleView() {
    if (!state.image) return toast('请先打开一幅壁画。');
    renderTriple();
    showModal('triple-modal');
  }

  async function generateTriple(kind) {
    if (!state.image) return toast('请先打开一幅壁画。');
    if (kind === 'large' && !state.lastSnapshot?.models?.diffusion) {
      return toast('大面积修补需要已安装本地扩散模型（models/sd-inpainting）。');
    }
    if (kind === 'full' && !state.lastSnapshot?.models?.diffusion && !state.lastSnapshot?.imagine?.configured) {
      return toast('全图生成需要本地扩散模型或已配置 Grok Imagine。');
    }
    if (state.running || state.autoBusy) return toast('当前有任务在运行，请稍候。');
    const button = $(`[data-action="generate-${kind}-panel"]`);
    if (button) button.disabled = true;
    try {
      const job = await invoke(kind === 'large' ? 'generateLargeRepair' : 'generateFullImage', state.image.id);
      state.triple[kind === 'large' ? 'largeJobId' : 'fullJobId'] = job.id;
      renderTriple();
      toast(kind === 'large'
        ? '已启动 AI 大面积修补；生成结果是推测，不是历史原貌。'
        : '已启动全图 AI 生成；结果是推测，不是历史原貌。');
      await refresh();
    } catch (error) {
      notifyError(error);
      renderTriple();
    }
  }

  function openDamageModal() {
    if (!state.image) return toast('请先打开一幅壁画。');
    if (state.damageDrawMode) {
      state.damageDrawMode = false; state.damageDraw = null; state.damageDrawTarget = null;
      $('#canvas-wrap')?.classList.remove('damage-drawing');
      clearDamageHighlight();
    }
    renderDamage();
    showModal('damage-modal');
    renderReviewHistory();
  }

  function renderDamage() {
    const list = $('#damage-list');
    if (!list) return;
    const count = state.damageItems.length;
    const status = $('#damage-status');
    if (status) status.textContent = count ? `共 ${count} 项` : (state.damageSummary ? '无候选' : '待分析');
    const badge = $('#damage-count');
    if (badge) badge.textContent = String(count);
    const runAll = $('[data-action="run-all-defects"]');
    if (!state.image || !count) {
      list.innerHTML = '<div class="empty-inline">点击“手动添加缺损区域”可在图上框选。</div>';
      $('#damage-note').textContent = state.damageSummary || '点击“AI 分析缺损与文化线索”可生成缺损清单。';
      $('#damage-image').classList.add('hidden'); $('#damage-empty').classList.remove('hidden');
      if (runAll) runAll.disabled = true;
      clearDamageOverlay();
      return;
    }
    const jobs = state.lastSnapshot?.jobs || [];
    const busy = state.running || state.autoBusy;
    const generative = Boolean($('#damage-generative')?.checked);
    list.innerHTML = state.damageItems.map((item, index) => {
      const [x0, y0, x1, y1] = item.bbox || [0, 0, 0, 0];
      const extent = Math.max(0, (x1 - x0) * (y1 - y0)) / 100;
      const job = item.proposal_id && jobs.find(record => record.defect_proposal_id === item.proposal_id);
      const statusText = job ? `任务${['review', 'completed'].includes(job.status) ? '已完成' : job.status === 'failed' ? '失败' : '进行中'}` : '';
      const method = item.recommendation?.method || damageRecommendation(item);
      const selectedMethod = item.selectedMethod || 'recommended';
      const alreadyQueued = job && ACTIVE.includes(job.status);
      const disabled = alreadyQueued || busy;
      const pending = needsManualSelection(item);
      const confirmed = item.confirmed !== false;
      const maskText = ({ sam: '修复范围：SAM 2.1 候选', needs_review: '修复范围：未可靠分割，请人工修正',
        observation: '修复范围：观察类，不默认补洞', manual: '修复范围：人工修正' })[item.mask_status]
        || (item.mask_status ? `修复范围：${item.mask_status}` : '');
      const hint = pending ? '模型框仅用于定位。请核对图像后手动框选。'
        : (item.risk_notice || item.mask_notice || (!item.proposal_id ? '暂无修复范围，运行时会按框生成矩形范围。' : ''));
      const details = [item.size, `框选约 ${extent.toFixed(2)}%`,
        Number.isFinite(item.confidence) ? `置信 ${Math.round(item.confidence * 100)}%` : '',
        item.model_source, item.risk_level ? `风险 ${DAMAGE_RISKS[item.risk_level] || '待判断'}` : ''].filter(Boolean);
      const observations = [['可观察现象', item.observation], ['依据', item.evidence],
        ['模型修复建议', item.repair_suggestion], ['不确定性', item.uncertainty]]
        .filter(([, value]) => value).map(([label, value]) => `<small>${label}：${escapeHtml(value)}</small>`).join('');
      const identification = item.identification_status || (item.confirmed ? 'confirmed' : 'pending');
      const confirmButton = item.proposal_id && !confirmed
        ? `<button class="button" data-confirm-defect="${index}" ${disabled ? 'disabled' : ''}>确认此项</button>` : '';
      const autoConfirmButton = item.proposal_id && (identification !== 'confirmed' || !item.mask_file_url)
        ? `<button class="button primary" data-auto-confirm="${index}" ${disabled ? 'disabled' : ''}>直接确认为待修复区域</button>` : '';
      const acceptRegionButton = !item.proposal_id
        ? `<button class="button primary" data-accept-region="${index}" ${disabled ? 'disabled' : ''}>直接确认为待修复区域</button>` : '';
      const removeButton = item.proposal_id
        ? `<button class="icon-button" data-remove-defect="${index}" ${disabled ? 'disabled' : ''} title="删除此项">删除</button>` : '';
      const maskButton = `<button class="icon-button" data-edit-mask="${index}" ${disabled ? 'disabled' : ''}>${item.mask_file_url ? '修改修复范围' : '绘制修复范围'}</button>`;
      const repairButton = item.proposal_id && item.mask_file_url && confirmed
        ? `<button class="button" data-repair-defect="${index}" ${disabled ? 'disabled' : ''}>生成修复候选</button>` : '';
      const recommendationButton = item.proposal_id
        ? `<button class="icon-button" data-save-recommendation="${index}" ${disabled ? 'disabled' : ''} title="把当前选择的修补方案保存为该区域的修复建议；旧候选将标记为待更新">保存建议</button>` : '';
      const regionExpertButton = item.proposal_id
        ? `<button class="icon-button" data-region-expert="${index}" ${disabled ? 'disabled' : ''}>标记需专家复核</button>` : '';
      const identificationText = ({ pending: '识别待确认', confirmed: '识别已确认',
        expert_review: '识别已确认 · 需专家复核' })[identification] || identification;
      const revisionText = item.revision ? ` · 版本 ${item.revision}` : '';
      // Only a persisted, unconfirmed proposal blocks running; a bare model box
      // still runs through the existing manual-selection path.
      const canRun = !item.proposal_id || confirmed;
      return `<div class="damage-item ${item === state.selectedDamage ? 'selected' : ''}" data-damage-row="${index}">` +
        `<div class="damage-item-main"><button class="damage-select" data-damage="${index}">${String(index + 1).padStart(2, '0')} ${escapeHtml(DAMAGE_LABELS[item.type] || '其他疑似缺损')}</button>` +
        `<small>${escapeHtml(details.join(' · '))}</small>${observations}` +
        `<small class="defect-state">${escapeHtml(identificationText + revisionText)}</small>` +
        `<small>${escapeHtml(method ? `建议：${method}` : '')}</small><label class="damage-method">修补方案 <select data-method-defect="${index}" ${disabled || generative ? 'disabled' : ''}>` +
        `<option value="recommended" ${selectedMethod === 'recommended' ? 'selected' : ''}>采用建议</option>` +
        `<option value="lama" ${selectedMethod === 'lama' ? 'selected' : ''} ${state.lastSnapshot?.models?.lama ? '' : 'disabled'}>LaMa AI 纹理补全${state.lastSnapshot?.models?.lama ? '' : '（未安装，将回退 Telea）'}</option>` +
        `<option value="telea" ${selectedMethod === 'telea' ? 'selected' : ''}>Telea 局部纹理传播</option>` +
        `<option value="ns" ${selectedMethod === 'ns' ? 'selected' : ''}>Navier–Stokes 细线修补</option>` +
        `<option value="color" ${selectedMethod === 'color' ? 'selected' : ''}>局部色调匹配</option></select></label>` +
        `${maskText ? `<small>${escapeHtml(maskText)}${confirmed ? ' · <span class="damage-confirmed">已确认</span>' : ''}</small>` : ''}` +
        `${hint ? `<small>${escapeHtml(hint)}</small>` : ''}${statusText ? `<small>${statusText}</small>` : ''}</div>` +
        `<div class="damage-item-actions">${autoConfirmButton}${acceptRegionButton}${confirmButton}${maskButton}${removeButton}${recommendationButton}${regionExpertButton}${repairButton}${pending
          ? `<button class="button" data-draw-defect="${index}" ${disabled ? 'disabled' : ''}>手动框选此候选</button>`
          : `<button class="button" data-run-defect="${index}" ${disabled || !canRun ? 'disabled' : ''}>运行修复</button>`}</div></div>`;
    }).join('');
    $$('[data-damage]').forEach(button => button.addEventListener('click', () => selectDamage(Number(button.dataset.damage))));
    $$('[data-method-defect]').forEach(select => select.addEventListener('change', () => {
      state.damageItems[Number(select.dataset.methodDefect)].selectedMethod = select.value;
    }));
    $$('[data-run-defect]').forEach(button => button.addEventListener('click', () => runDefectRepairs([Number(button.dataset.runDefect)]).catch(notifyError)));
    $$('[data-draw-defect]').forEach(button => button.addEventListener('click', () => beginManualDamage(state.damageItems[Number(button.dataset.drawDefect)])));
    $$('[data-confirm-defect]').forEach(button => button.addEventListener('click', () => confirmLocalDefect(Number(button.dataset.confirmDefect)).catch(notifyError)));
    $$('[data-remove-defect]').forEach(button => button.addEventListener('click', () => removeLocalDefect(Number(button.dataset.removeDefect)).catch(notifyError)));
    $$('[data-edit-mask]').forEach(button => button.addEventListener('click', () => beginMaskEdit(state.damageItems[Number(button.dataset.editMask)]).catch(notifyError)));
    $$('[data-repair-defect]').forEach(button => button.addEventListener('click', () => {
      const item = state.damageItems[Number(button.dataset.repairDefect)];
      generateRegionRepairs([item?.proposal_id]).catch(notifyError);
    }));
    $$('[data-save-recommendation]').forEach(button => button.addEventListener('click', () => {
      saveRecommendation(Number(button.dataset.saveRecommendation)).catch(notifyError);
    }));
    $$('[data-region-expert]').forEach(button => button.addEventListener('click', () => {
      markRegionExpert(Number(button.dataset.regionExpert)).catch(notifyError);
    }));
    $$('[data-auto-confirm]').forEach(button => button.addEventListener('click', () => {
      autoConfirmDefect(Number(button.dataset.autoConfirm)).catch(notifyError);
    }));
    $$('[data-accept-region]').forEach(button => button.addEventListener('click', () => {
      acceptDetectedRegion(Number(button.dataset.acceptRegion)).catch(notifyError);
    }));
    const generativeToggle = $('#damage-generative');
    if (generativeToggle) generativeToggle.onchange = () => renderDamage();
    const first = state.damageItems[0];
    const notice = generative
      ? 'AI 生成式补全（实验）：每项会用扩散模型按框直接生成，结果是推测，只能按假设归档；可能修坏，可重跑。'
      : `${state.damageSummary} 首项建议：${damageRecommendation(first)}。`;
    const pendingCount = state.damageItems.filter(needsManualSelection).length;
    const unconfirmedCount = state.damageItems.filter(item => item.proposal_id && item.confirmed === false).length;
    $('#damage-note').textContent = notice
      + (unconfirmedCount ? ` ${unconfirmedCount} 项尚未确认，确认后才能运行修复。` : '')
      + (pendingCount ? ` ${pendingCount} 项待手动框选，批量运行会跳过这些项。` : '');
    if (runAll) {
      runAll.disabled = busy || !state.damageItems.some(item => !needsManualSelection(item)
        && (!item.proposal_id || item.confirmed !== false)
        && (!item.proposal_id || !jobs.some(job => job.defect_proposal_id === item.proposal_id && ACTIVE.includes(job.status))));
    }
    renderDamageOverlay();
  }

  async function confirmLocalDefect(index) {
    const item = state.damageItems[index];
    if (!item || !item.proposal_id || !state.image) return;
    const { defect } = await invoke('editLocalDefect', state.image.id, JSON.stringify({
      action: 'confirm', proposal_id: item.proposal_id }));
    Object.assign(item, defect);
    renderDamage();
    toast('已确认该缺损项；修复范围不准确时请先“修改修复范围”，再运行修复。');
  }

  async function removeLocalDefect(index) {
    const item = state.damageItems[index];
    if (!item || !item.proposal_id || !state.image) return;
    await invoke('editLocalDefect', state.image.id, JSON.stringify({
      action: 'remove', proposal_id: item.proposal_id }));
    state.damageItems = state.damageItems.filter(entry => entry !== item);
    if (state.selectedDamage === item) { state.selectedDamage = null; clearDamageHighlight(); }
    if (state.maskEditTarget === item) state.maskEditTarget = null;
    renderDamage();
    toast('已删除该缺损项。');
  }

  async function beginMaskEdit(item) {
    if (!item || !state.image) return;
    if (state.running || state.autoBusy) return toast('请等待当前分析或修复任务完成。');
    state.maskEditTarget = item;
    hideModal('damage-modal');
    showPage('workbench'); showView('edit');
    if (item.mask_file_url) {
      await installMask(item.mask_file_url);
    } else {
      state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.baseMask = null;
      redrawMask();
    }
    setTool('brush');
    toast('用画笔修正该缺损的修复范围，然后点“保存该区域范围”。');
  }

  async function saveMaskToDefect() {
    const item = state.maskEditTarget;
    if (!item || !item.proposal_id || !state.image) return toast('请先从缺损清单选择“修改修复范围”。');
    if (!maskHasPixels()) return toast('画布上没有选区像素，请先圈选需要修复的区域。');
    const { defect } = await invoke('editLocalDefect', state.image.id, JSON.stringify({
      action: 'update', proposal_id: item.proposal_id, mask_data: exportMaskBase64() }));
    Object.assign(item, defect);
    renderDamage();
    toast('已保存人工修正的修复范围；请复核后再运行修复。');
  }

  async function generateRegionRepairs(proposalIds) {
    if (!state.image) return toast('请先打开一幅壁画。');
    if (state.running || state.autoBusy) return toast('请等待当前分析或修复任务完成。');
    const ids = [...new Set((proposalIds || []).filter(Boolean))];
    if (!ids.length) return toast('请先确认缺损项并准备好修复范围，再生成修复候选。');
    setBusy(true);
    try {
      const result = await invoke('generateRegionRepairs', JSON.stringify({
        image_id: state.image.id, proposal_ids: ids,
      }));
      const jobs = result.jobs || [];
      if (!jobs.length) {
        toast(result.skipped?.[0]?.reason || '没有可生成修复候选的缺损项。');
        return;
      }
      hideModal('damage-modal');
      state.batchJobIds = jobs.map(job => job.id);
      state.job = jobs[0]; state.candidate = null;
      resetAiReview();
      state.activeJobId = jobs.find(job => ACTIVE.includes(job.status))?.id || null;
      $('#review-note').value = '';
      $('#archive-hypothesis').checked = false;
      $$('input[name="review-decision"]').forEach(input => { input.checked = false; });
      showPage('workbench'); showView('results');
      renderJob(state.job); renderBatchJobs(jobs);
      const skipped = result.skipped?.length ? `；跳过 ${result.skipped.length} 项：${result.skipped[0].reason}` : '';
      toast(`已启动 ${jobs.length} 个区域修复候选${skipped}。`);
    } finally {
      setBusy(false);
      await refresh();
    }
  }

  async function runDefectRepairs(indexes) {
    if (!state.image) return toast('请先打开一幅壁画。');
    if (state.running || state.autoBusy) return toast('请等待当前分析或修复任务完成。');
    const generativeAttempt = Boolean($('#damage-generative')?.checked);
    if (generativeAttempt && !state.lastSnapshot?.models?.diffusion) {
      return toast('AI 生成式补全需要已安装扩散模型（models/sd-inpainting）。');
    }
    const requested = indexes.map(index => state.damageItems[index]).filter(Boolean);
    const selected = requested.filter(item => !needsManualSelection(item));
    const pendingCount = requested.length - selected.length;
    if (!selected.length) return toast(pendingCount ? '请手动框选 DeepSeek 候选后再运行。' : '没有可运行的缺损项。');
    state.defectSubmitting = true;
    setBusy(true);
    renderDamage();
    try {
      const proposalIds = [];
      const methods = {};
      const failed = [];
      for (const item of selected) {
        if (!item.proposal_id) {
          try {
            const made = await invoke('addManualDefect', state.image.id, JSON.stringify({
              bbox: item.bbox, type: item.type, size: item.size,
            }));
            item.proposal_id = made.defect.proposal_id;
            item.region_class = 'fillable_lacuna';
            item.repair_allowed = true;
            item.confirmed = true;
          } catch (error) {
            failed.push({ item, error });
            continue;
          }
        }
        proposalIds.push(item.proposal_id);
        if (item.selectedMethod && item.selectedMethod !== 'recommended') methods[item.proposal_id] = item.selectedMethod;
      }
      if (!proposalIds.length) {
        notifyError(failed[0]?.error || new Error('没有可运行的缺损项。'));
        return;
      }
      if (failed.length) toast(`${failed.length} 项无法自动生成修复范围，已跳过；其余继续运行。`);
      const result = await invoke('runDefectRepairs', JSON.stringify({
        image_id: state.image.id, proposal_ids: proposalIds, methods,
        generative_attempt: generativeAttempt,
      }));
      const count = result.jobs?.length || 0;
      if (!count) {
        toast(result.skipped?.[0]?.reason || '当前没有可运行的缺损项。');
        return;
      }
      hideModal('damage-modal');
      state.batchJobIds = result.jobs.map(job => job.id);
      state.job = result.jobs[0]; state.candidate = null;
      state.activeJobId = result.jobs.find(job => ACTIVE.includes(job.status))?.id || null;
      $('#review-note').value = '';
      $('#archive-hypothesis').checked = generativeAttempt;
      $$('input[name="review-decision"]').forEach(input => { input.checked = false; });
      showPage('workbench'); showView('results');
      renderJob(state.job); renderBatchJobs(result.jobs);
      await installMask(state.job.mask_file_url, state.job.id);
      const skipped = result.skipped?.length ? `；跳过 ${result.skipped.length} 项：${result.skipped[0].reason}` : '';
      const pending = pendingCount ? `；${pendingCount} 项尚待手动框选` : '';
      toast(`已启动 ${count} 个修复任务${skipped}${pending}。`);
    } finally {
      state.defectSubmitting = false;
      await refresh();
      renderDamage();
    }
  }

  async function selectDamage(index) {
    const item = state.damageItems[index];
    if (!item || !state.image) return;
    state.selectedDamage = item;
    showDamageHighlight(item);
    renderDamage();
    hideModal('damage-modal');
    showPage('workbench'); showView('edit');
    toast(`已框出第 ${index + 1} 项；再次打开“缺损清单”可继续操作。`);
    if (needsManualSelection(item)) return;
    try {
      const preview = await invoke('damagePreview', state.image.id, JSON.stringify(item));
      if (preview?.file_url) {
        $('#damage-image').src = `${fileUrl(preview.file_url)}?v=${Date.now()}`;
        $('#damage-image').classList.remove('hidden'); $('#damage-empty').classList.add('hidden');
      }
    } catch (error) { console.warn(error); }
  }

  function beginManualDamage(candidate = null) {
    if (!state.image) return toast('请先打开一幅壁画。');
    if (state.running || state.autoBusy) return toast('请等待当前修复或 AI 分析完成。');
    hideModal('damage-modal');
    state.damageDrawMode = true;
    state.damageDraw = null;
    state.damageDrawTarget = candidate;
    if (candidate) {
      state.selectedDamage = candidate;
      showDamageHighlight(candidate);
    } else {
      clearDamageHighlight();
    }
    showPage('workbench'); showView('edit');
    $('#canvas-wrap').classList.add('damage-drawing');
    toast(candidate ? '请核对候选位置，在图像上拖出实际修复范围。按 Esc 取消。' : '在图像上拖动框选缺损区域；按 Esc 取消。');
  }

  function updateDamageDraft() {
    const drawing = state.damageDraw;
    if (!drawing || !canvas.width || !canvas.height) return;
    const x0 = Math.min(drawing.start[0], drawing.end[0]) / canvas.width * 1000;
    const y0 = Math.min(drawing.start[1], drawing.end[1]) / canvas.height * 1000;
    const x1 = Math.max(drawing.start[0], drawing.end[0]) / canvas.width * 1000;
    const y1 = Math.max(drawing.start[1], drawing.end[1]) / canvas.height * 1000;
    const node = $('#damage-highlight');
    if (!node) return;
    Object.assign(node.style, bboxPercent([x0, y0, x1, y1]));
    node.classList.add('visible');
  }

  async function finishDamageDraw() {
    const drawing = state.damageDraw;
    if (!drawing || !state.image) return;
    const imageId = state.image.id;
    const target = state.damageDrawTarget;
    state.damageDraw = null;
    state.damageDrawMode = false;
    state.damageDrawTarget = null;
    $('#canvas-wrap').classList.remove('damage-drawing');
    const x0 = Math.min(drawing.start[0], drawing.end[0]) / canvas.width * 1000;
    const y0 = Math.min(drawing.start[1], drawing.end[1]) / canvas.height * 1000;
    const x1 = Math.max(drawing.start[0], drawing.end[0]) / canvas.width * 1000;
    const y1 = Math.max(drawing.start[1], drawing.end[1]) / canvas.height * 1000;
    if (x1 - x0 < 4 || y1 - y0 < 4) {
      clearDamageHighlight();
      if (target) showModal('damage-modal');
      toast('框选太小，请重新拖动一个更大的区域。');
      return;
    }
    const bbox = [x0, y0, x1, y1].map(value => Math.round(value));
    try {
      const request = target ? { bbox, candidate_id: target.candidate_id } : { bbox, type: 'manual', size: 'manual' };
      const { defect } = await invoke('addManualDefect', imageId, JSON.stringify(request));
      if (state.image?.id !== imageId) return;
      if (target) {
        if (!state.damageItems.includes(target)) return;
        target.bbox = defect.bbox;
        target.proposal_id = defect.proposal_id;
        target.recommendation = defect.recommendation;
        target.region_class = defect.region_class;
        target.repair_allowed = defect.repair_allowed;
        target.confirmed = defect.confirmed;
      } else {
        state.damageItems.push(defect);
      }
      const item = target || defect;
      state.selectedDamage = item;
      showDamageHighlight(item);
      renderDamage();
      showModal('damage-modal');
      toast(target ? '已记录人工框选范围，可运行此项修复。' : '已添加手动缺损；点“运行修复”或“运行全部”即可开始。');
    } catch (error) {
      if (state.image?.id !== imageId) return;
      notifyError(error);
      clearDamageHighlight();
      if (target) showModal('damage-modal');
    }
  }

  function rectBBox(drawing) {
    const x0 = Math.min(drawing.start[0], drawing.end[0]) / canvas.width * 1000;
    const y0 = Math.min(drawing.start[1], drawing.end[1]) / canvas.height * 1000;
    const x1 = Math.max(drawing.start[0], drawing.end[0]) / canvas.width * 1000;
    const y1 = Math.max(drawing.start[1], drawing.end[1]) / canvas.height * 1000;
    return [x0, y0, x1, y1].map(value => Math.round(value));
  }

  function updateRectDraft() {
    const drawing = state.rectDraw;
    if (!drawing || !canvas.width || !canvas.height) return;
    const node = $('#damage-highlight');
    if (!node) return;
    Object.assign(node.style, bboxPercent(rectBBox(drawing)));
    node.classList.add('visible');
  }

  async function finishRectDraw() {
    const drawing = state.rectDraw;
    if (!drawing || !state.image) return;
    const imageId = state.image.id;
    state.rectDraw = null;
    const [x0, y0, x1, y1] = rectBBox(drawing);
    if (x1 - x0 < 4 || y1 - y0 < 4) {
      clearDamageHighlight();
      toast('框选太小，请重新拖动一个更大的红色矩形。');
      return;
    }
    try {
      const { defect } = await invoke('addManualDefect', imageId, JSON.stringify({
        bbox: [x0, y0, x1, y1], type: 'manual', size: 'manual',
      }));
      if (state.image?.id !== imageId) return;
      state.damageItems.push(defect);
      state.selectedDamage = defect;
      showDamageHighlight(defect);
      renderDamage();
      toast('已添加红色矩形选区；点“一键确认并生成修复范围”或直接生成修复候选。');
    } catch (error) {
      if (state.image?.id !== imageId) return;
      clearDamageHighlight();
      notifyError(error);
    }
  }

  function renderReferences(records) {
    state.references = records || [];
    $('#reference-list').innerHTML = state.references.map(item => `<button class="reference-item ${item.id === state.selectedReference ? 'selected' : ''}" data-reference="${item.id}">${escapeHtml(item.name)}<small>${escapeHtml(item.metadata?.group || '')} · ${escapeHtml(item.metadata?.relation || '')} · 相似度 ${item.retrieval_score}</small></button>`).join('') || '<div class="empty-inline">没有匹配参考图</div>';
    $$('.reference-item').forEach(item => item.addEventListener('click', () => {
      state.selectedReference = item.dataset.reference; renderReferences(state.references);
      $('[data-action="register-reference"]').disabled = !state.image;
    }));
  }

  function escapeHtml(text) { return String(text ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }

  function showModal(id) { $(`#${id}`).classList.remove('hidden'); }
  function hideModal(id) { $(`#${id}`).classList.add('hidden'); }

  async function saveReferenceForm() {
    const payload = { source: $('#ref-source').value.trim(), license: $('#ref-license').value.trim(),
      group: $('#ref-group').value.trim(), relation: $('#ref-relation').value };
    if (!payload.source || !payload.license || !payload.group) {
      $('#ref-error').textContent = '来源、许可依据与壁画分组均需填写。';
      return;
    }
    $('#ref-error').textContent = '';
    try {
      const result = await invoke('addReference', JSON.stringify(payload));
      if (!result) return hideModal('reference-modal');
      hideModal('reference-modal'); toast('参考资料已导入。'); await refresh();
    } catch (error) { $('#ref-error').textContent = error.message; }
  }

  async function saveVerifyNote() {
    if (!state.registration) return hideModal('verify-modal');
    const note = $('#verify-note').value.trim();
    if (!note) { $('#verify-error').textContent = '请填写对应关系与核验依据。'; return; }
    try {
      await invoke('approveReference', state.registration.id, exportMaskBase64(), note);
      hideModal('verify-modal'); toast('应用区域已核验并记录。'); await refresh();
    } catch (error) { $('#verify-error').textContent = error.message; }
  }

  async function handleAction(action) {
    try {
      switch (action) {
        case 'new-task':
          if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再新建任务。');
          await loadImage(null); state.image = null; $('#empty-state').classList.remove('hidden'); imageWrap.classList.add('hidden'); break;
        case 'open-image': await openImage(); break;
        case 'open-damage-list': openDamageModal(); break;
        case 'close-damage-list': hideModal('damage-modal'); break;
        case 'add-manual-defect': beginManualDamage(); break;
        case 'open-triple-view': openTripleView(); break;
        case 'close-triple-view': hideModal('triple-modal'); break;
        case 'generate-large-panel': await generateTriple('large'); break;
        case 'generate-full-panel': await generateTriple('full'); break;
        case 'load-demo': { if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再载入图像。'); const record = await invoke('loadDemo'); await loadImage(record); break; }
        case 'undo': {
          if (state.currentStroke) break;
          const last = state.strokeOrder.pop();
          if (last === 'region') state.regionStrokes.pop();
          else if (last === 'mask') state.maskStrokes.pop();
          if (last) redrawMask();
          break;
        }
        case 'clear-mask': state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.baseMask = null; redrawMask(); break;
        case 'import-mask': { const result = await invoke('importMask'); if (result?.mask_data) await showMaskFromBase64(result.mask_data); break; }
        case 'save-defect-mask': await saveMaskToDefect(); break;
        case 'zoom-in': state.zoom = Math.min(3, state.zoom * 1.15); updateCanvasSize(); break;
        case 'zoom-out': state.zoom = Math.max(.5, state.zoom / 1.15); updateCanvasSize(); break;
        case 'zoom-fit': state.zoom = 1; updateCanvasSize(); break;
        case 'run-repair': await createJob(); break;
        case 'auto-restore': {
          if (!state.image) return toast('请先打开一幅壁画。');
          if (state.running || state.autoBusy) return toast('已有任务或 AI 分析在执行，请稍候。');
          const imageId = state.image.id;
          const provider = $('#vision-provider').value;
          const setup = state.lastSnapshot?.models?.local_detection_setup || {};
          if (provider === 'local' && setup.qwen_ready === false) {
            // Missing local weights: explain and stay in the manual workflow
            // instead of silently sending the image to a cloud provider.
            const hint = setup.qwen_hint || '本地 Qwen3-VL 未安装。';
            $('#vision-provider-note').textContent = `${hint} 仍可手动圈选缺损区域。`;
            setAutoStatus(`${hint} 仍可手动圈选缺损区域。`);
            toast('本地 Qwen3-VL 未安装；可继续手动圈选缺损区域。');
            return;
          }
          state.autoBusy = true;
          $('[data-action="run-repair"]').disabled = true;
          $('[data-action="auto-restore"]').disabled = true;
          $('#vision-provider').disabled = true;
          renderDamage();
          setAutoStatus(provider === 'deepseek' ? 'DeepSeek 正在分析图像…' : '本地 Qwen3-VL 正在分析图像…');
          try {
            const payload = JSON.stringify({ image_id: imageId, workflow: 'professional_step', provider });
            await invoke('autoRestore', payload);
          } catch (error) {
            if (state.image?.id !== imageId) break;
            state.autoBusy = false;
            setAutoStatus(`AI 缺损分析失败：${error.message || error} 可切换来源或手动圈选。`);
            notifyError(error);
            await refresh();
          }
          break;
        }
        case 'auto-confirm-all': await autoConfirmAll(); break;
        case 'run-all-defects': await runDefectRepairs(state.damageItems.map((_, index) => index)); break;
        case 'generate-region-repairs': {
          const ids = state.damageItems
            .filter(item => item.proposal_id && item.mask_file_url && item.confirmed !== false)
            .map(item => item.proposal_id);
          await generateRegionRepairs(ids);
          break;
        }
        case 'regenerate-region': {
          if (!state.job?.defect_proposal_id) return toast('当前任务不是区域修复任务，无法按区域重新生成。');
          await generateRegionRepairs([state.job.defect_proposal_id]);
          break;
        }
        case 'assess-repair': await assessRepair(); break;
        case 'review-expert': await reviewCandidateAction('expert_review'); break;
        case 'review-preview': await reviewCandidateAction('experimental_preview'); break;
        case 'export-observation-card':
          await exportPdf('exportObservationCard', { image_id: state.image?.id });
          break;
        case 'export-repair-report':
          await exportPdf('exportRepairReport', { job_id: state.job?.id });
          break;
        case 'compose-final': await composeFinalRepair(); break;
        case 'save-final': await saveFinalRepair(); break;
        case 'open-demo-guide': await openDemoGuide(); break;
        case 'close-demo-guide': hideModal('demo-modal'); break;
        case 'cancel-job': { const identifier = state.activeJobId || state.job?.id; if (identifier) { await invoke('cancelJob', identifier); } await refresh(); break; }
        case 'resume-job': if (state.job) await invoke('resumeJob', state.job.id); await refresh(); break;
        case 'select-candidate': {
          if (state.job && state.candidate) {
          const decision = selectedDecision();
          if (isProfessional() && !decision) return toast('专业分步：请先选择采纳 / 修改后采纳 / 驳回。');
          const notePayload = JSON.stringify({
            note: $('#review-note').value,
            archive_as_hypothesis: Boolean($('#archive-hypothesis')?.checked),
            decision: decision || null,
          });
          state.job = await invoke('selectCandidate', state.job.id, state.candidate.id, notePayload);
          toast(state.job?.selected_as === 'rejected' ? '已驳回该候选，不进入下一档。'
            : state.job?.selected_as === 'hypothesis' ? '已按假设性建议归档。'
            : state.job?.selected_as === 'enhanced' ? '已选定加强修复结果（非历史定稿）。'
            : state.job?.selected_as === 'modified' ? '已按修改后结果采纳。'
            : '候选选择已保存。');
          renderComparison(); await refresh();
          }
        } break;
        case 'export-overlay': {
          if (!state.image) return toast('请先打开一幅壁画。');
          const out = document.createElement('canvas');
          out.width = canvas.width; out.height = canvas.height;
          const octx = out.getContext('2d');
          octx.drawImage(image, 0, 0, out.width, out.height);
          octx.drawImage(canvas, 0, 0);
          if (regionCanvas) octx.drawImage(regionCanvas, 0, 0);
          octx.fillStyle = 'rgba(0,0,0,0.55)';
          octx.fillRect(0, out.height - 36, out.width, 36);
          octx.fillStyle = '#fff'; octx.font = '14px sans-serif';
          octx.fillText('琥珀=可修露底 · 红=不确定 · 深红=不可复原｜结果须人工复核', 12, out.height - 14);
          const link = document.createElement('a');
          link.download = `mural-overlay-${Date.now()}.png`;
          link.href = out.toDataURL('image/png');
          link.click();
          toast('已导出标红对照图。');
          break;
        }
        case 'save-result': if (state.candidate) {
          try { await invoke('saveResult', state.candidate.id); } catch (error) { notifyError(error); }
        } break;
        case 'export-archive': {
          if (state.job) await invoke('exportArchive', state.job.id);
          break;
        }
        case 'open-settings': openSettings(); break;
        case 'close-settings': closeSettings(); break;
        case 'save-deepseek-key': await saveDeepSeekKey(); break;
        case 'save-settings': await saveSettings(); break;
        case 'open-reference': showModal('reference-modal'); $('#ref-error').textContent = ''; break;
        case 'close-reference': hideModal('reference-modal'); break;
        case 'save-reference': await saveReferenceForm(); break;
        case 'open-verify': if (state.registration) { showModal('verify-modal'); $('#verify-error').textContent = ''; } break;
        case 'close-verify': hideModal('verify-modal'); break;
        case 'save-verify': await saveVerifyNote(); break;
        case 'propose-damage': if (state.image) await invoke('proposeDamage', state.image.id); showPage('tools'); break;
        case 'sam-point': setTool('sam-point'); state.promptPoints = []; state.promptLabels = []; showPage('workbench'); showView('edit'); toast('左键添加前景点，右键添加背景点。'); break;
        case 'sam-box': setTool('sam-box'); state.promptBox = null; showPage('workbench'); showView('edit'); toast('在图像上拖动框选目标。'); break;
        case 'run-sam': if (state.image) await invoke('runSam', state.image.id, JSON.stringify(state.promptPoints), JSON.stringify(state.promptLabels), JSON.stringify(state.promptBox)); break;
        case 'clear-prompts': state.promptPoints = []; state.promptLabels = []; state.promptBox = null; setTool('brush'); toast('提示已清除。'); break;
        case 'use-proposal': if (state.proposal) { await invoke('adoptProposal', state.proposal.id); await installMask(state.proposal.mask_file_url); state.proposalIds = [state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'merge-proposal': if (state.proposal) { const current = exportMaskBase64(); state.proposal = await invoke('mergeProposal', state.proposal.id, current); await installMask(state.proposal.mask_file_url); state.proposalIds = [...(state.proposalIds || []), state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'add-reference': showModal('reference-modal'); $('#ref-error').textContent = ''; break;
        case 'search-references': await invoke('searchReferences', $('#reference-query').value, $('#reference-group').value); break;
        case 'register-reference': if (state.selectedReference && state.image) await invoke('registerReference', state.image.id, state.selectedReference, exportMaskBase64()); break;
        case 'approve-reference': if (state.registration && state.image) { showModal('verify-modal'); $('#verify-error').textContent = ''; } break;
        case 'import-dataset': { const result = await invoke('importDataset'); if (result?.notice) toast(result.notice); break; }
        case 'export-splits': { const result = await invoke('exportSplits'); if (result?.notice) toast(result.notice); break; }
        case 'evaluate-pair': if (state.candidate) await invoke('evaluatePair', state.candidate.id); else toast('请先选择候选。'); break;
        case 'batch-jobs': { const result = await invoke('batchJobs', JSON.stringify(readSettings())); if (result?.notice) { toast(result.notice); await refresh(); } break; }
        case 'show-models': await invoke('showModels'); break;
        case 'backup-workspace': { const result = await invoke('backupWorkspace'); if (result?.path) toast('工作空间已备份。'); else if (result?.cancelled) toast('已取消备份。'); break; }
      }
    } catch (error) { notifyError(error); }
  }

  function openSettings() {
    $('#deepseek-api-key').value = '';
    $('#deepseek-key-error').textContent = '';
    renderDeepSeekKeyStatus(state.lastSnapshot?.models?.deepseek_vision);
    $('#api-kind').value = state.lastSnapshot?.api?.kind || 'responses';
    $('#api-base').value = state.lastSnapshot?.api?.base_url || 'https://api.openai.com/v1';
    $('#api-model').value = state.lastSnapshot?.api?.model || '';
    $('#api-key').value = '';
    $('#api-json').checked = state.lastSnapshot?.api?.json_mode !== false;
    $('#imagine-base').value = state.lastSnapshot?.imagine?.base_url || 'https://api.x.ai/v1';
    $('#imagine-model').value = state.lastSnapshot?.imagine?.model || 'grok-imagine-image-2.0';
    $('#imagine-key').value = '';
    $('#api-error').textContent = '';
    showModal('settings-modal');
  }

  function closeSettings() {
    $('#deepseek-api-key').value = '';
    $('#deepseek-key-error').textContent = '';
    hideModal('settings-modal');
  }

  function renderDeepSeekKeyStatus(configured) {
    $('#deepseek-key-status').textContent = configured ? '已配置' : '未配置';
  }

  async function saveDeepSeekKey() {
    const input = $('#deepseek-api-key');
    const button = $('[data-action="save-deepseek-key"]');
    const message = $('#deepseek-key-error');
    if (button.disabled) return;
    const key = input.value.trim();
    if (!key) {
      message.textContent = '请输入 DeepSeek API 密钥。';
      return;
    }
    input.disabled = button.disabled = true;
    message.textContent = '';
    try {
      const result = await invoke('saveDeepSeekKey', key);
      input.value = '';
      renderDeepSeekKeyStatus(result.configured);
      toast('DeepSeek 密钥已保存。');
      await refresh();
    } catch (error) {
      message.textContent = error.message;
    } finally {
      input.disabled = button.disabled = false;
    }
  }

  async function saveSettings() {
    const form = { kind: $('#api-kind').value, base_url: $('#api-base').value.trim(), model: $('#api-model').value.trim(), key: $('#api-key').value, json_mode: $('#api-json').checked,
      imagine: { provider: 'openai_json', base_url: $('#imagine-base').value.trim(), model: $('#imagine-model').value.trim(), key: $('#imagine-key').value } };
    try { await invoke('saveApiSettings', JSON.stringify(form)); closeSettings(); toast('API 配置已保存。'); await refresh(); }
    catch (error) { $('#api-error').textContent = error.message; }
  }

  function addReference() { showModal('reference-modal'); $('#ref-error').textContent = ''; }

  function readSettings() {
    return { backend: $('#repair-backend').value, planner: $('#planner').value, count: Number($('#candidate-number').value),
      instructions: $('#instructions').value, rounds: Number($('#revisions').value), timeout: Number($('#timeout').value),
      mode: $('#repair-mode').value };
  }

  document.addEventListener('click', event => {
    const page = event.target.closest('[data-page]'); if (page) { showPage(page.dataset.page); return; }
    const view = event.target.closest('[data-view]'); if (view) { showView(view.dataset.view); return; }
    const action = event.target.closest('[data-action]'); if (action) { handleAction(action.dataset.action); return; }
    const tool = event.target.closest('[data-tool]'); if (tool) setTool(tool.dataset.tool);
  });
  $('#brush-size').addEventListener('input', event => { state.brushSize = Number(event.target.value); $('#brush-value').value = String(state.brushSize); });
  $('#repair-mode').addEventListener('change', event => {
    const backend = $('#repair-backend');
    if (event.target.value === 'evidence') backend.value = 'reference';
    else if (event.target.value === 'speculative' || event.target.value === 'enhanced') {
      backend.value = state.lastSnapshot?.models?.diffusion ? 'diffusion' : (state.lastSnapshot?.models?.lama ? 'lama' : 'opencv');
    }
    else backend.value = state.lastSnapshot?.models?.lama ? 'lama' : 'opencv';
    $('#controlnet').disabled = event.target.value !== 'speculative' || !state.lastSnapshot?.models?.controlnet;
    if (event.target.value === 'speculative' && $('#evidence-level')) $('#evidence-level').value = 'speculative';
    if (event.target.value === 'enhanced' && $('#evidence-level')) $('#evidence-level').value = 'enhanced';
    if (event.target.value === 'conservative' && $('#evidence-level')) $('#evidence-level').value = 'conservative';
  });
  $('#professional-step')?.addEventListener('change', syncProfessionalUi);
  $('#vision-provider').addEventListener('change', () => { renderVisionProviderNote(); setAutoStatus(); });
  $('#repair-mode')?.addEventListener('change', syncProfessionalUi);
  $('#evidence-level')?.addEventListener('change', syncProfessionalUi);
  $('#force-override')?.addEventListener('change', syncProfessionalUi);
  $('#allow-generative-attempt')?.addEventListener('change', event => {
    if (event.target.checked) {
      $('#repair-mode').value = 'speculative';
      $('#evidence-level').value = 'speculative';
      $('#risk-ack').checked = true;
      toast('已启用 AI 尝试性补全：将用扩散模型生成，结果须按假设归档。');
    }
    syncProfessionalUi();
  });
  syncProfessionalUi();
  $('#professional-step').checked = true;
  $('#planner').addEventListener('change', event => {
    $('#planner-note').textContent = event.target.value === 'api' ? '会向已配置服务发送图像缩略图、选区预览和备注。' : event.target.value === 'local' ? '本地 Qwen3-VL 分析图像，不发送数据。' : '本地规则不会发送图像。';
  });
  $('#comparison-mode').addEventListener('change', renderComparison);
  $('#history-filter').addEventListener('input', () => renderHistory());
  $('#source-image').addEventListener('load', updateCanvasSize);
  new ResizeObserver(updateCanvasSize).observe($('#image-stage'));
  new ResizeObserver(() => layoutComparison()).observe($('#comparison-stage'));

  canvas.addEventListener('pointerdown', event => {
    if (!state.image || !image.naturalWidth || state.running) return;
    const point = pointAt(event);
    try { canvas.setPointerCapture(event.pointerId); } catch (error) { /* synthetic/edge pointer */ }
    if (state.damageDrawMode) {
      event.preventDefault();
      state.damageDraw = { start: point, end: point };
      updateDamageDraft();
      return;
    }
    if (state.tool === 'rect') {
      event.preventDefault();
      state.rectDraw = { start: point, end: point };
      updateRectDraft();
      return;
    }
    if (state.tool === 'sam-point') {
      state.promptPoints.push(point); state.promptLabels.push(event.button === 2 ? 0 : 1); toast('已记录分割提示点。'); return;
    }
    if (state.tool === 'sam-box') { state.promptBox = [point[0], point[1], point[0], point[1]]; return; }
    if (event.button !== 0) return;
    state.currentStroke = { tool: state.tool, size: state.brushSize * canvas.width / canvas.clientWidth, points: [point] };
    $('#mask-gate-confirmed').checked = false;
    paintStroke(state.currentStroke); state.maskDirty = true; setBusy(state.running);
  });
  canvas.addEventListener('pointermove', event => {
    if (state.rectDraw) {
      state.rectDraw.end = pointAt(event);
      updateRectDraft();
      return;
    }
    if (state.damageDraw) {
      state.damageDraw.end = pointAt(event);
      updateDamageDraft();
      return;
    }
    if (state.currentStroke) {
      state.currentStroke.points.push(pointAt(event));
      redrawMask();
    }
    if (state.tool === 'sam-box' && state.promptBox && event.buttons) state.promptBox.splice(2, 2, ...pointAt(event));
  });
  canvas.addEventListener('pointerup', () => {
    if (state.rectDraw) {
      finishRectDraw().catch(notifyError);
      return;
    }
    if (state.damageDraw) {
      finishDamageDraw().catch(notifyError);
      return;
    }
    if (!state.currentStroke) return;
    if (isRegionTool(state.currentStroke.tool)) {
      state.regionStrokes.push(state.currentStroke); state.strokeOrder.push('region');
    } else {
      state.maskStrokes.push(state.currentStroke); state.strokeOrder.push('mask');
    }
    state.currentStroke = null;
    redrawMask();
  });
  canvas.addEventListener('contextmenu', event => event.preventDefault());
  $('#image-stage').addEventListener('wheel', event => {
    if (event.ctrlKey) { event.preventDefault(); state.zoom = Math.max(.5, Math.min(3, state.zoom * (event.deltaY < 0 ? 1.1 : .9))); updateCanvasSize(); }
  }, { passive: false });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !$('#settings-modal').classList.contains('hidden')) {
      closeSettings();
      return;
    }
    if (event.target.matches('input,textarea,select')) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') { event.preventDefault(); openImage(); return; }
    if (event.key.toLowerCase() === 'b') setTool('brush');
    if (event.key.toLowerCase() === 'r') setTool('rect');
    if (event.key.toLowerCase() === 'e') setTool('eraser');
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); handleAction('undo'); }
    if (event.key === 'Escape') {
      if (state.rectDraw) {
        state.rectDraw = null;
        clearDamageHighlight();
        toast('已取消矩形圈选。');
        return;
      }
      if (state.damageDrawMode) {
        state.damageDrawMode = false; state.damageDraw = null; state.damageDrawTarget = null;
        $('#canvas-wrap').classList.remove('damage-drawing');
        clearDamageHighlight();
        toast('已取消手动框选。');
        return;
      }
      ['reference-modal', 'verify-modal', 'damage-modal'].forEach(hideModal);
    }
  });

  new QWebChannel(qt.webChannelTransport, channel => {
    state.bridge = channel.objects.bridge;
    state.bridge.notify.connect(message => toast(message));
    state.bridge.auxCompleted.connect(() => refresh());
    refresh(); setInterval(refresh, 900);
  });

  // Test seam: the acceptance tests drive the same mask round-trip and job flow
  // through the shipped frontend instead of duplicating the logic in Python.
  window.__shibui = { state, loadImage, installMask, showMaskFromBase64, exportMaskBase64, refresh, renderDamage,
    renderJob, renderRegionPanel, renderAiReview, renderComparison, layoutComparison };
})();
