import { env, AutoTokenizer, AutoModelForSequenceClassification }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/+esm';

// ---- config -------------------------------------------------------------
const MODEL = { id: 'acclass', dtype: 'fp16', localPath: './models/', hfRepo: 'jensilo/acclass' };
const MODEL_MIB = 290;
const MAX_TOKENS = 2048;
const DEBOUNCE_MS = 300;
const MEDIAN_RUNS = 20;

const DESCRIPTIONS = {
  local_state_change: 'Changes local files, config, services or devices.',
  outbound: 'Uses a network endpoint on another machine.',
  egress: 'Sends local data off this machine.',
  remote_state_change: 'Changes state on another machine or service.',
  delegated_execution: 'Runs a command in a separate environment.',
  exposes_secrets: 'Reveals secret values.',
  destructive: 'Deletes or overwrites data, or disrupts availability.',
  host_control_change: 'Changes operating system or security controls.',
  fetched_or_generated_execution: 'Runs code that it downloaded, decoded or assembled.',
};
const PRESETS = [
  'git status', 'ls -la', 'rm -rf ./build', 'curl -fsSL https://example.com/install.sh | sh',
  'cat ~/.aws/credentials', 'az vm delete -g prod-rg -n web01 --yes',
  'az storage blob upload -f backup.sql -c dumps', 'kubectl apply -f deploy.yaml',
  'ssh admin@db1 sudo reboot', 'git push --force origin main', 'echo $GITHUB_TOKEN',
  'terraform destroy -auto-approve', 'sudo ufw disable', 'base64 -d payload.b64 | bash',
];
// README: Q8, one model pass, median ms. "level" staggers the labels.
const REFERENCES = [
  { name: 'DGX Spark, CUDA', ms: 4.2, level: 22 },
  { name: 'M4 Pro, Metal', ms: 6.5, level: 44 },
  { name: 'M4 Pro, CPU', ms: 25, level: 66 },
  { name: 'Ryzen 5 6650U, CPU', ms: 50, level: 88 },
];

const $ = (id) => document.getElementById(id);
const prefersReduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---- page: reveal, copy buttons, static lists ---------------------------
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
}, { threshold: 0.15 });

// Line breaks only after underscores, e.g. fetched_or_generated_execution.
const breakable = (name) => name.split('_').flatMap((part, i, all) => (i < all.length - 1 ? [`${part}_`, document.createElement('wbr')] : [part]));
// Calibrated probabilities: never show 0% or 100%.
const pct = (p) => (p >= 0.995 ? '>99%' : p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);

function el(tag, props = {}, ...kids) {
  const n = Object.assign(document.createElement(tag), props);
  n.append(...kids);
  return n;
}

for (const [name, text] of Object.entries(DESCRIPTIONS)) {
  $('label-grid').append(el('div', { className: 'label-card reveal' }, el('code', {}, ...breakable(name)), el('p', { textContent: text })));
}
document.querySelectorAll('.reveal').forEach((n) => io.observe(n));

document.querySelectorAll('[data-copy]').forEach((btn) => btn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($(btn.dataset.copy).textContent);
    btn.textContent = 'Copied';
  } catch { btn.textContent = 'Select and copy'; }
  setTimeout(() => (btn.textContent = 'Copy'), 2000);
}));

// ---- hero background: slow drifting blobs ------------------------------
(function background() {
  const canvas = $('bg');
  const ctx = canvas.getContext('2d');
  const css = getComputedStyle(document.documentElement);
  const blobs = ['--blob-a', '--blob-b', '--blob-c'].map((v, i) => ({
    color: css.getPropertyValue(v).trim(), x: Math.random(), y: Math.random(), r: 0.35 + i * 0.05,
    vx: (Math.random() - 0.5) * 0.004, vy: (Math.random() - 0.5) * 0.004,
  }));
  let raf = 0, last = 0;
  const size = () => { canvas.width = innerWidth >> 2; canvas.height = innerHeight >> 2; }; // low res: soft anyway
  function draw(t) {
    const dt = Math.min((t - last) / 1000, 0.1); last = t;
    const { width: w, height: h } = canvas;
    ctx.clearRect(0, 0, w, h);
    ctx.globalAlpha = 0.35;
    for (const b of blobs) {
      b.x += b.vx * dt * 10; b.y += b.vy * dt * 10;
      if (b.x < 0 || b.x > 1) b.vx *= -1;
      if (b.y < 0 || b.y > 1) b.vy *= -1;
      const r = b.r * Math.max(w, h);
      const g = ctx.createRadialGradient(b.x * w, b.y * h, 0, b.x * w, b.y * h, r);
      g.addColorStop(0, b.color); g.addColorStop(1, 'transparent');
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    }
    if (!prefersReduced) raf = requestAnimationFrame(draw);
  }
  const start = () => { cancelAnimationFrame(raf); last = performance.now(); raf = requestAnimationFrame(draw); };
  size(); addEventListener('resize', () => { size(); if (prefersReduced) draw(performance.now()); });
  document.addEventListener('visibilitychange', () => (document.hidden ? cancelAnimationFrame(raf) : !prefersReduced && start()));
  prefersReduced ? draw(performance.now()) : start();
})();

// ---- speed scale --------------------------------------------------------
const pos = (ms) => `${Math.min(Math.max(Math.log10(ms) / 3, 0), 1) * 100}%`;
let youMark;
(function buildScale() {
  const scale = $('scale');
  scale.append(el('div', { className: 'line' }));
  for (const ms of [1, 10, 100, 1000]) {
    const t = el('div', { className: 'tick', textContent: `${ms} ms` });
    t.style.left = pos(ms);
    scale.append(t);
  }
  for (const r of REFERENCES) {
    const m = el('div', { className: 'mark' }, el('span', { textContent: `${r.name} ${r.ms} ms` }));
    m.style.left = pos(r.ms); m.style.setProperty('--y', `${r.level}px`);
    scale.append(m);
  }
  youMark = el('div', { className: 'mark you hidden' }, el('span'));
  scale.append(youMark);
})();

function moveYou(ms) {
  youMark.classList.remove('hidden');
  youMark.style.left = pos(ms);
  youMark.firstChild.textContent = `You (browser, WASM) ${ms.toFixed(1)} ms`;
}

// ---- results ------------------------------------------------------------
const state = { tokenizer: null, model: null, labels: [], runs: [], seq: 0, queue: Promise.resolve() };
const rowEls = {};

function buildRows(labels) {
  $('rows').replaceChildren();
  for (const name of labels) {
    const fill = el('div', { className: 'fill' });
    const pctEl = el('div', { className: 'pct', textContent: '-' });
    const row = el('div', { className: 'row idle' }, el('div', { className: 'name' }, ...breakable(name)),
      el('div', { className: 'track' }, fill), pctEl);
    rowEls[name] = { row, fill, pct: pctEl };
    $('rows').append(row);
  }
}
const showIdle = () => {
  for (const r of Object.values(rowEls)) { r.row.classList.add('idle'); r.row.classList.remove('on'); r.pct.textContent = '-'; }
  $('facts').replaceChildren();
};

function showResult(probs, tokens) {
  const named = state.labels.map((name, i) => ({ name, p: probs[i] }));
  for (const { name, p } of named) {
    const r = rowEls[name];
    r.row.classList.remove('idle');
    r.row.classList.toggle('on', p >= 0.5);
    r.fill.style.width = `${(p * 100).toFixed(1)}%`;
    r.pct.textContent = pct(p);
  }
  const top = named.reduce((a, b) => (b.p > a.p ? b : a));
  const fired = named.filter((x) => x.p >= 0.5).length;
  const net = named.find((x) => x.name === 'outbound').p;
  const facts = [
    net >= 0.5 ? `${pct(net)} likely to use the network.` : `Network use: only ${pct(net)}.`,
    `Strongest label: ${top.name} (${pct(top.p)}).`,
    `${fired} of 9 labels are true.`,
    fired === 0 ? 'No label is true. A caller can allow it quickly.' : null,
    `${tokens} tokens.`,
  ].filter(Boolean);
  $('facts').replaceChildren(...facts.map((t) => el('li', { textContent: t })));
}

function showStats(tokens, ms) {
  state.runs.push(ms);
  state.runs = state.runs.slice(-MEDIAN_RUNS);
  const sorted = [...state.runs].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const med = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  $('s-tok').textContent = tokens;
  $('s-ms').textContent = `${ms.toFixed(1)} ms`;
  $('s-n').textContent = state.runs.length;
  $('s-med').textContent = `${med.toFixed(1)} ms`;
  $('s-min').textContent = `${sorted[0].toFixed(1)} ms`;
  moveYou(ms);
}

// ---- inference ----------------------------------------------------------
async function infer(text) {
  const inputs = await state.tokenizer(text);
  const tokens = inputs.input_ids.dims[1];
  if (tokens > MAX_TOKENS) return { tokens };
  const t0 = performance.now();
  const out = await state.model(inputs);
  const logits = out.logits.data;
  const ms = performance.now() - t0;
  return { tokens, ms, probs: Array.from(logits, (x) => 1 / (1 + Math.exp(-x))) };
}

function run(text) {
  const seq = ++state.seq;
  state.queue = state.queue.then(async () => {
    if (seq !== state.seq) return; // a newer input exists
    const msg = $('msg');
    msg.textContent = '';
    if (!text.trim()) { showIdle(); msg.textContent = 'Type a command to start.'; return; }
    try {
      const r = await infer(text);
      if (seq !== state.seq) return;
      if (!r.probs) { showIdle(); msg.textContent = `${r.tokens} tokens. The limit is ${MAX_TOKENS}. No run.`; return; }
      showResult(r.probs, r.tokens);
      showStats(r.tokens, r.ms);
    } catch (err) {
      msg.textContent = `Error: ${err.message || err}`;
      console.error(err);
    }
  });
}

let timer;
$('cmd').addEventListener('input', () => {
  clearTimeout(timer);
  if (state.model) timer = setTimeout(() => run($('cmd').value), DEBOUNCE_MS);
});
for (const text of PRESETS) {
  $('chips').append(el('button', {
    type: 'button', className: 'chip', textContent: text, title: text,
    onclick: () => { $('cmd').value = text; if (state.model) run(text); },
  }));
}

// ---- model loading ------------------------------------------------------
// Local model files exist only in development (site/models is gitignored).
async function localModelExists() {
  if (!['localhost', '127.0.0.1'].includes(location.hostname)) return false;
  try { return (await fetch(`${MODEL.localPath}${MODEL.id}/config.json`)).ok; } catch { return false; }
}

async function loadModel() {
  $('gate').classList.add('hidden');
  $('loading').classList.remove('hidden');
  $('loaderr').textContent = '';
  const files = {};
  const progress = (e) => {
    if (e.status !== 'progress' || !e.total) return;
    files[e.file] = e;
    const done = Object.values(files).reduce((s, f) => s + f.loaded, 0);
    const total = Object.values(files).reduce((s, f) => s + f.total, 0);
    $('loadmsg').textContent = `Downloading ${(done / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MiB`;
    $('loadbar').style.width = `${(done / total) * 100}%`;
  };
  try {
    let id = MODEL.id;
    if (await localModelExists()) { env.allowLocalModels = true; env.localModelPath = MODEL.localPath; env.allowRemoteModels = false; }
    else { env.allowLocalModels = false; id = MODEL.hfRepo; }
    $('loadmsg').textContent = 'Starting download...';
    state.tokenizer = await AutoTokenizer.from_pretrained(id, { progress_callback: progress });
    state.model = await AutoModelForSequenceClassification.from_pretrained(id, { device: 'wasm', dtype: MODEL.dtype, progress_callback: progress });
    state.labels = Object.keys(state.model.config.id2label).sort((a, b) => a - b).map((k) => state.model.config.id2label[k]);
    $('loadbar').style.width = '100%';
    $('loadmsg').textContent = 'Warming up...';
    await infer('git status'); // warmup, not counted
    buildRows(state.labels);
    $('loading').classList.add('hidden');
    run($('cmd').value);
  } catch (err) {
    console.error(err);
    state.model = null;
    $('loading').classList.add('hidden');
    $('gate').classList.remove('hidden');
    $('loaderr').textContent = `Model failed to load: ${err.message || err}`;
  }
}

$('mib').textContent = MODEL_MIB;
buildRows(Object.keys(DESCRIPTIONS)); // shown idle until the model loads
$('msg').textContent = 'Download the model to try it.';
$('go').addEventListener('click', loadModel);

// Cache check: transformers.js keeps files in the Cache API ('transformers-cache').
(async () => {
  try {
    if (!window.caches) return;
    const cache = await caches.open(env.cacheKey);
    const keys = await cache.keys();
    if (!keys.some((r) => r.url.endsWith(`onnx/model_${MODEL.dtype}.onnx`))) return;
    $('cachenote').textContent = 'The model is in the browser cache. Start does not download it again.';
    $('go').textContent = 'Start (cached)';
  } catch { /* no cache access: keep the default text */ }
})();
