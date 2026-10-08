// EmbeddingGemma 2 on VENTUNO Q: browser side.
const $ = (id) => document.getElementById(id);
const ws = new WebUI();
let state = {};

async function api(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

function toast(msg, err = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = err ? 'muted err' : 'muted';
}

async function run(btn, fn) {
  btn.disabled = true;
  try { await fn(); } catch (e) { toast(e.message, true); } finally { btn.disabled = false; }
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// Status ------------------------------------------------------------------------------
function renderStatus(s) {
  state = s;
  const st = $('status');
  st.textContent = s.ready ? `Ready · ${s.dim}-d` : s.status;
  st.className = s.ready ? 'pill ready' : 'pill';
  if (document.activeElement !== $('labels')) $('labels').value = s.labels.join(', ');
  if (document.activeElement !== $('prompt')) $('prompt').value = s.prompt;
  $('ex-label').innerHTML = s.labels.map((l) => `<option>${esc(l)}</option>`).join('');
  $('live').checked = s.live;
  if (s.camera && !$('cam').src) $('cam').src = '/api/camera';
  $('cam').hidden = !s.camera;
  $('nocam').hidden = s.camera;
  const p = s.project;
  $('ei-connected').hidden = !p;
  $('ei-disconnected').hidden = !!p;
  if (p) {
    $('ei-link').textContent = `${p.name} (#${p.id})`;
    $('ei-link').href = p.studio_url;
    $('ei-labels').textContent = p.labels.length ? `labels: ${p.labels.join(', ')}` : '';
  }
  renderIndex(s.index);
}

function renderIndex(ix) {
  if (!ix) return;
  const pr = $('index-progress');
  pr.hidden = !ix.running;
  pr.max = ix.total || 1;
  pr.value = ix.done;
  $('index').textContent = ix.running ? 'Stop' : ix.indexed ? 'Update embeddings' : 'Embed project images';
  const counts = Object.entries(ix.labels || {}).map(([l, n]) => `${l || 'unlabelled'}: ${n}`).join(', ');
  $('index-status').textContent = ix.error ? `Error: ${ix.error}`
    : ix.running ? `Embedding ${ix.done} / ${ix.total}`
    : ix.indexed ? `${ix.indexed} images embedded (${counts})` : '';
}

ws.on_message('status', renderStatus);
ws.on_message('index', renderIndex);
ws.on_connect(() => api('GET', '/api/status').then(renderStatus).catch(() => {}));

// Live classification -----------------------------------------------------------------
ws.on_message('classification', (c) => {
  $('results').innerHTML = c.results.slice(0, 6).map((r) => {
    const n = c.examples[r.label] ? ` <span class="muted">· ${c.examples[r.label]} ex</span>` : '';
    return `<li><div class="bar"><i style="width:${(100 * r.probability).toFixed(1)}%"></i>` +
      `<span>${esc(r.label)}${n}</span></div><small>${(100 * r.probability).toFixed(0)}% · ${r.score.toFixed(3)}</small></li>`;
  }).join('');
  if (c.embed_ms) $('latency').textContent = `${c.embed_ms} ms per frame embedding`;
});

$('live').onchange = (e) => api('POST', '/api/live', { live: e.target.checked }).then(renderStatus);

$('file').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const data = await new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(f); });
  $('cam').src = data;
  $('cam').hidden = false;
  $('nocam').hidden = true;
  if (state.live && state.camera) await api('POST', '/api/live', { live: false }).then(renderStatus);
  try { await api('POST', '/api/classify', { image: data }); toast(''); } catch (err) { toast(err.message, true); }
};

// Labels and examples -----------------------------------------------------------------
$('save-labels').onclick = (e) => run(e.target, async () => {
  renderStatus(await api('POST', '/api/labels', { labels: $('labels').value, prompt: $('prompt').value }));
  toast('Labels updated');
});
$('add-ex').onclick = (e) => run(e.target, async () => {
  await api('POST', '/api/examples', { label: $('ex-label').value });
  toast(`Added an example of "${$('ex-label').value}"`);
});
$('clear-ex').onclick = (e) => run(e.target, async () => { await api('DELETE', '/api/examples'); toast('Examples cleared'); });

// Edge Impulse ------------------------------------------------------------------------
$('ei-connect').onclick = (e) => run(e.target, async () => {
  renderStatus(await api('POST', '/api/ei/connect', { api_key: $('ei-key').value }));
  $('ei-key').value = '';
  toast('Connected');
});
$('upload').onclick = (e) => run(e.target, async () => {
  const r = await api('POST', '/api/ei/upload', { label: $('up-label').value, category: $('up-cat').value });
  toast(`Uploaded to ${r.category} as "${r.label}"`);
});
$('index').onclick = (e) => run(e.target, async () => {
  renderIndex(await api(state.index?.running ? 'DELETE' : 'POST', '/api/ei/index'));
  state.index = { ...state.index, running: !state.index?.running };
  poll();
});
$('learn').onclick = (e) => run(e.target, async () => {
  renderStatus(await api('POST', '/api/learn-from-project'));
  toast('Labels and examples now come from the project training set');
});

async function poll() {
  const ix = await api('GET', '/api/ei/index');
  renderIndex(ix);
  state.index = ix;
  if (ix.running) setTimeout(poll, 1000);
}

function showOut(title, html) {
  $('out-card').hidden = false;
  $('out-title').textContent = title;
  $('out').innerHTML = html || '<p class="muted">Nothing to show.</p>';
  $('out-card').scrollIntoView({ behavior: 'smooth' });
}

const thumb = (id) => `<img loading="lazy" src="/api/ei/thumb/${id}" alt="Edge Impulse sample ${id}">`;

$('curate').onclick = (e) => run(e.target, async () => {
  const r = await api('POST', '/api/ei/curate', {});
  showOut(`Label issues: ${r.suggestions.length} of ${r.checked} samples`, r.suggestions.map((s) =>
    `<div class="tile">${thumb(s.id)}<div>${esc(s.filename)}<br>` +
    `${esc(s.category)} · ${esc(s.kind)}<br><b>${esc(s.label || 'unlabelled')}</b> → <span class="sug">${esc(s.suggested)}</span><br>` +
    `<span class="muted">margin ${s.margin.toFixed(3)}</span><br>` +
    `<button data-id="${s.id}" data-label="${esc(s.suggested)}">Relabel as ${esc(s.suggested)}</button></div></div>`).join(''));
});

$('out').onclick = (e) => {
  const b = e.target.closest('button[data-id]');
  if (!b) return;
  run(b, async () => {
    await api('POST', '/api/ei/relabel', { id: +b.dataset.id, label: b.dataset.label });
    b.textContent = `Relabelled`;
    toast(`Sample ${b.dataset.id} is now "${b.dataset.label}"`);
  });
};

$('search').onclick = (e) => run(e.target, async () => {
  const r = await api('POST', '/api/search', { query: $('q').value, k: 12 });
  showOut(`Closest to ${r.query}`, r.results.map((s) =>
    `<div class="tile"><a href="${s.studio_url}" target="_blank" rel="noopener">${thumb(s.id)}</a><div>` +
    `${esc(s.label || 'unlabelled')} · ${esc(s.category)}<br><span class="muted">${s.score.toFixed(3)}</span></div></div>`).join(''));
});
$('q').onkeydown = (e) => { if (e.key === 'Enter') $('search').click(); };
