// StationTruth — forecast page: the blend forecast for one airport (data/forecast/<ID>.json, docs/FORECAST_API.md)
import {
  MODEL, MODELS, modelName, modelText, METARPARSE, isNum, esc, fmt, fmtInt, getJSON, loadMeta, setStatus, cssVar, modelColour,
  onThemeChange, axisStyle, observeWidth, seg, legendSwatch, initHeader, runLabel, ago,
} from './common.js?v=__BUILD__';

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { return localStorage.getItem(`st.fc.${k}`) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`st.fc.${k}`, v); } catch { /* private mode / blocked storage */ } },
};
const pick = (v, allowed, d) => (allowed.includes(v) ? v : d);
const narrowMQ = matchMedia('(max-width: 720px)');
const QUICK = ['EGLL', 'KJFK', 'LKPR', 'EDDF', 'LFPG', 'RJTT', 'YSSY', 'KLAX', 'OMDB', 'SBGR', 'FAOR', 'CYYZ'];
// comparison order: corrected models (registry order), then raw ones, then anything unknown
const modelOrder = (ids) => {
  const known = MODELS.map((m) => m.id).filter((id) => id !== 'blend');
  const ord = [...known.filter((id) => id.endsWith('_bc')), ...known.filter((id) => !id.endsWith('_bc'))];
  return [...ord.filter((id) => ids.includes(id)), ...ids.filter((id) => !ord.includes(id) && id !== 'blend')];
};
const label = (m) => (MODEL[m] ? modelText(m) : m);
const nameHTML = (m) => (MODEL[m] ? modelName(m) : esc(m));

const state = {
  meta: null,
  stations: [],
  byId: new Map(),
  id: null,
  doc: null, // decoded forecast file
  tu: pick(store.get('tu'), ['c', 'f'], 'c'),
  wu: pick(store.get('wu'), ['kt', 'kmh', 'ms'], 'kt'),
  tz: pick(store.get('tz'), ['utc', 'local'], 'utc'),
  cmp: new Set(JSON.parse(store.get('cmp', '[]') || '[]').filter((m) => typeof m === 'string')),
  charts: [],
  ro: null,
  cursorIdx: 0,
};

/* ------------------------------------------------------------------ units */
const T_UNIT = { c: '°C', f: '°F' };
const W_UNIT = { kt: 'kt', kmh: 'km/h', ms: 'm/s' };
const cT = (v) => (isNum(v) ? (state.tu === 'f' ? v * 9 / 5 + 32 : v) : null);
const cTd = (v) => (isNum(v) ? (state.tu === 'f' ? v * 9 / 5 : v) : null); // a difference
const cW = (v) => (isNum(v) ? v * ({ kt: 1, kmh: 1.852, ms: 0.514444 }[state.wu]) : null);
const fT = (v, d = 0) => (isNum(v) ? `${fmt(cT(v), d)}°` : '–');
const fW = (v) => (isNum(v) ? fmt(cW(v), 0) : '–');

/* ------------------------------------------------------------------ time */
function parts(ts) {
  const d = new Date(ts * 1000);
  if (state.tz === 'utc') return { h: d.getUTCHours(), m: d.getUTCMinutes(), wd: d.getUTCDay(), day: d.getUTCDate(), mon: d.getUTCMonth(), key: d.toISOString().slice(0, 10) };
  return { h: d.getHours(), m: d.getMinutes(), wd: d.getDay(), day: d.getDate(), mon: d.getMonth(), key: `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}` };
}
const WD = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const hourLabel = (ts) => { const p = parts(ts); return state.tz === 'utc' ? `${String(p.h).padStart(2, '0')}Z` : `${String(p.h).padStart(2, '0')}:${String(p.m).padStart(2, '0')}`; };
const dayLabel = (ts) => { const p = parts(ts); return `${WD[p.wd]} ${p.day} ${MON[p.mon]}`; };
const tzName = () => (state.tz === 'utc' ? 'UTC' : 'your time');

/* ------------------------------------------------------------------ decode */
// Grid blocks (docs/DATA.md): value i at start + i*step; integers in 1/scale units, delta-coded; null = missing.
function decodeBlock(blk, scale) {
  if (!blk || typeof blk.start !== 'string') return null;
  const t0 = Math.round(Date.parse(blk.start) / 1000);
  const step = (blk.step || 60) * 60;
  const vals = {};
  let n = 0;
  for (const [k, a] of Object.entries(blk)) {
    if (!Array.isArray(a)) continue;
    let acc = 0;
    vals[k] = a.map((x) => (isNum(x) ? (acc += x) / scale : null));
    n = Math.max(n, a.length);
  }
  return { run: blk.run, step, ts: Array.from({ length: n }, (_, i) => t0 + i * step), vals };
}
function decodeDoc(d) {
  const scale = d.scale || 10;
  const models = {};
  for (const [m, b] of Object.entries(d.models || {})) { const x = decodeBlock(b, scale); if (x) models[m] = x; }
  return { ...d, blend: decodeBlock(d.blend, scale), models };
}

/* ------------------------------------------------------------------ sun (night shading) */
// Solar elevation (degrees), low-precision almanac formula — plenty for shading night hours.
function sunElev(lat, lon, ts) {
  const rad = Math.PI / 180;
  const d = ts / 86400 + 2440587.5 - 2451545.0;
  const g = (357.529 + 0.98560028 * d) * rad;
  const q = 280.459 + 0.98564736 * d;
  const L = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * rad;
  const e = (23.439 - 0.00000036 * d) * rad;
  const ra = Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L));
  const dec = Math.asin(Math.sin(e) * Math.sin(L));
  const gmst = (18.697374558 + 24.06570982441908 * d) % 24;
  const ha = (gmst * 15 + lon) * rad - ra;
  return Math.asin(Math.sin(lat * rad) * Math.sin(dec) + Math.cos(lat * rad) * Math.cos(dec) * Math.cos(ha)) / rad;
}
function nightSpans(lat, lon, t0, t1) {
  if (!isNum(lat) || !isNum(lon)) return [];
  const out = [];
  let start = null;
  for (let t = t0; t <= t1 + 1; t += 600) {
    const night = sunElev(lat, lon, t) < -0.833;
    if (night && start == null) start = t;
    if ((!night || t > t1) && start != null) { out.push([start, Math.min(t, t1)]); start = null; }
  }
  if (start != null) out.push([start, t1]);
  return out;
}

/* ------------------------------------------------------------------ colours */
function rgba(col, a) {
  const c = col.trim();
  const m = /^#([0-9a-f]{6})$/i.exec(c);
  if (m) { const n = parseInt(m[1], 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
  return c;
}
const COL = () => ({ t: cssVar('--fc-t'), td: cssVar('--fc-td'), wind: cssVar('--fc-wind'), p: cssVar('--fc-p'), cloud: cssVar('--fc-cloud'), night: cssVar('--fc-night'), text: cssVar('--text'), muted: cssVar('--muted') });

/* ------------------------------------------------------------------ search */
const input = $('fc-search');
const list = $('fc-list');
let results = []; let active = -1;
const norm = (t) => String(t || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
function search(q) {
  q = norm(q.trim());
  if (!q) return [];
  const up = q.toUpperCase();
  const exact = []; const a = []; const b = [];
  // IATA: from the index when present, else the ICAO = K/C/P + IATA rule (US, Canada, Alaska/Hawaii)
  if (/^[a-z0-9]{3}$/.test(q)) {
    for (const s of state.stations) if (s.iata === up) exact.push(s);
    for (const p of ['K', 'C', 'P']) { const s = state.byId.get(p + up); if (s && !exact.includes(s)) exact.push(s); }
  }
  for (const s of state.stations) {
    if (exact.includes(s)) continue;
    if (s.id.startsWith(up)) a.push(s);
    else if (norm(s.name).includes(q) || norm(s.country) === q) b.push(s);
    if (a.length > 30) break;
  }
  a.sort((x, y) => x.id.length - y.id.length || x.id.localeCompare(y.id));
  return exact.concat(a, b).slice(0, 12);
}
function renderResults() {
  const q = input.value.trim();
  if (!q) { list.hidden = true; input.setAttribute('aria-expanded', 'false'); return; }
  results = search(q);
  active = results.length ? 0 : -1;
  list.innerHTML = results.length
    ? results.map((s, i) => `<li role="option" id="fopt-${i}" data-id="${esc(s.id)}" aria-selected="${i === active}"><span></span><span class="sid">${esc(s.id)}</span><span class="sname">${esc(s.name)}</span><span class="scc">${esc(s.country)}</span></li>`).join('')
    : `<li class="empty">No airport with a current forecast matches “${esc(q)}”.</li>`;
  list.hidden = false;
  input.setAttribute('aria-expanded', 'true');
  if (active >= 0) input.setAttribute('aria-activedescendant', `fopt-${active}`); else input.removeAttribute('aria-activedescendant');
}
function setActive(i) {
  if (!results.length) return;
  active = (i + results.length) % results.length;
  list.querySelectorAll('li[role=option]').forEach((li, k) => li.setAttribute('aria-selected', String(k === active)));
  input.setAttribute('aria-activedescendant', `fopt-${active}`);
  list.querySelector(`#fopt-${active}`)?.scrollIntoView({ block: 'nearest' });
}
function choose(id) {
  list.hidden = true; input.setAttribute('aria-expanded', 'false'); input.value = ''; input.blur();
  go(id, { focus: true });
}
input.addEventListener('input', renderResults);
input.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') { e.preventDefault(); setActive(active + 1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(active - 1); }
  else if (e.key === 'Enter') {
    e.preventDefault();
    if (active >= 0) choose(results[active].id);
    else if (/^[A-Za-z0-9]{3,5}$/.test(input.value.trim())) choose(input.value.trim().toUpperCase());
  } else if (e.key === 'Escape') { input.value = ''; renderResults(); }
});
input.addEventListener('blur', () => setTimeout(() => { list.hidden = true; input.setAttribute('aria-expanded', 'false'); }, 150));
input.addEventListener('focus', () => { if (input.value.trim()) renderResults(); });
list.addEventListener('mousedown', (e) => { const li = e.target.closest('li[data-id]'); if (li) { e.preventDefault(); choose(li.dataset.id); } });
document.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName);
  if (e.key === '/' && !typing) { e.preventDefault(); input.focus(); }
});

/* ------------------------------------------------------------------ near me */
function nearest(lat, lon) {
  const rad = Math.PI / 180;
  let best = null; let bd = Infinity;
  for (const s of state.stations) {
    if (!isNum(s.lat) || !isNum(s.lon)) continue;
    const dl = (s.lat - lat) * rad; const dn = (s.lon - lon) * rad;
    const a = Math.sin(dl / 2) ** 2 + Math.cos(lat * rad) * Math.cos(s.lat * rad) * Math.sin(dn / 2) ** 2;
    if (a < bd) { bd = a; best = s; }
  }
  return best ? { s: best, km: 2 * 6371 * Math.asin(Math.sqrt(bd)) } : null;
}
$('near-btn').addEventListener('click', () => {
  const btn = $('near-btn');
  if (!navigator.geolocation) { toast('Location is not available in this browser.'); return; }
  btn.disabled = true;
  navigator.geolocation.getCurrentPosition((pos) => {
    btn.disabled = false;
    const n = nearest(pos.coords.latitude, pos.coords.longitude);
    if (!n) { toast('No airport with a forecast found.'); return; }
    toast(`Nearest: ${n.s.id} ${n.s.name} (${fmtInt(n.km)} km)`);
    go(n.s.id, { focus: true });
  }, (err) => {
    btn.disabled = false;
    toast(err.code === 1 ? 'Location permission denied.' : 'Could not get your location.');
  }, { maximumAge: 600000, timeout: 15000 });
});

/* ------------------------------------------------------------------ units controls */
seg($('tu-seg'), state.tu, (v) => { state.tu = v; store.set('tu', v); render(); });
seg($('wu-seg'), state.wu, (v) => { state.wu = v; store.set('wu', v); render(); });
seg($('tz-seg'), state.tz, (v) => { state.tz = v; store.set('tz', v); render(); });

/* ------------------------------------------------------------------ toast + copy */
let toastT;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 2600); }
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch { /* fall back */ }
  const ta = document.createElement('textarea');
  ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch { ok = false; }
  ta.remove();
  return ok;
}
const apiUrl = (id) => new URL(`data/forecast/${encodeURIComponent(id)}.json`, location.href).href;

/* ------------------------------------------------------------------ routing */
function hashId() {
  let h = location.hash.slice(1);
  try { h = decodeURIComponent(h); } catch { /* malformed escape */ }
  h = h.trim().toUpperCase();
  return /^[A-Z0-9]{3,5}$/.test(h) ? h : null;
}
function go(id, { focus = false } = {}) {
  state.focusNext = focus;
  if (hashId() === id) load(id); else location.hash = id;
}
window.addEventListener('hashchange', () => { const id = hashId(); if (id) load(id); else showWelcome(); });

let loadSeq = 0;
async function load(id) {
  // 3-letter input: resolve IATA (K/C/P + code) when the ICAO itself is unknown
  if (id.length === 3 && !state.byId.has(id)) {
    const hit = search(id)[0];
    if (hit) { history.replaceState(null, '', `#${hit.id}`); id = hit.id; }
  }
  const seq = ++loadSeq;
  state.id = id;
  document.title = `${id} forecast — StationTruth`;
  $('fc-body').innerHTML = '<div class="panel"><p class="muted" style="margin:0">Loading forecast…</p></div>';
  let raw = null; let err = null;
  // the index lists every file: skip the request (and its 404) for unknown codes
  const known = !state.stations.length || state.byId.has(id);
  if (known) {
    try { raw = await getJSON(`data/forecast/${encodeURIComponent(id)}.json`, { optional: true }); } catch (e) { err = e; }
  }
  if (seq !== loadSeq) return;
  if (!raw) {
    destroyCharts();
    const s = state.byId.get(id);
    $('fc-body').innerHTML = `<div class="panel"><p style="margin:0"><b>${esc(id)}</b>: ${err ? `couldn't load the forecast (${esc(err.message)}).` : 'no forecast. Forecasts are published for airports with an observation in the last 3 hours.'}
      ${s ? '' : ' Check the code, or search by name.'} <a href="./#${encodeURIComponent(id)}">Look it up on the map</a>.</p></div>`;
    return;
  }
  state.doc = decodeDoc(raw);
  state.rawUrl = apiUrl(id);
  document.title = `${state.doc.id} ${state.doc.name ? `${state.doc.name} ` : ''}forecast — StationTruth`;
  render();
  if (state.focusNext) { $('fc-h2')?.focus({ preventScroll: true }); $('fc-body').scrollIntoView({ block: 'start', behavior: 'smooth' }); }
  state.focusNext = false;
}

function showWelcome() {
  destroyCharts();
  state.doc = null; state.id = null;
  document.title = 'Airport forecast — StationTruth';
  const q = QUICK.filter((id) => state.byId.has(id));
  $('fc-body').innerHTML = `<div class="panel welcome">
    <p style="margin:0">Search for an airport above, or use <b>Near me</b>. Every airport with a recent METAR has a forecast${state.stations.length ? ` — ${fmtInt(state.stations.length)} right now` : ''}.</p>
    ${q.length ? `<div class="quick">${q.map((id) => `<a class="btn small" href="#${id}" title="${esc(state.byId.get(id).name)}">${id}</a>`).join('')}</div>` : ''}
  </div>`;
}

/* ------------------------------------------------------------------ render */
function render() {
  if (!state.doc) return;
  const d = state.doc;
  const b = d.blend;
  destroyCharts();
  const sub = [d.country, isNum(d.elev_m) ? `${fmtInt(d.elev_m)} m` : null, isNum(d.lat) ? `${fmt(Math.abs(d.lat), 2)}°${d.lat >= 0 ? 'N' : 'S'} ${fmt(Math.abs(d.lon), 2)}°${d.lon >= 0 ? 'E' : 'W'}` : null].filter(Boolean).join(' · ');
  const runs = modelOrder(Object.keys(d.models)).filter((m) => !m.endsWith('_bc')).map((m) => `${label(m)} ${runLabel(d.models[m].run)}`);
  const head = `<div class="fc-head">
    <div class="fc-title">
      <span class="icao">${esc(d.id)}</span>
      <h2 id="fc-h2" tabindex="-1">${esc(d.name || d.id)}</h2>
      <div class="sub">${esc(sub)}</div>
      <div class="sub">${b ? `Blend run ${esc(runLabel(b.run))}` : 'No blend forecast'} · built ${esc(ago(d.generated))}${runs.length ? ` · <span title="${esc(runs.join(', '))}">model runs</span>` : ''}</div>
    </div>
    <div class="fc-actions">
      <button class="btn" type="button" id="copy-api" title="${esc(state.rawUrl)}">
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>Copy API URL</button>
      <a class="btn" href="./#${encodeURIComponent(d.id)}">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2z"/><path d="M9 4v14M15 6v14"/></svg>Map &amp; scores</a>
      <a class="btn" href="${METARPARSE}#${encodeURIComponent(d.id)}" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><path d="M12 12 12 3"/></svg>Current METAR</a>
    </div>
  </div>`;

  if (!b) {
    $('fc-body').innerHTML = `${head}<div class="panel"><p class="muted" style="margin:0">No blend forecast is available for this airport right now; the individual model runs are in the <a href="${esc(state.rawUrl)}">JSON file</a>.</p></div>${trustHTML(d)}`;
    wireHead();
    return;
  }

  $('fc-body').innerHTML = `${head}
    <div class="fc-now" id="fc-now">${nowHTML(d)}</div>
    <div class="fc-grid">
      <section class="panel" aria-labelledby="h-mg">
        <h3 id="h-mg" style="margin:0 0 4px;font-size:15px">Next ${Math.round((b.ts[b.ts.length - 1] - b.ts[0]) / 3600)} hours</h3>
        <p class="note" style="margin:0 0 8px">Times in ${tzName()}. Shaded bands: 80% range from this airport's recent forecast errors. <span class="lk" style="display:inline-flex;gap:5px;align-items:center"><i class="sw-night"></i>night</span></p>
        <div class="readout" id="readout" aria-live="off"></div>
        <div class="mg-row"><div class="mg-label"><span>Temperature</span><span class="lk">
          <span><i class="sw-l" style="--c:var(--fc-t)"></i>Temp</span><span><i class="sw-band" style="--c:var(--fc-t)"></i>80% range</span>
          <span><i class="sw-l" style="--c:var(--fc-td)"></i>Dew point</span><span><i class="sw-l dash" style="--c:var(--fc-t)"></i>Feels like</span></span></div>
          <div class="mg" id="mg-t" role="img" aria-label="${esc(chartLabel('t'))}"></div></div>
        <div class="mg-row"><div class="mg-label"><span>Wind</span><span class="lk">
          <span><i class="sw-l" style="--c:var(--fc-wind)"></i>Speed</span><span><i class="sw-band" style="--c:var(--fc-wind)"></i>80% range</span>
          <span><i class="sw-l dash" style="--c:var(--fc-wind)"></i>Gusts</span><span>↑ arrows: where it blows to</span></span></div>
          <div class="mg" id="mg-w" role="img" aria-label="${esc(chartLabel('w'))}"></div></div>
        <div class="mg-row"><div class="mg-label"><span>Pressure (sea level)</span><span class="lk"><span><i class="sw-l" style="--c:var(--fc-p)"></i>hPa</span><span><i class="sw-band" style="--c:var(--fc-p)"></i>80% range</span></span></div>
          <div class="mg" id="mg-p" role="img" aria-label="${esc(chartLabel('p'))}"></div></div>
        <div class="mg-row"><div class="mg-label"><span>Cloud cover</span><span class="lk"><span><i class="sw-band" style="--c:var(--fc-cloud);opacity:.5"></i>% of sky (raw models, not corrected)</span></span></div>
          <div class="mg" id="mg-c" role="img" aria-label="${esc(chartLabel('c'))}"></div></div>
        ${cmpHTML(d)}
      </section>
      <div>${trustHTML(d)}${apiHTML(d)}</div>
    </div>
    <h2 style="margin-top:28px">Every 3 hours</h2>
    <div class="panel"><div class="tbl-wrap">${tableHTML(d)}</div></div>`;
  wireHead();
  $('copy-api2')?.addEventListener('click', copyApi);
  $('cmp')?.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-m]');
    if (!btn) return;
    const m = btn.dataset.m;
    if (state.cmp.has(m)) state.cmp.delete(m); else state.cmp.add(m);
    store.set('cmp', JSON.stringify([...state.cmp]));
    render();
  });
  drawCharts(d);
}
function wireHead() { $('copy-api')?.addEventListener('click', copyApi); }
async function copyApi() { toast((await copyText(state.rawUrl)) ? 'API URL copied' : state.rawUrl); }

function chartLabel(kind) {
  const b = state.doc.blend; const v = b.vals;
  const range = (a, f) => { const x = (a || []).filter(isNum).map(f); return x.length ? [Math.min(...x), Math.max(...x)] : null; };
  if (kind === 't') { const r = range(v.t, cT); return r ? `Temperature chart: between ${fmt(r[0], 0)} and ${fmt(r[1], 0)} ${T_UNIT[state.tu]} over the forecast. Values are in the table below.` : 'Temperature chart'; }
  if (kind === 'w') { const r = range(v.wspd, cW); const g = range(v.gust, cW); return r ? `Wind chart: ${fmt(r[0], 0)} to ${fmt(r[1], 0)} ${W_UNIT[state.wu]}${g ? `, gusts up to ${fmt(g[1], 0)}` : ''}.` : 'Wind chart'; }
  if (kind === 'p') { const r = range(v.mslp, (x) => x); return r ? `Pressure chart: ${fmt(r[0], 0)} to ${fmt(r[1], 0)} hPa.` : 'Pressure chart: no data'; }
  const r = range(v.tcc, (x) => x); return r ? `Cloud cover chart: ${fmt(r[0], 0)} to ${fmt(r[1], 0)} percent.` : 'Cloud cover chart: no data';
}

/* ---------- headline cards */
function nowHTML(d) {
  const o = d.obs || {};
  const b = d.blend;
  const t0 = b.ts[0];
  const idx24 = b.ts.map((t, i) => [t, i]).filter(([t]) => t < t0 + 24 * 3600).map(([, i]) => i);
  const sel = (k) => idx24.map((i) => b.vals[k]?.[i]).filter(isNum);
  const ts = sel('t'); const ws = sel('wspd'); const gs = sel('gust');
  const rhs = sel('rh');
  const tcc = sel('tcc');
  const owind = isNum(o.wspd) ? (o.wspd === 0 ? 'calm' : `${isNum(o.wdir) ? `${fmt(o.wdir, 0)}° ` : ''}${fW(o.wspd)} ${W_UNIT[state.wu]}${isNum(o.gust) ? ` G${fW(o.gust)}` : ''}`) : '–';
  const kpi = (k, v, s) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  return [
    kpi('Observed', isNum(o.t) ? fT(o.t) : '–', `${esc(owind)} · ${esc(ago(o.time))}`),
    kpi('Next 24 h', ts.length ? `${fT(Math.max(...ts))} <span class="muted" style="font-size:15px;font-weight:600">/ ${fT(Math.min(...ts))}</span>` : '–', 'High / low'),
    kpi('Wind, next 24 h', ws.length ? `${fW(Math.max(...ws))} <span class="muted" style="font-size:13px;font-weight:600">${W_UNIT[state.wu]}</span>` : '–', gs.length ? `Max sustained · gusts to ${fW(Math.max(...gs))}` : 'Max sustained'),
    kpi('Humidity / cloud', rhs.length ? `${fmt(Math.min(...rhs), 0)}–${fmt(Math.max(...rhs), 0)}%` : '–', tcc.length ? `RH range · cloud avg ${fmt(tcc.reduce((a, x) => a + x, 0) / tcc.length, 0)}%` : 'RH range'),
  ].join('');
}

/* ---------- comparison chips */
function cmpModels(d) {
  return modelOrder(Object.keys(d.models));
}
function cmpHTML(d) {
  const ms = cmpModels(d);
  if (!ms.length) return '';
  return `<div class="cmp" id="cmp" role="group" aria-label="Compare with individual models"><span class="cap">Compare:</span>${ms.map((m) => `<button type="button" class="key-btn" data-m="${esc(m)}" aria-pressed="${state.cmp.has(m)}">${MODEL[m] ? legendSwatch(m) : ''}${nameHTML(m)} <span class="muted">${esc(runLabel(d.models[m].run))}</span></button>`).join('')}</div>`;
}

/* ---------- trust box */
function trustHTML(d) {
  const sk = d.skill || {};
  const sm = sk.summary;
  const rows = [];
  const order = ['blend', ...modelOrder(Object.keys(sk.models || {}))];
  let bestT = Infinity;
  for (const m of order) { const a = sk.models?.[m]?.t; if (a && a[0] >= 20 && a[2] < bestT) bestT = a[2]; }
  for (const m of order) {
    const s = sk.models?.[m];
    if (!s) continue;
    const t = s.t; const w = s.wspd;
    const cell = (a, conv, digits) => (a && a[0] ? fmt(conv(a[2]), digits) : '–');
    rows.push(`<tr${m === 'blend' ? ' class="obs"' : ''}><td><span class="mname">${MODEL[m] ? legendSwatch(m) : ''}${nameHTML(m)}</span></td>
      <td${t && t[0] >= 20 && t[2] === bestT ? ' class="best"' : ''}>${cell(t, cTd, 1)}</td><td>${cell(w, cW, 1)}</td><td>${t ? fmtInt(t[0]) : '–'}</td></tr>`);
  }
  let headline;
  if (sm) {
    const mae = cTd(sm.mae);
    const vs = sm.better_pct || {};
    const cmp = ['ecmwf', 'gfs'].filter((m) => isNum(vs[m])).map((m) => {
      const p = vs[m]; const name = label(m);
      return p > 0 ? `<b class="good">${p}% better</b> than ${esc(name)}` : p < 0 ? `${-p}% worse than ${esc(name)}` : `as good as ${esc(name)}`;
    });
    headline = `<p class="big">Blend error here: <b>${fmt(mae, 1)} ${T_UNIT[state.tu]}</b> typical for temperature${cmp.length ? `, ${cmp.join(' and ')}` : ''}.</p>
      <p class="note" style="margin-top:0">Mean absolute error over ${fmtInt(sm.n)} checked forecasts in the last ${sk.window_days} days; the comparison uses only forecasts both made.</p>`;
  } else {
    headline = '<p class="big">Not enough verified forecasts at this airport yet for a track record.</p>';
  }
  const band = d.band?.t;
  let bandTxt = '';
  if (band) {
    const src = band.src.filter(Boolean);
    const own = src.filter((s) => s === 'station').length;
    bandTxt = own === src.length
      ? `The temperature band comes from this airport's own errors (${fmtInt(band.n.reduce((a, x) => a + x, 0))} past forecasts).`
      : own ? `The temperature band uses this airport's own errors where it has ${d.band.min_sample}+ cases per lead time, its region's otherwise.`
        : `Too few cases here yet: the temperature band uses ${src[0] === 'global' ? 'all airports' : 'airports in the same region'}.`;
  }
  return `<section class="panel trust" aria-labelledby="h-trust">
    <h3 id="h-trust">Why trust this?</h3>
    ${headline}
    ${rows.length ? `<div class="tbl-wrap"><table class="tbl"><caption>Typical error (MAE) at ${esc(d.id)}, last ${sk.window_days} days. ★ = best temperature.</caption>
      <thead><tr><th scope="col">Model</th><th scope="col">Temp<span class="u">${T_UNIT[state.tu]}</span></th><th scope="col">Wind<span class="u">${W_UNIT[state.wu]}</span></th><th scope="col">n</th></tr></thead>
      <tbody>${rows.join('')}</tbody></table></div>` : ''}
    <ul>
      <li>Each model is corrected for its own recent bias at this airport, then the corrected models are averaged.</li>
      ${bandTxt ? `<li>${bandTxt} In about 8 of 10 cases the observation fell inside it.</li>` : ''}
      <li>See every past forecast against what was observed on the <a href="./#${encodeURIComponent(d.id)}">map</a>, and the current report on <a href="${METARPARSE}#${encodeURIComponent(d.id)}" rel="noopener">MetarParse Live</a>.</li>
    </ul>
  </section>`;
}

function apiHTML(d) {
  return `<section class="panel api" aria-labelledby="h-api">
    <h3 id="h-api" style="margin:0 0 4px;font-size:15px">Use this forecast in your app</h3>
    <p class="note" style="margin:0">Free, no key, CORS-enabled JSON, refreshed every hour.</p>
    <code>${esc(state.rawUrl)}</code>
    <div class="fc-actions"><button class="btn small" type="button" id="copy-api2">Copy API URL</button>
    <a class="btn small" href="https://github.com/JerryWoof/StationTruth/blob/main/docs/FORECAST_API.md" rel="noopener">API docs</a>
    <a class="btn small" href="${esc(state.rawUrl)}">Raw JSON</a></div>
  </section>`;
}

/* ---------- 3-hourly table */
function arrowSVG(dir) {
  if (!isNum(dir)) return '';
  return `<svg class="arrow" viewBox="0 0 24 24" aria-hidden="true" style="transform:rotate(${Math.round(dir + 180)}deg)"><path d="M12 2l6 10h-4.2v10h-3.6V12H6z"/></svg>`;
}
function tableHTML(d) {
  const b = d.blend; const v = b.vals;
  const tu = T_UNIT[state.tu]; const wu = W_UNIT[state.wu];
  let lastDay = null;
  const rows = [];
  b.ts.forEach((t, i) => {
    if (new Date(t * 1000).getUTCHours() % 3 !== 0) return;
    const p = parts(t);
    if (p.key !== lastDay) { lastDay = p.key; rows.push(`<tr class="day"><th colspan="9" scope="rowgroup">${esc(dayLabel(t))}</th></tr>`); }
    const tLo = v.t_lo?.[i]; const tHi = v.t_hi?.[i];
    const rng = isNum(tLo) && isNum(tHi) ? `<span class="rng">${fmt(cT(tLo), 0)} to ${fmt(cT(tHi), 0)}</span>` : '';
    const wd = v.wdir?.[i];
    const tcc = v.tcc?.[i];
    rows.push(`<tr><th scope="row" style="text-align:left;font-weight:650;color:var(--text)">${esc(hourLabel(t))}</th>
      <td><span class="v">${fmt(cT(v.t?.[i]), 0)}°</span>${rng}</td>
      <td>${fmt(cT(v.td?.[i]), 0)}°</td>
      <td>${fmt(v.rh?.[i], 0)}%</td>
      <td>${fmt(cT(v.feels?.[i]), 0)}°</td>
      <td>${isNum(v.wspd?.[i]) ? `${arrowSVG(wd)}<span class="sr-only">from </span>${isNum(wd) ? `${fmt(wd, 0)}° ` : ''}${fmt(cW(v.wspd[i]), 0)}` : '–'}</td>
      <td>${fW(v.gust?.[i])}</td>
      <td>${fmt(v.mslp?.[i], 0)}</td>
      <td>${isNum(tcc) ? `${fmt(tcc, 0)}%<span class="cloudbar" aria-hidden="true"><i style="width:${Math.round(tcc)}%"></i></span>` : '–'}</td></tr>`);
  });
  return `<table class="tbl fc-tbl"><caption>Blend forecast, ${esc(tzName())}. Temperature with its 80% range; wind direction is where it comes from (arrow points where it blows).</caption>
    <thead><tr><th scope="col">Time</th><th scope="col">Temp<span class="u">${tu}</span></th><th scope="col">Dew pt<span class="u">${tu}</span></th><th scope="col">RH</th><th scope="col">Feels<span class="u">${tu}</span></th>
    <th scope="col">Wind<span class="u">${wu}</span></th><th scope="col">Gust<span class="u">${wu}</span></th><th scope="col">Pressure<span class="u">hPa</span></th><th scope="col">Cloud</th></tr></thead>
    <tbody>${rows.join('')}</tbody></table>`;
}

/* ------------------------------------------------------------------ charts */
function destroyCharts() {
  state.ro?.disconnect(); state.ro = null;
  for (const c of state.charts) c.destroy();
  state.charts = [];
}
function modelOnGrid(d, m, key, conv) {
  const blk = d.models[m];
  const b = d.blend;
  const out = new Array(b.ts.length).fill(null);
  if (!blk?.vals?.[key]) return out;
  const at = new Map(b.ts.map((t, i) => [t, i]));
  blk.ts.forEach((t, i) => { const j = at.get(t); if (j != null) out[j] = conv(blk.vals[key][i]); });
  return out;
}
function drawCharts(d) {
  if (typeof uPlot === 'undefined') return;
  const b = d.blend; const v = b.vals;
  const C = COL();
  const ax = axisStyle();
  const x = b.ts;
  const nights = nightSpans(d.lat, d.lon, x[0], x[x.length - 1]);
  const nowSec = Math.round(Date.now() / 1000);
  const dpr = () => devicePixelRatio || 1;
  const map = (a, f) => (a || []).map((y) => (isNum(y) ? f(y) : null));
  const id = (y) => y;
  const cmp = cmpModels(d).filter((m) => state.cmp.has(m));
  const narrow = narrowMQ.matches;
  const tzDate = state.tz === 'utc' ? (ts) => uPlot.tzDate(new Date(ts * 1e3), 'Etc/UTC') : (ts) => new Date(ts * 1e3);
  const xFmt = state.tz === 'utc'
    ? [[3600 * 24, '{WWW} {D}', null, null, null, null, null, null, 1], [3600, '{HH}Z', '\n{WWW} {D}', null, '\n{WWW} {D}', null, null, null, 1]]
    : [[3600 * 24, '{WWW} {D}', null, null, null, null, null, null, 1], [3600, '{HH}:00', '\n{WWW} {D}', null, '\n{WWW} {D}', null, null, null, 1]];

  const sync = uPlot.sync('fc');
  const width = $('mg-t').clientWidth || 360;
  const base = (height, yAxis, last, extra = {}) => ({
    width, height, tzDate,
    legend: { show: false },
    cursor: { sync: { key: sync.key }, y: false, points: { show: false }, drag: { x: false, y: false } },
    scales: { x: { time: true }, ...(extra.scales || {}) },
    padding: extra.padding || [8, 6, 0, 0],
    axes: [
      { ...ax, size: last ? 36 : 6, space: narrow ? 44 : 56, values: last ? xFmt : () => [], ticks: { ...ax.ticks, show: last } },
      { ...ax, size: 42, ...yAxis },
    ],
    hooks: {
      drawClear: [(u) => {
        const { top, height: h } = u.bbox;
        const ctx = u.ctx; ctx.save(); ctx.fillStyle = C.night;
        for (const [a, z] of nights) {
          const x0 = u.valToPos(a, 'x', true); const x1 = u.valToPos(z, 'x', true);
          ctx.fillRect(x0, top, Math.max(0, x1 - x0), h);
        }
        ctx.restore();
      }],
      draw: [(u) => {
        const x0 = u.valToPos(nowSec, 'x', true);
        const { left, top, width: w, height: h } = u.bbox;
        if (x0 >= left && x0 <= left + w) {
          const ctx = u.ctx; ctx.save(); ctx.strokeStyle = C.text; ctx.globalAlpha = 0.6; ctx.lineWidth = 1 * dpr(); ctx.setLineDash([3 * dpr(), 3 * dpr()]);
          ctx.beginPath(); ctx.moveTo(x0, top); ctx.lineTo(x0, top + h); ctx.stroke(); ctx.restore();
        }
        extra.draw?.(u);
      }],
      setCursor: [(u) => { const i = u.cursor.idx; if (i != null && i !== state.cursorIdx) { state.cursorIdx = i; readout(i); } }],
    },
  });
  const line = (label, stroke, extra = {}) => ({ label, stroke, width: 2, points: { show: false }, spanGaps: false, ...extra });
  const ghost = (label) => ({ label, stroke: 'transparent', width: 0, points: { show: false } });
  const cmpSeries = (key, conv) => cmp.map((m) => [line(label(m), modelColour(m), { width: MODEL[m]?.raw ? 1.2 : 1.6, dash: MODEL[m]?.raw ? [5, 4] : undefined, spanGaps: true }), modelOnGrid(d, m, key, conv)]);
  const H = narrow ? 150 : 170;
  const tVals = (vals) => vals.map((t) => fmt(t, 0));

  // temperature
  const tc = cmpSeries('t', cT);
  const tChart = new uPlot({
    ...base(narrow ? 170 : 200, { values: (u, vals) => tVals(vals), label: T_UNIT[state.tu], labelSize: 12, labelFont: ax.font }, false),
    series: [{}, ghost('t hi'), ghost('t lo'), line('Temp', C.t, { width: 2.4 }), ghost('td hi'), ghost('td lo'), line('Dew point', C.td), line('Feels like', C.t, { width: 1.2, dash: [4, 4] }), ...tc.map((s) => s[0])],
    bands: [{ series: [1, 2], fill: rgba(C.t, 0.16) }, { series: [4, 5], fill: rgba(C.td, 0.14) }],
  }, [x, map(v.t_hi, cT), map(v.t_lo, cT), map(v.t, cT), map(v.td_hi, cT), map(v.td_lo, cT), map(v.td, cT), map(v.feels, cT), ...tc.map((s) => s[1])], $('mg-t'));

  // wind (+ direction arrows above the plot, every 3 h)
  const wc = cmpSeries('wspd', cW);
  const drawArrows = (u) => {
    const ctx = u.ctx; const r = dpr();
    const y = u.bbox.top - 9 * r;
    const minGap = 26 * r;
    let lastX = -Infinity;
    ctx.save(); ctx.fillStyle = C.wind;
    x.forEach((t, i) => {
      if (new Date(t * 1000).getUTCHours() % 3 !== 0) return;
      const dir = v.wdir?.[i];
      if (!isNum(dir)) return;
      const px = u.valToPos(t, 'x', true);
      if (px - lastX < minGap || px < u.bbox.left || px > u.bbox.left + u.bbox.width) return;
      lastX = px;
      ctx.save(); ctx.translate(px, y); ctx.rotate(((dir + 180) * Math.PI) / 180);
      ctx.beginPath(); ctx.moveTo(0, -7 * r); ctx.lineTo(5 * r, 2 * r); ctx.lineTo(1.6 * r, 1 * r); ctx.lineTo(1.6 * r, 7 * r); ctx.lineTo(-1.6 * r, 7 * r); ctx.lineTo(-1.6 * r, 1 * r); ctx.lineTo(-5 * r, 2 * r); ctx.closePath(); ctx.fill();
      ctx.restore();
    });
    ctx.restore();
  };
  const wChart = new uPlot({
    ...base(H + 18, { values: (u, vals) => vals.map((w) => fmt(w, 0)), label: W_UNIT[state.wu], labelSize: 12, labelFont: ax.font }, false, { padding: [24, 6, 0, 0], draw: drawArrows, scales: { y: { range: (u, mn, mx) => [0, Math.max(5, mx * 1.08)] } } }),
    series: [{}, ghost('w hi'), ghost('w lo'), line('Wind', C.wind, { width: 2.2 }), line('Gust', C.wind, { width: 1.4, dash: [5, 4] }), ...wc.map((s) => s[0])],
    bands: [{ series: [1, 2], fill: rgba(C.wind, 0.14) }],
  }, [x, map(v.wspd_hi, cW), map(v.wspd_lo, cW), map(v.wspd, cW), map(v.gust, cW), ...wc.map((s) => s[1])], $('mg-w'));

  // pressure
  const pc = cmpSeries('mslp', id);
  const pChart = new uPlot({
    ...base(H - 20, { values: (u, vals) => vals.map((p) => fmt(p, 0)), label: 'hPa', labelSize: 12, labelFont: ax.font }, false),
    series: [{}, ghost('p hi'), ghost('p lo'), line('Pressure', C.p), ...pc.map((s) => s[0])],
    bands: [{ series: [1, 2], fill: rgba(C.p, 0.14) }],
  }, [x, map(v.mslp_hi, id), map(v.mslp_lo, id), map(v.mslp, id), ...pc.map((s) => s[1])], $('mg-p'));

  // cloud cover (raw models only)
  const cc = cmp.filter((m) => d.models[m]?.vals?.tcc).map((m) => [line(label(m), modelColour(m), { width: 1.2, dash: [5, 4], spanGaps: true }), modelOnGrid(d, m, 'tcc', id)]);
  const cChart = new uPlot({
    ...base(H - 30 + 36, { values: (u, vals) => vals.map((c) => fmt(c, 0)), splits: () => [0, 50, 100], label: '%', labelSize: 12, labelFont: ax.font }, true, { scales: { y: { range: [0, 100] } } }),
    series: [{}, line('Cloud', C.cloud, { width: 1.5, fill: rgba(C.cloud, 0.28) }), ...cc.map((s) => s[0])],
  }, [x, map(v.tcc, id), ...cc.map((s) => s[1])], $('mg-c'));

  state.charts = [tChart, wChart, pChart, cChart];
  for (const c of state.charts) sync.sub(c);
  readout(Math.min(state.cursorIdx, x.length - 1));
  const host = $('mg-t');
  state.ro = observeWidth(host, (w) => state.charts.forEach((c) => c.setSize({ width: w, height: c.height })));
}

function readout(i) {
  const el = $('readout');
  const d = state.doc;
  if (!el || !d?.blend) return;
  const b = d.blend; const v = b.vals;
  const t = b.ts[i];
  if (t == null) return;
  const tr = isNum(v.t_lo?.[i]) && isNum(v.t_hi?.[i]) ? ` <span class="muted">(${fmt(cT(v.t_lo[i]), 0)}–${fmt(cT(v.t_hi[i]), 0)})</span>` : '';
  const wind = isNum(v.wspd?.[i]) ? `${isNum(v.wdir?.[i]) ? `${fmt(v.wdir[i], 0)}° ` : ''}<b>${fmt(cW(v.wspd[i]), 0)}</b>${isNum(v.gust?.[i]) ? ` G${fmt(cW(v.gust[i]), 0)}` : ''} ${W_UNIT[state.wu]}` : '–';
  el.innerHTML = `<span><b>${esc(dayLabel(t))} ${esc(hourLabel(t))}</b></span>
    <span class="rk"><i class="sw-l" style="--c:var(--fc-t)"></i><b>${fmt(cT(v.t?.[i]), 1)}</b> ${T_UNIT[state.tu]}${tr}</span>
    <span class="rk"><i class="sw-l" style="--c:var(--fc-td)"></i>dew ${fmt(cT(v.td?.[i]), 1)} · RH ${fmt(v.rh?.[i], 0)}%</span>
    <span class="rk"><i class="sw-l" style="--c:var(--fc-wind)"></i>${wind}</span>
    <span class="rk"><i class="sw-l" style="--c:var(--fc-p)"></i>${fmt(v.mslp?.[i], 0)} hPa</span>
    <span class="rk">cloud ${fmt(v.tcc?.[i], 0)}%</span>`;
}

onThemeChange(() => { if (state.doc) render(); });
let narrowWas = narrowMQ.matches;
narrowMQ.addEventListener('change', () => { if (narrowMQ.matches !== narrowWas) { narrowWas = narrowMQ.matches; if (state.doc) render(); } });

/* ------------------------------------------------------------------ boot */
async function boot() {
  initHeader({ active: 'forecast' });
  try {
    state.meta = await loadMeta();
    const idx = await getJSON('data/forecast/index.json', { optional: true }).catch(() => null);
    state.stations = Array.isArray(idx?.stations) ? idx.stations.filter((s) => s && typeof s.id === 'string') : [];
    state.byId = new Map(state.stations.map((s) => [s.id, s]));
    setStatus(state.meta, state.stations.length ? `${fmtInt(state.stations.length)} forecasts` : '');
  } catch (e) {
    console.error(e);
    setStatus(null);
  }
  const id = hashId();
  if (id) load(id); else showWelcome();
}
boot();
