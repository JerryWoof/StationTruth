// StationTruth — map page
import {
  MODELS, MODEL, VARS, METARPARSE, isNum, esc, fmt, fmtSigned, fmtInt, median, parseTime, runLabel, ago, utcLabel,
  getJSON, loadMeta, progress, collectingHTML, setStatus, cssVar, modelColour, divColour, DIV_GRADIENT,
  darkMQ, onThemeChange, axisStyle, observeWidth, seg, legendSwatch,
} from './common.js';

const $ = (id) => document.getElementById(id);
const MAP_VARS = ['t', 'td', 'wspd', 'mslp'];
const LOW_HIGH = {
  t: ['too cold', 'too warm'], td: ['too dry', 'too moist'], wspd: ['too weak', 'too strong'], mslp: ['too low', 'too high'],
};
const MIN_N = 8;

const store = {
  get(k, d) { try { return localStorage.getItem(`st.${k}`) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`st.${k}`, v); } catch { /* private mode */ } },
};
const pick = (v, allowed, d) => (allowed.includes(v) ? v : d);

const state = {
  meta: null,
  scores: null,
  stations: [],
  byId: new Map(),
  mode: pick(store.get('mode'), ['error', 'bias', 'best'], 'error'),
  model: pick(store.get('model'), MODELS.map((m) => m.id), 'ecmwf'),
  v: pick(store.get('var'), MAP_VARS, 't'),
  selected: null,
  detail: null, // stations/<ID>.json of the selected station
  chartVar: 't',
  scoreWin: '7d',
  hidden: new Set(JSON.parse(store.get('hidden', '[]') || '[]')),
};
const now = () => (state.meta?.generated ? parseTime(state.meta.generated) : new Date());

/* ------------------------------------------------------------------ map */
const isTouch = matchMedia('(pointer: coarse)').matches;
const narrowMQ = matchMedia('(max-width: 720px)');
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

const renderer = L.canvas({ padding: 0.5, tolerance: isTouch ? 10 : 4 });
const map = L.map('map', {
  preferCanvas: true, renderer, zoomControl: false, worldCopyJump: true,
  minZoom: 2, maxZoom: 12, zoomSnap: 0.5, wheelPxPerZoomLevel: 90,
}).setView([28, narrowMQ.matches ? 10 : -5], clamp(Math.floor(Math.log2(($('map').clientWidth || 1000) / 300) * 2) / 2, 2, 3.5));
L.control.zoom({ position: 'topright' }).addTo(map);
map.attributionControl.setPrefix(
  '<a href="https://leafletjs.com" rel="noopener">Leaflet</a> · Obs: <a href="https://aviationweather.gov" rel="noopener">aviationweather.gov</a> · Forecasts: Source <a href="https://www.ecmwf.int" rel="noopener">ECMWF</a> (<a href="https://creativecommons.org/licenses/by/4.0/" rel="noopener">CC BY 4.0</a>, modified), NOAA GFS · Not for operational use',
);

// Basemap: Esri Canvas (keyless) — same as MetarParse Live. CARTO now requires an API key.
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_{style}/MapServer/tile/{z}/{y}/{x}';
const ESRI_ATTR = 'Tiles &copy; <a href="https://www.esri.com" rel="noopener">Esri</a> — Esri, HERE, Garmin, &copy; <a href="https://www.openstreetmap.org/copyright" rel="noopener">OpenStreetMap</a> contributors';
const tileUrl = (kind) => ESRI.replace('{style}', `${darkMQ.matches ? 'Dark' : 'Light'}_Gray_${kind}`);
const tiles = L.tileLayer(tileUrl('Base'), { maxZoom: 16, maxNativeZoom: 16, attribution: ESRI_ATTR }).addTo(map);
map.createPane('labels');
map.getPane('labels').style.zIndex = 250;
map.getPane('labels').style.pointerEvents = 'none';
const labels = L.tileLayer(tileUrl('Reference'), { maxZoom: 16, maxNativeZoom: 16, pane: 'labels', opacity: 0.85 }).addTo(map);

const group = L.featureGroup().addTo(map);
const markers = new Map();
let selRing = null;
let COL = {};
function readColours() {
  COL = { stroke: cssVar('--dot-stroke'), unk: cssVar('--unk') };
  for (const m of MODELS) COL[m.id] = modelColour(m.id);
}

function radiusFor(z, small) { const r = clamp(1.4 + z * 0.8, 2.8, 9); return small ? Math.max(1.8, r * 0.55) : r; }

// value used to colour a station in the current mode; null = no data
function valueOf(s) {
  if (state.mode === 'best') return s.best || null;
  if (state.mode === 'bias') { const b = s.bias7?.[state.model]?.[state.v]; return isNum(b) ? b : null; }
  const f = s.fc?.[state.model]?.[state.v];
  const o = s.obs?.[state.v];
  return isNum(f) && isNum(o) ? f - o : null;
}
function look(s) {
  const val = valueOf(s);
  if (val == null) return [COL.unk, -1, true];
  if (state.mode === 'best') return [COL[val] || COL.unk, 1, false];
  const r = VARS[state.v].range * (state.mode === 'bias' ? 0.75 : 1);
  return [divColour(val / r), Math.abs(val), false];
}

function restyleAll() {
  const z = map.getZoom();
  const items = [];
  for (const s of state.stations) {
    const m = markers.get(s.id);
    if (!m) continue;
    const [fill, rank, small] = look(s);
    m.setStyle({ fillColor: fill, color: COL.stroke, weight: z >= 4 ? 1 : 0.6, fillOpacity: small ? 0.5 : 0.95, opacity: small ? 0.35 : 0.9 });
    m.setRadius(radiusFor(z, small));
    m._small = small;
    items.push([rank, m]);
  }
  // canvas draws in insertion order: the largest errors last so they sit on top
  group.clearLayers();
  items.sort((a, b) => a[0] - b[0]);
  for (const [, m] of items) group.addLayer(m);
}
map.on('zoomend', () => {
  const z = map.getZoom();
  for (const m of markers.values()) { m.setRadius(radiusFor(z, m._small)); m.setStyle({ weight: z >= 4 ? 1 : 0.6 }); }
});

function buildMarkers() {
  markers.clear();
  for (const s of state.stations) {
    const m = L.circleMarker([s.lat, s.lon], { renderer, radius: 3, stroke: true, bubblingMouseEvents: false });
    m._sid = s.id;
    markers.set(s.id, m);
  }
  restyleAll();
}

const tip = L.tooltip({ className: 'st-tip', direction: 'top', offset: [0, -6], opacity: 1 });
group.on('mouseover', (e) => {
  if (isTouch) return;
  const s = state.byId.get(e.layer._sid);
  if (!s) return;
  tip.setLatLng(e.layer.getLatLng()).setContent(`<span class="tid">${esc(s.id)}</span> ${esc(s.name || '')}<br><span class="tval">${esc(tipValue(s))}</span>`);
  map.openTooltip(tip);
});
group.on('mouseout', () => map.closeTooltip(tip));
group.on('click', (e) => { map.closeTooltip(tip); openStation(e.layer._sid, { fly: false }); });
map.on('click', () => { if (state.selected) closeStation(); });

function tipValue(s) {
  const v = valueOf(s);
  const V = VARS[state.v];
  const mname = MODEL[state.model].label;
  if (state.mode === 'best') return v ? `Best 7-day temperature model: ${MODEL[v]?.label || v}` : 'Not enough history yet';
  if (v == null) return state.mode === 'bias' ? `No 7-day ${V.label.toLowerCase()} bias yet` : `No ${V.label.toLowerCase()} to compare`;
  if (state.mode === 'bias') return `${mname} 7-day bias ${fmtSigned(v)} ${V.unit}`;
  return `Obs ${fmt(s.obs[state.v])} · ${mname} ${fmt(s.fc[state.model][state.v])} ${V.unit} (${fmtSigned(v)})`;
}

/* ------------------------------------------------------------------ controls + legend */
const modelSel = $('model-sel');
modelSel.innerHTML = MODELS.map((m) => `<option value="${m.id}">${esc(m.long)}</option>`).join('');
modelSel.value = state.model;
modelSel.addEventListener('change', () => { state.model = modelSel.value; store.set('model', state.model); refresh(); });
const varSel = $('var-sel');
varSel.value = state.v;
varSel.addEventListener('change', () => { state.v = varSel.value; store.set('var', state.v); refresh(); });
seg($('mode-seg'), state.mode, (v) => { state.mode = v; store.set('mode', v); refresh(); });

function refresh() {
  const best = state.mode === 'best';
  modelSel.disabled = best;
  varSel.disabled = best;
  restyleAll();
  renderLegend();
}

function renderLegend() {
  const el = $('legend');
  const withData = state.stations.filter((s) => valueOf(s) != null).length;
  if (state.mode === 'best') {
    const counts = Object.fromEntries(MODELS.map((m) => [m.id, 0]));
    for (const s of state.stations) if (s.best && s.best in counts) counts[s.best]++;
    el.innerHTML = `<div class="cap">Lowest 7-day temperature RMSE at each station (≥ ${MIN_N} pairs)</div>
      <div class="sw">${MODELS.map((m) => `<span><i style="background:var(--m-${m.id})"></i>${esc(m.label)} <b>${fmtInt(counts[m.id])}</b></span>`).join('')}</div>
      ${withData ? '' : '<div class="cap" style="margin-top:6px">No station has enough history yet.</div>'}`;
    return;
  }
  const V = VARS[state.v];
  const r = V.range * (state.mode === 'bias' ? 0.75 : 1);
  const [lo, hi] = LOW_HIGH[state.v];
  const what = state.mode === 'bias' ? `${MODEL[state.model].label} 7-day mean error` : `${MODEL[state.model].label} forecast − observed, now`;
  el.innerHTML = `<div class="cap">${esc(what)} · ${esc(V.label.toLowerCase())}, ${esc(V.unit)}</div>
    <div class="ramp" style="background:${DIV_GRADIENT}"></div>
    <div class="ticks"><span>≤ −${fmt(r, r < 4 ? 1 : 0)}</span><span>−${fmt(r / 2, r < 4 ? 1 : 0)}</span><span>0</span><span>+${fmt(r / 2, r < 4 ? 1 : 0)}</span><span>≥ +${fmt(r, r < 4 ? 1 : 0)}</span></div>
    <div class="ends"><span class="c">◀ model ${lo}</span><span class="muted">${fmtInt(withData)} stations</span><span class="w">model ${hi} ▶</span></div>`;
}

/* ------------------------------------------------------------------ headline */
function absErrors(model, v) {
  const out = [];
  for (const s of state.stations) {
    const f = s.fc?.[model]?.[v]; const o = s.obs?.[v];
    if (isNum(f) && isNum(o)) out.push(Math.abs(f - o));
  }
  return out;
}
function renderHeadline() {
  const raw = absErrors('ecmwf', 't');
  const bc = absErrors('ecmwf_bc', 't');
  const mRaw = median(raw); const mBc = median(bc);
  const head = $('headline'); const short = $('headline-short'); const sub = $('headline-sub');
  if (mRaw == null) {
    head.innerHTML = 'No recent forecast–observation pairs yet. The map fills in as soon as the first model run is verified.';
    short.textContent = 'Waiting for the first verified forecasts';
  } else {
    let tail;
    // compare what we display (1 decimal) so we never claim "cuts 0.9 to 0.9"
    const r = Number(fmt(mRaw)); const rb = mBc == null ? null : Number(fmt(mBc));
    if (rb != null && rb < r) tail = `; bias correction cuts it to <span class="good">${fmt(mBc)} °C</span>.`;
    else if (rb != null && rb > r) tail = `; bias correction isn't helping yet (${fmt(mBc)} °C).`;
    else tail = '; bias correction hasn\'t changed it yet — it needs a few days of history.';
    head.innerHTML = `Right now, <b>ECMWF's</b> median temperature error is <b>${fmt(mRaw)} °C</b>${tail}`;
    short.textContent = rb != null && rb < r ? `ECMWF median temp. error ${fmt(mRaw)} °C → ${fmt(mBc)} °C corrected` : `ECMWF median temperature error ${fmt(mRaw)} °C`;
  }
  const g = median(absErrors('gfs', 't')); const gb = median(absErrors('gfs_bc', 't'));
  const parts = [];
  if (g != null) parts.push(`GFS ${fmt(g)} °C${gb != null && Number(fmt(gb)) < Number(fmt(g)) ? ` → ${fmt(gb)} °C corrected` : ''}`);
  parts.push(`${fmtInt(raw.length)} stations reporting in the last 3 h`);
  // best model over the last 7 days (needs a meaningful sample)
  const glob = state.scores?.windows?.['7d']?.global;
  if (glob) {
    let best = null;
    for (const m of MODELS) {
      const a = glob[m.id]?.t?.all;
      if (a && a.n >= 1000 && isNum(a.rmse) && (!best || a.rmse < best[1])) best = [m.id, a.rmse];
    }
    if (best) parts.push(`best over 7 days: <b>${esc(MODEL[best[0]].label)}</b> (RMSE ${fmt(best[1])} °C) — <a href="scores.html">scoreboard</a>`);
  }
  sub.innerHTML = parts.join(' · ');
  $('collecting').innerHTML = collectingHTML(state.meta);
}
const infoToggle = $('info-toggle');
infoToggle.addEventListener('click', () => infoToggle.setAttribute('aria-expanded', String(infoToggle.getAttribute('aria-expanded') !== 'true')));

/* ------------------------------------------------------------------ station panel */
const panel = $('station');
const body = $('station-body');
let chart = null; let chartRO = null;

async function openStation(id, { fly = true, push = true } = {}) {
  id = String(id || '').toUpperCase();
  const s = state.byId.get(id);
  state.selected = id;
  state.detail = null;
  if (push && decodeURIComponent(location.hash.slice(1)).toUpperCase() !== id) history.pushState({ id }, '', `#${id}`);
  document.body.classList.add('panel-open');
  panel.hidden = false;
  if (s) {
    const ll = [s.lat, s.lon];
    if (!selRing) selRing = L.marker(ll, { icon: L.divIcon({ className: 'sel-ring', iconSize: [22, 22] }), interactive: false, keyboard: false }).addTo(map);
    else selRing.setLatLng(ll).addTo(map);
    if (fly) {
      const z = Math.max(map.getZoom(), 6);
      const pt = map.project(ll, z);
      // keep the station clear of the side panel / bottom sheet
      if (narrowMQ.matches) pt.y += (window.innerHeight * 0.62) / 2 - 20;
      else pt.x += (panel.offsetWidth + 16) / 2;
      map.flyTo(map.unproject(pt, z), z, { duration: 0.8 });
    }
  } else if (selRing) selRing.remove();
  renderStation({ loading: true });
  const detail = await getJSON(`data/stations/${encodeURIComponent(id)}.json`, { optional: true }).catch(() => null);
  if (state.selected !== id) return;
  state.detail = detail;
  if (!s && !detail) {
    toast(`No recent data for ${id}`);
    closeStation({ push: false });
    history.replaceState(null, '', location.pathname + location.search);
    return;
  }
  renderStation({});
  if (!s && detail && fly) map.flyTo([detail.lat, detail.lon], 7, { duration: 0.8 });
}

function closeStation({ push = true } = {}) {
  state.selected = null;
  state.detail = null;
  panel.hidden = true;
  document.body.classList.remove('panel-open');
  if (selRing) selRing.remove();
  destroyChart();
  if (push && location.hash) history.pushState(null, '', location.pathname + location.search);
}
function destroyChart() { chartRO?.disconnect(); chartRO = null; chart?.destroy(); chart = null; }

function errCell(f, o, d = 1) {
  if (!isNum(f)) return '<span class="v muted">–</span>';
  if (!isNum(o)) return `<span class="v">${fmt(f, d)}</span>`;
  const e = f - o;
  const cls = Math.abs(e) < 0.05 ? 'zero' : e < 0 ? 'cold' : 'warm';
  return `<span class="v">${fmt(f, d)}</span><span class="e ${cls}">${fmtSigned(e, d)}</span>`;
}

function renderStation({ loading = false, keepScroll = false } = {}) {
  const id = state.selected;
  const s = state.byId.get(id);
  const d = state.detail;
  const info = s || d || { id };
  const scroll = panel.scrollTop;
  destroyChart();
  const cc = info.country ? ` · ${esc(info.country)}` : '';
  const elev = isNum(d?.elev_m) ? ` · ${fmtInt(d.elev_m)} m` : '';
  const pos = isNum(info.lat) ? `${Math.abs(info.lat).toFixed(2)}°${info.lat >= 0 ? 'N' : 'S'} ${Math.abs(info.lon).toFixed(2)}°${info.lon >= 0 ? 'E' : 'W'}` : '';

  let html = `<div class="sp-head">
    <div class="sp-title">
      <div class="sp-ids"><span class="icao">${esc(id)}</span><span>${pos}${elev}${cc}</span></div>
      <h2 tabindex="-1" id="sp-name">${esc(info.name || id)}</h2>
      <div class="sp-sub">${s ? `Observed ${esc(utcLabel(s.time))} (${esc(ago(s.time, now()))})` : 'No observation in the last 3 hours'}
        · <a href="${METARPARSE}#${encodeURIComponent(id)}" rel="noopener">Live METAR ↗</a></div>
    </div>
    <div class="sp-actions">
      <button class="btn icon-only" type="button" id="sp-close" aria-label="Close station details" title="Close (Esc)">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>
      </button>
    </div>
  </div>`;

  // --- obs now vs forecasts
  html += '<section class="sec"><div class="sec-h"><h3>Now: observed vs forecast</h3></div>';
  if (s) {
    const o = s.obs || {};
    html += `<div class="tbl-wrap"><table class="tbl now-tbl">
      <thead><tr><th scope="col">Source</th><th scope="col">Temp<span class="u">°C</span></th><th scope="col">Dew pt<span class="u">°C</span></th><th scope="col">Wind<span class="u">kt</span></th><th scope="col">MSLP<span class="u">hPa</span></th></tr></thead>
      <tbody><tr class="obs"><td><span class="mname"><i class="sw-obs"></i>Observed</span></td>
        <td>${fmt(o.t)}</td><td>${fmt(o.td)}</td><td>${fmt(o.wspd, 0)}</td><td>${fmt(o.mslp)}</td></tr>
      ${MODELS.map((m) => {
        const f = s.fc?.[m.id];
        const lead = f ? `${runLabel(f.run)} +${f.lead_h} h` : 'no forecast';
        return `<tr><td><span class="mname" title="Run ${esc(lead)}">${legendSwatch(m.id, false)}${esc(m.label)}</span></td>
          <td>${errCell(f?.t, o.t)}</td><td>${errCell(f?.td, o.td)}</td><td>${errCell(f?.wspd, o.wspd)}</td><td>${errCell(f?.mslp, o.mslp)}</td></tr>`;
      }).join('')}</tbody></table></div>
      <p class="note">Small numbers are forecast − observed: <span style="color:var(--cold);font-weight:600">blue</span> = model too low/cold, <span style="color:var(--warm);font-weight:600">red</span> = too high/warm. MSLP is only verified where the station reports sea-level pressure.</p>`;
  } else {
    html += '<div class="empty-note">This station has no observation in the last 3 hours.</div>';
  }
  html += '</section>';

  // --- chart
  html += `<section class="sec"><div class="sec-h"><h3>Last 7 days &amp; next 72 h</h3>
      <div class="seg small" id="cv-seg" role="group" aria-label="Chart variable">
        ${MAP_VARS.map((v) => `<button type="button" data-v="${v}">${esc(VARS[v].short)}</button>`).join('')}
      </div></div>`;
  if (loading) html += '<div class="chart"><div class="loading" style="position:absolute;background:none"><div class="spinner"></div></div></div>';
  else if (!d) html += '<div class="empty-note">No time series for this station yet.</div>';
  else {
    html += `<div class="chart" id="chart" role="img" aria-label="${esc(VARS[state.chartVar].label)}: observed vs model forecasts"></div>
      <div class="chart-key" id="chart-key">
        <span class="key-btn key-static"><i class="sw-obs"></i>Observed</span>
        ${MODELS.map((m) => `<button type="button" class="key-btn" data-m="${m.id}" aria-pressed="${!state.hidden.has(m.id)}">${legendSwatch(m.id, false)}${esc(m.label)}</button>`).join('')}
      </div>
      <div class="chart-hint"><span><i class="past"></i>past: 24 h forecasts</span><span><i class="fut"></i>future: latest run${d.future?.ecmwf?.run ? ` (ECMWF ${esc(runLabel(d.future.ecmwf.run))})` : ''}</span></div>`;
  }
  html += '</section>';

  // --- station scores
  html += `<section class="sec"><div class="sec-h"><h3>Scores here · ${esc(VARS[state.chartVar].label.toLowerCase())}</h3>
      <div class="seg small" id="win-seg" role="group" aria-label="Score window"><button type="button" data-v="7d">7 days</button><button type="button" data-v="30d">30 days</button></div></div>
      <div id="st-scores">${loading ? '' : stationScoresHTML(d)}</div></section>`;
  if (state.meta && progress(state.meta).early) html += collectingHTML(state.meta);

  body.innerHTML = html;
  $('sp-close').addEventListener('click', () => closeStation());
  seg($('cv-seg'), state.chartVar, (v) => { state.chartVar = v; renderStation({ keepScroll: true }); });
  seg($('win-seg'), state.scoreWin, (v) => { state.scoreWin = v; $('st-scores').innerHTML = stationScoresHTML(state.detail); });
  $('chart-key')?.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-m]');
    if (!b) return;
    const m = b.dataset.m;
    if (state.hidden.has(m)) state.hidden.delete(m); else state.hidden.add(m);
    store.set('hidden', JSON.stringify([...state.hidden]));
    b.setAttribute('aria-pressed', String(!state.hidden.has(m)));
    chart?._toggle(m, !state.hidden.has(m));
  });
  if (d && !loading) drawChart(d);
  panel.scrollTop = keepScroll ? scroll : 0;
  if (!keepScroll && !loading && !narrowMQ.matches) $('sp-name')?.focus({ preventScroll: true });
}

function stationScoresHTML(d) {
  const sc = d?.scores?.[state.scoreWin];
  const v = state.chartVar; const unit = VARS[v].unit;
  const rows = MODELS.map((m) => [m, sc?.[m.id]?.[v]]);
  const ok = rows.filter(([, r]) => r && r.n >= MIN_N);
  if (!ok.length) {
    return `<div class="empty-note">Not enough verified pairs here yet — a score needs at least ${MIN_N} forecast–observation pairs per model.</div>`;
  }
  const best = (k, f = (x) => x) => Math.min(...ok.map(([, r]) => (isNum(r[k]) ? f(r[k]) : Infinity)));
  const bB = best('bias', Math.abs); const bM = best('mae'); const bR = best('rmse');
  const cell = (r, k, b, f = (x) => x, signed = false) => {
    if (!r || r.n < MIN_N || !isNum(r[k])) return '<td class="muted">–</td>';
    return `<td class="${Math.abs(f(r[k]) - b) < 1e-9 ? 'best' : ''}">${signed ? fmtSigned(r[k]) : fmt(r[k])}</td>`;
  };
  return `<div class="tbl-wrap"><table class="tbl">
    <thead><tr><th scope="col">Model</th><th scope="col">n</th><th scope="col">Bias</th><th scope="col">MAE</th><th scope="col">RMSE</th></tr></thead>
    <tbody>${rows.map(([m, r]) => `<tr class="${!r || r.n < MIN_N ? 'thin' : ''}"><td><span class="mname">${legendSwatch(m.id, false)}${esc(m.label)}</span></td>
      <td>${fmtInt(r?.n ?? 0)}</td>${cell(r, 'bias', bB, Math.abs, true)}${cell(r, 'mae', bM)}${cell(r, 'rmse', bR)}</tr>`).join('')}</tbody>
    <caption>All forecast lead times, ${esc(unit)}. ★ = best. Bias &gt; 0 means the model forecasts too high. Rows with fewer than ${MIN_N} pairs are not scored.</caption>
  </table></div>`;
}

/* ------------------------------------------------------------------ chart */
const toSec = (t) => Math.round(parseTime(t) / 1000);

function drawChart(d) {
  const el = $('chart');
  if (!el || typeof uPlot === 'undefined') return;
  const v = state.chartVar;
  const V = VARS[v];
  // union of all timestamps
  const xs = new Set();
  const obsT = (d.obs?.time || []).map(toSec);
  obsT.forEach((t) => xs.add(t));
  const src = [];
  for (const m of MODELS) {
    for (const kind of ['past', 'future']) {
      const blk = d[kind]?.[m.id];
      const ts = (blk?.valid || []).map(toSec);
      ts.forEach((t) => xs.add(t));
      src.push({ m: m.id, kind, ts, vals: blk?.[v] || [] });
    }
  }
  const x = [...xs].sort((a, b) => a - b);
  const idx = new Map(x.map((t, i) => [t, i]));
  const col = (ts, vals) => { const a = new Array(x.length).fill(null); ts.forEach((t, i) => { if (isNum(vals[i])) a[idx.get(t)] = vals[i]; }); return a; };
  const data = [x, col(obsT, d.obs?.[v] || [])];
  const series = [{}, { label: 'Observed', stroke: cssVar('--obs'), width: 2.6, spanGaps: false, points: { show: false } }];
  const seriesModel = [null, null];
  for (const s of src) {
    data.push(col(s.ts, s.vals));
    const raw = MODEL[s.m].raw;
    series.push({
      label: `${MODEL[s.m].label} ${s.kind}`, stroke: modelColour(s.m), width: raw ? 1.4 : 1.9,
      dash: s.kind === 'future' ? [6, 4] : undefined, spanGaps: true, show: !state.hidden.has(s.m), points: { show: false },
    });
    seriesModel.push(s.m);
  }
  if (!data.slice(1).some((a) => a.some(isNum))) {
    el.outerHTML = `<div class="empty-note">No ${esc(V.label.toLowerCase())} data at this station.</div>`;
    return;
  }
  const nowSec = Math.round(now() / 1000);
  const ax = axisStyle();
  const nowCol = cssVar('--text'); const futFill = cssVar('--grid');
  const tipEl = document.createElement('div');
  tipEl.className = 'chart-tip';
  tipEl.hidden = true;
  const opts = {
    width: el.clientWidth || 380,
    height: narrowMQ.matches ? 210 : 240,
    tzDate: (ts) => uPlot.tzDate(new Date(ts * 1e3), 'Etc/UTC'),
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false } },
    scales: { x: { time: true } },
    axes: [
      { ...ax, size: 34, space: 60, values: [[3600 * 24, '{D} {MMM}', null, null, null, null, null, null, 1], [3600, '{HH}Z', '\n{D} {MMM}', null, '\n{D} {MMM}', null, null, null, 1]] },
      { ...ax, size: 44, label: V.unit, labelSize: 14, labelFont: ax.font, values: (u, vals) => vals.map((t) => fmt(t, V.unit === 'hPa' || Math.abs(t) >= 10 ? 0 : 1)) },
    ],
    series,
    hooks: {
      drawClear: [(u) => {
        // shade the future
        const x0 = u.valToPos(nowSec, 'x', true);
        const { left, top, width, height } = u.bbox;
        if (x0 < left + width) {
          const ctx = u.ctx; ctx.save(); ctx.fillStyle = futFill;
          ctx.fillRect(Math.max(left, x0), top, left + width - Math.max(left, x0), height); ctx.restore();
        }
      }],
      draw: [(u) => {
        const x0 = u.valToPos(nowSec, 'x', true);
        const { left, top, width, height } = u.bbox;
        if (x0 < left || x0 > left + width) return;
        const ctx = u.ctx; const dpr = devicePixelRatio || 1;
        ctx.save(); ctx.strokeStyle = nowCol; ctx.lineWidth = 1.5 * dpr; ctx.setLineDash([3 * dpr, 3 * dpr]);
        ctx.beginPath(); ctx.moveTo(x0, top); ctx.lineTo(x0, top + height); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = nowCol; ctx.font = `700 ${10 * dpr}px system-ui, sans-serif`; ctx.textAlign = 'left';
        ctx.fillText('now', x0 + 4 * dpr, top + 11 * dpr); ctx.restore();
      }],
      setCursor: [(u) => {
        const i = u.cursor.idx;
        if (i == null) { tipEl.hidden = true; return; }
        const t = u.data[0][i];
        const rows = [];
        const near = (si) => {
          // models are 3-hourly: use the nearest point within 90 min
          for (let k = 0; k <= 3; k++) {
            for (const j of [i - k, i + k]) {
              if (j >= 0 && j < x.length && Math.abs(x[j] - t) <= 5400 && isNum(u.data[si][j])) return u.data[si][j];
            }
          }
          return null;
        };
        const ob = u.data[1][i];
        if (isNum(ob)) rows.push(`<div class="tr"><span><i class="sw-obs"></i>Observed</span><b>${fmt(ob)}</b></div>`);
        const seen = new Set();
        for (let si = 2; si < u.series.length; si++) {
          const m = seriesModel[si];
          if (seen.has(m) || !u.series[si].show) continue;
          const val = near(si);
          if (val == null) continue;
          seen.add(m);
          const e = isNum(ob) ? ` <span class="muted">${fmtSigned(val - ob)}</span>` : '';
          rows.push(`<div class="tr"><span>${legendSwatch(m, false)}${esc(MODEL[m].label)}</span><b>${fmt(val)}${e}</b></div>`);
        }
        if (!rows.length) { tipEl.hidden = true; return; }
        const dt = new Date(t * 1000);
        tipEl.innerHTML = `<div class="tt">${dt.getUTCDate()} ${dt.toLocaleString('en-GB', { month: 'short', timeZone: 'UTC' })} ${utcLabel(dt)} · ${esc(V.unit)}</div>${rows.join('')}`;
        tipEl.hidden = false;
        const left = u.cursor.left + u.bbox.left / (devicePixelRatio || 1);
        const w = tipEl.offsetWidth; const W = el.clientWidth;
        tipEl.style.left = `${left + 14 + w > W ? Math.max(0, left - 14 - w) : left + 14}px`;
      }],
    },
  };
  chart = new uPlot(opts, data, el);
  el.appendChild(tipEl);
  el.addEventListener('mouseleave', () => { tipEl.hidden = true; });
  chart._toggle = (m, show) => { seriesModel.forEach((sm, si) => { if (sm === m) chart.setSeries(si, { show }); }); };
  chartRO = observeWidth(el, (w) => chart?.setSize({ width: w, height: opts.height }));
}

/* ------------------------------------------------------------------ search */
const input = $('search-input');
const list = $('search-list');
let results = []; let active = -1;
function norm(t) { return String(t || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase(); }
function search(q) {
  q = norm(q.trim());
  if (!q) return [];
  const up = q.toUpperCase();
  const a = []; const b = [];
  for (const s of state.stations) {
    if (s.id.startsWith(up)) a.push(s);
    else if (norm(s.name).includes(q)) b.push(s);
    if (a.length > 30) break;
  }
  a.sort((x, y) => x.id.length - y.id.length || x.id.localeCompare(y.id));
  return a.concat(b).slice(0, 12);
}
function renderResults() {
  const q = input.value.trim();
  if (!q) { list.hidden = true; input.setAttribute('aria-expanded', 'false'); return; }
  results = search(q);
  active = results.length ? 0 : -1;
  list.innerHTML = results.length
    ? results.map((s, i) => `<li role="option" id="opt-${i}" data-id="${esc(s.id)}" aria-selected="${i === active}"><span class="dot" style="background:${look(s)[0]}"></span><span class="sid">${esc(s.id)}</span><span class="sname">${esc(s.name)}</span><span class="scc">${esc(s.country)}</span></li>`).join('')
    : `<li class="empty">No station with a recent observation matches “${esc(q)}”.</li>`;
  list.hidden = false;
  input.setAttribute('aria-expanded', 'true');
  input.setAttribute('aria-activedescendant', active >= 0 ? `opt-${active}` : '');
}
function setActive(i) {
  if (!results.length) return;
  active = (i + results.length) % results.length;
  list.querySelectorAll('li[role=option]').forEach((li, k) => li.setAttribute('aria-selected', String(k === active)));
  input.setAttribute('aria-activedescendant', `opt-${active}`);
  list.querySelector(`#opt-${active}`)?.scrollIntoView({ block: 'nearest' });
}
function choose(id) { list.hidden = true; input.setAttribute('aria-expanded', 'false'); input.value = ''; input.blur(); openStation(id); }
input.addEventListener('input', renderResults);
input.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') { e.preventDefault(); setActive(active + 1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(active - 1); }
  else if (e.key === 'Enter') {
    e.preventDefault();
    if (active >= 0) choose(results[active].id);
    else if (/^[A-Za-z0-9]{3,4}$/.test(input.value.trim())) choose(input.value.trim());
  } else if (e.key === 'Escape') { input.value = ''; renderResults(); input.blur(); }
});
input.addEventListener('blur', () => setTimeout(() => { list.hidden = true; input.setAttribute('aria-expanded', 'false'); }, 150));
input.addEventListener('focus', () => { if (input.value.trim()) renderResults(); });
list.addEventListener('mousedown', (e) => { const li = e.target.closest('li[data-id]'); if (li) { e.preventDefault(); choose(li.dataset.id); } });
document.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName);
  if (e.key === '/' && !typing) { e.preventDefault(); input.focus(); }
  else if (e.key === 'Escape' && state.selected && !typing) closeStation();
});

/* ------------------------------------------------------------------ hash + misc */
function applyHash() {
  const id = decodeURIComponent(location.hash.slice(1)).trim().toUpperCase();
  if (id && /^[A-Z0-9]{3,5}$/.test(id)) { if (id !== state.selected) openStation(id, { push: false }); }
  else if (state.selected) closeStation({ push: false });
}
window.addEventListener('popstate', applyHash);
window.addEventListener('hashchange', applyHash);

let toastT;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 2600); }

onThemeChange(() => {
  tiles.setUrl(tileUrl('Base'));
  labels.setUrl(tileUrl('Reference'));
  readColours();
  restyleAll();
  renderLegend();
  if (state.selected) renderStation({ keepScroll: true });
});

/* ------------------------------------------------------------------ boot */
async function boot() {
  readColours();
  if (narrowMQ.matches) infoToggle.setAttribute('aria-expanded', 'false');
  else infoToggle.setAttribute('aria-expanded', 'true');
  const loading = $('loading');
  try {
    state.meta = await loadMeta();
    const [latest, scores] = await Promise.all([
      getJSON('data/latest.json', { optional: true }),
      getJSON('data/scores.json', { optional: true }).catch(() => null),
    ]);
    state.scores = scores;
    state.stations = (Array.isArray(latest) ? latest : []).filter((s) => isNum(s.lat) && isNum(s.lon));
    state.byId = new Map(state.stations.map((s) => [s.id, s]));
    setStatus(state.meta, state.stations.length ? `${fmtInt(state.stations.length)} stations` : '');
    buildMarkers();
    refresh();
    renderHeadline();
    loading.classList.add('gone');
    if (!state.meta && !state.stations.length) {
      $('headline').innerHTML = 'No data published yet. The first observations and forecasts are being collected — check back in an hour.';
      $('headline-short').textContent = 'No data yet — collecting';
      $('headline-sub').textContent = '';
      $('collecting').innerHTML = '';
    }
    applyHash();
  } catch (err) {
    console.error(err);
    loading.classList.add('err');
    loading.innerHTML = `<span>Couldn't load the data (${esc(err.message)}).</span><button class="btn" type="button" onclick="location.reload()">Retry</button>`;
  }
}
boot();
// re-check for fresh data every 10 minutes (the site regenerates hourly)
setInterval(async () => {
  const meta = await loadMeta().catch(() => null);
  if (!meta || meta.generated === state.meta?.generated) { setStatus(state.meta, state.stations.length ? `${fmtInt(state.stations.length)} stations` : ''); return; }
  const latest = await getJSON('data/latest.json', { optional: true }).catch(() => null);
  if (!Array.isArray(latest)) return;
  state.meta = meta;
  state.stations = latest.filter((s) => isNum(s.lat) && isNum(s.lon));
  state.byId = new Map(state.stations.map((s) => [s.id, s]));
  setStatus(meta, `${fmtInt(state.stations.length)} stations`);
  buildMarkers(); refresh(); renderHeadline();
  if (state.selected) openStation(state.selected, { fly: false, push: false });
}, 600000);
