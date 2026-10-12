// StationTruth — analysis page: ECMWF background corrected toward METARs (optimal interpolation)
import {
  isNum, esc, fmt, fmtSigned, fmtInt, getJSON, loadMeta, setStatus, initHeader, darkMQ, onThemeChange, seg,
  runLabel, utcLabel, dateLabel, parseTime,
} from './common.js?v=__BUILD__';

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { return localStorage.getItem(`st.an.${k}`) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`st.an.${k}`, v); } catch { /* private mode */ } },
};
const pick = (v, allowed, d) => (allowed.includes(v) ? v : d);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

const LAYERS = { analysis: 'Analysis', background: 'Model', increment: 'Correction' };
const LOW_HIGH = { t: ['colder', 'warmer'], wspd: ['calmer', 'windier'] };
const STATUS = { 1: 'rejected: implausible vs neighbours / model', 2: 'not used: elevation differs from model terrain', 3: 'outside model grid' };

const state = {
  doc: null,
  v: pick(store.get('var'), ['t', 'wspd'], 't'),
  layer: pick(store.get('layer'), Object.keys(LAYERS), 'analysis'),
  opacity: clamp(Number(store.get('opacity', 0.75)) || 0.75, 0.2, 1),
  dots: store.get('dots', '0') === '1',
};

/* ------------------------------------------------------------------ map */
const isTouch = matchMedia('(pointer: coarse)').matches;
const narrowMQ = matchMedia('(max-width: 720px)');
const renderer = L.canvas({ padding: 0.5, tolerance: isTouch ? 10 : 4 });
const map = L.map('map', {
  preferCanvas: true, renderer, zoomControl: false, worldCopyJump: true,
  minZoom: 2, maxZoom: 10, zoomSnap: 0.5, wheelPxPerZoomLevel: 90,
}).setView([30, narrowMQ.matches ? 10 : 0], clamp(Math.floor(Math.log2(($('map').clientWidth || 1000) / 300) * 2) / 2, 2, 3.5));
L.control.zoom({ position: 'topright' }).addTo(map);
map.attributionControl.setPrefix(
  '<a href="https://leafletjs.com" rel="noopener">Leaflet</a> · Obs: <a href="https://aviationweather.gov" rel="noopener">aviationweather.gov</a> · Background: Source <a href="https://www.ecmwf.int" rel="noopener">ECMWF</a> (<a href="https://creativecommons.org/licenses/by/4.0/" rel="noopener">CC BY 4.0</a>, modified) · Not for operational use',
);
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_{style}/MapServer/tile/{z}/{y}/{x}';
const ESRI_ATTR = 'Tiles &copy; <a href="https://www.esri.com" rel="noopener">Esri</a> — Esri, HERE, Garmin, &copy; <a href="https://www.openstreetmap.org/copyright" rel="noopener">OpenStreetMap</a> contributors';
const tileUrl = (kind) => ESRI.replace('{style}', `${darkMQ.matches ? 'Dark' : 'Light'}_Gray_${kind}`);
const tiles = L.tileLayer(tileUrl('Base'), { maxZoom: 16, maxNativeZoom: 16, attribution: ESRI_ATTR }).addTo(map);
// field between the base map and the labels, so place names stay readable
map.createPane('field');
map.getPane('field').style.zIndex = 240;
map.getPane('field').style.pointerEvents = 'none';
map.createPane('labels');
map.getPane('labels').style.zIndex = 250;
map.getPane('labels').style.pointerEvents = 'none';
const labels = L.tileLayer(tileUrl('Reference'), { maxZoom: 16, maxNativeZoom: 16, pane: 'labels', opacity: 0.85 }).addTo(map);
onThemeChange(() => { tiles.setUrl(tileUrl('Base')); labels.setUrl(tileUrl('Reference')); restyleDots(); });

let overlays = [];
function drawField() {
  for (const o of overlays) o.remove();
  overlays = [];
  const V = state.doc?.variables?.[state.v];
  if (!V) return;
  const img = V.images[state.layer];
  const url = `data/analysis/${img.file}?v=${encodeURIComponent(state.doc.generated)}`;
  const [[s, w], [n, e]] = state.doc.bounds;
  // one copy either side so the field continues across the date line
  for (const off of [-360, 0, 360]) {
    overlays.push(L.imageOverlay(url, [[s, w + off], [n, e + off]], {
      pane: 'field', opacity: state.opacity, interactive: false, className: 'field', alt: '',
    }).addTo(map));
  }
}

/* ------------------------------------------------------------------ colours */
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rampFn(stops) {
  const s = stops.map(([v, c]) => [v, hexRgb(c)]);
  return (x) => {
    if (!(x > s[0][0])) return `rgb(${s[0][1]})`;
    for (let i = 1; i < s.length; i++) {
      if (x <= s[i][0]) {
        const f = (x - s[i - 1][0]) / (s[i][0] - s[i - 1][0]);
        return `rgb(${s[i - 1][1].map((a, k) => Math.round(a + (s[i][1][k] - a) * f))})`;
      }
    }
    return `rgb(${s[s.length - 1][1]})`;
  };
}
const gradient = (stops) => {
  const lo = stops[0][0]; const hi = stops[stops.length - 1][0];
  return `linear-gradient(90deg, ${stops.map(([v, c]) => `${c} ${(((v - lo) / (hi - lo)) * 100).toFixed(1)}%`).join(', ')})`;
};

/* ------------------------------------------------------------------ values */
function sampleGrid(V, lat, lon) {
  const g = V.grid;
  const fj = (lat - g.lat0) / g.step;
  if (fj < 0 || fj > g.nlat - 1) return null;
  const j0 = Math.min(Math.floor(fj), g.nlat - 2); const wj = fj - j0;
  let fi = (((lon - g.lon0) % 360) + 360) % 360 / g.step;
  const i0 = Math.floor(fi) % g.nlon; const i1 = (i0 + 1) % g.nlon; fi -= Math.floor(fi);
  const at = (arr) => {
    const v = (j, i) => arr[j * g.nlon + i];
    return ((1 - wj) * ((1 - fi) * v(j0, i0) + fi * v(j0, i1)) + wj * ((1 - fi) * v(j0 + 1, i0) + fi * v(j0 + 1, i1))) / g.scale;
  };
  const a = at(g.analysis); const inc = at(g.increment);
  return { a, inc, b: a - inc };
}
const latLabel = (lat) => `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}`;
const lonLabel = (lon) => { const l = ((((lon + 180) % 360) + 360) % 360) - 180; return `${Math.abs(l).toFixed(1)}°${l >= 0 ? 'E' : 'W'}`; };

function showReadout(ll) {
  const V = state.doc?.variables?.[state.v];
  const el = $('readout');
  if (!V) return;
  const r = sampleGrid(V, ll.lat, ll.lng);
  if (!r) { el.innerHTML = `<span class="pos">${latLabel(ll.lat)} ${lonLabel(ll.lng)}</span> · outside the map`; return; }
  const u = V.unit;
  const cls = Math.abs(r.inc) < 0.05 ? '' : r.inc > 0 ? 'warm' : 'cold';
  el.innerHTML = `<span class="pos">${latLabel(ll.lat)} ${lonLabel(ll.lng)}</span><br>
    Analysis <b>${fmt(r.a)} ${esc(u)}</b> · Model ${fmt(r.b)} · Correction <span class="inc ${cls}">${fmtSigned(r.inc)}</span>`;
}
let hoverRaf = 0;
map.on('mousemove', (e) => {
  if (isTouch) return;
  cancelAnimationFrame(hoverRaf);
  hoverRaf = requestAnimationFrame(() => showReadout(e.latlng));
});
map.on('click', (e) => showReadout(e.latlng));

/* ------------------------------------------------------------------ station dots */
const dotGroup = L.featureGroup();
const tip = L.tooltip({ className: 'st-tip', direction: 'top', offset: [0, -6], opacity: 1 });
function radiusFor(z) { return clamp(1 + z * 0.6, 2.2, 7); }

function buildDots() {
  dotGroup.clearLayers();
  const V = state.doc?.variables?.[state.v];
  if (!V || !state.dots) { dotGroup.remove(); return; }
  const S = V.stations;
  const z = map.getZoom();
  const col = rampFn(V.scale.value);
  const unk = getComputedStyle(document.documentElement).getPropertyValue('--unk').trim() || '#94a3b8';
  const stroke = getComputedStyle(document.documentElement).getPropertyValue('--dot-stroke').trim() || '#fff';
  // rejected first, so used stations draw on top
  const order = S.id.map((_, k) => k).sort((a, b) => (S.status[b] - S.status[a]));
  for (const k of order) {
    if (!isNum(S.lat[k]) || !isNum(S.lon[k])) continue;
    const used = S.status[k] === 0;
    const m = L.circleMarker([S.lat[k], S.lon[k]], {
      renderer, radius: radiusFor(z) * (used ? 1 : 0.8), bubblingMouseEvents: false,
      fillColor: used ? col(S.obs[k]) : 'transparent', fillOpacity: used ? 0.95 : 0,
      color: used ? stroke : unk, weight: used ? 0.8 : 1.4, opacity: 0.95,
    });
    m._k = k;
    dotGroup.addLayer(m);
  }
  dotGroup.addTo(map);
}
function restyleDots() { if (state.dots) buildDots(); }
map.on('zoomend', () => { if (!state.dots) return; const r = radiusFor(map.getZoom()); dotGroup.eachLayer((m) => m.setRadius(r * (state.doc.variables[state.v].stations.status[m._k] === 0 ? 1 : 0.8))); });

function dotText(k) {
  const V = state.doc.variables[state.v];
  const S = V.stations; const u = V.unit;
  const model = isNum(S.bg[k]) ? S.bg[k] - (S.adj[k] || 0) : null; // background at the station's height
  let html = `<span class="tid">${esc(S.id[k])}</span> reported <b>${fmt(S.obs[k])} ${esc(u)}</b><br>`;
  if (S.status[k] !== 0) return `${html}<span class="tval">Model ${fmt(model)} ${esc(u)} · ${esc(STATUS[S.status[k]] || 'not used')}</span>`;
  const without = isNum(S.loo[k]) ? S.obs[k] - S.loo[k] : null;
  return `${html}<span class="tval">Model ${fmt(model)} · analysis without this station ${fmt(without)} ${esc(u)}</span>`;
}
dotGroup.on('mouseover', (e) => {
  if (isTouch) return;
  tip.setLatLng(e.layer.getLatLng()).setContent(dotText(e.layer._k));
  map.openTooltip(tip);
});
dotGroup.on('mouseout', () => map.closeTooltip(tip));
dotGroup.on('click', (e) => {
  tip.setLatLng(e.layer.getLatLng()).setContent(dotText(e.layer._k));
  map.openTooltip(tip);
  showReadout(e.layer.getLatLng());
});

/* ------------------------------------------------------------------ legend + text */
function renderLegend() {
  const V = state.doc?.variables?.[state.v];
  const el = $('legend');
  if (!V) { el.innerHTML = ''; return; }
  const u = V.unit;
  if (state.layer === 'increment') {
    const r = V.scale.increment.range;
    const [lo, hi] = LOW_HIGH[state.v] || ['lower', 'higher'];
    const st = V.scale.increment.stops;
    const ticks = [-r, -r / 2, 0, r / 2, r];
    el.innerHTML = `<div class="cap">Correction = analysis − model, ${esc(u)} · blank where |correction| &lt; ${fmt(V.scale.increment.threshold, 1)}</div>
      <div class="ramp" style="background:${gradient(st)}"></div>
      <div class="ticks">${ticks.map((t) => `<span style="left:${(((t + r) / (2 * r)) * 100).toFixed(1)}%">${t === 0 ? '0' : fmtSigned(t, 0)}</span>`).join('')}</div>
      <div class="ends"><span class="c">◀ reports ${lo} than model</span><span class="w">reports ${hi} ▶</span></div>`;
    return;
  }
  const stops = V.scale.value;
  const lo = stops[0][0]; const hi = stops[stops.length - 1][0];
  const ticks = state.v === 't' ? [-40, -20, 0, 20, 40] : [0, 10, 20, 34, 48, 64];
  el.innerHTML = `<div class="cap">${esc(LAYERS[state.layer])} · ${esc(V.label)}, ${esc(u)}</div>
    <div class="ramp" style="background:${gradient(stops)}"></div>
    <div class="ticks">${ticks.map((t) => `<span style="left:${(((t - lo) / (hi - lo)) * 100).toFixed(1)}%">${t}${state.v === 't' ? '°' : ''}</span>`).join('')}</div>`;
}

function renderInfo() {
  const d = state.doc; const V = d?.variables?.[state.v];
  if (!V) return;
  const valid = parseTime(d.valid);
  const bgd = d.background;
  const va = V.validation; const u = V.unit;
  const a = va.analysis?.mae; const b = va.background?.mae; const raw = va.model_raw?.mae;
  const better = isNum(a) && isNum(b) && b > 0 ? Math.round((1 - a / b) * 100) : null;
  $('headline').innerHTML = `${esc(V.label[0].toUpperCase() + V.label.slice(1))} at <b>${esc(utcLabel(valid))}</b>, ${esc(dateLabel(valid))}: the ECMWF forecast, corrected toward <b>${fmtInt(V.counts.used)}</b> airport reports.`;
  $('headline-short').textContent = isNum(a) && isNum(b)
    ? `${utcLabel(valid)} · withheld-station error ${fmt(a)} vs ${fmt(b)} ${u} for the model`
    : `${utcLabel(valid)} analysis`;
  $('val').innerHTML = `
    <div class="kpi good"><div class="k">Analysis</div><div class="v">${fmt(a, 2)} ${esc(u)}</div><div class="s">typical error at withheld stations${isNum(better) && better > 0 ? ` · ${better}% lower` : ''}</div></div>
    <div class="kpi"><div class="k">Model alone</div><div class="v">${fmt(b, 2)} ${esc(u)}</div><div class="s">same stations, same hour</div></div>`;
  const rawNote = state.v === 't' && isNum(raw) && Math.abs(raw - b) >= 0.005
    ? ` Without the height adjustment the raw model is off by ${fmt(raw, 2)} ${esc(u)}.` : '';
  $('val-sub').innerHTML = `At withheld stations the analysis error is <b>${fmt(a, 2)} ${esc(u)}</b> vs <b>${fmt(b, 2)} ${esc(u)}</b> for the model (mean absolute error, ${fmtInt(va.analysis?.n)} stations, each left out together with neighbours within ${fmt(va.withheld_radius_km, 0)} km).${rawNote}`;
  const p = V.params;
  $('how-body').innerHTML = `<ol>
      <li><b>Background:</b> ECMWF IFS run ${esc(runLabel(bgd.run))}, +${fmt(bgd.lead_h, 0)} h (steps ${bgd.steps.join(' & ')} h blended in time), 0.25° grid.</li>
      <li><b>Observations:</b> the METAR nearest ${esc(utcLabel(valid))} (±${d.obs_window_min} min) at each airport${V.params.lapse_rate_k_per_km ? `, moved to the model's terrain height with ${fmt(V.params.lapse_rate_k_per_km, 1)} °C/km` : ''}.</li>
      <li><b>Quality control:</b> ${fmtInt(V.counts.rejected_orography)} stations more than ${fmtInt(p.max_dz_m)} m above/below the model terrain and ${fmtInt(V.counts.rejected_gross)} reports more than ${fmt(p.qc_sd, 0)} robust SD from the typical model miss are left out.</li>
      <li><b>Spreading:</b> optimal interpolation; model errors are assumed correlated as e<sup>−r/${fmtInt(p.corr_length_km)} km</sup>, report/model error ratio ${fmt(p.sigma_o, 1)}/${fmt(p.sigma_b, 1)} ${esc(u)}. Corrections fade to zero within ${fmtInt(p.cutoff_km)} km, so over oceans and data voids the map is the model.</li>
    </ol>
    <p class="muted">Temperatures are at the model's terrain height, which is smoother than the real one. Built ${esc(dateLabel(parseTime(d.generated)))} ${esc(utcLabel(parseTime(d.generated)))}.</p>`;
}

function refresh() {
  drawField();
  renderLegend();
  renderInfo();
  buildDots();
}

/* ------------------------------------------------------------------ controls */
seg($('var-seg'), state.v, (v) => { state.v = v; store.set('var', v); refresh(); });
seg($('layer-seg'), state.layer, (v) => { state.layer = v; store.set('layer', v); drawField(); renderLegend(); });
const op = $('opacity');
op.value = String(state.opacity);
op.addEventListener('input', () => {
  state.opacity = Number(op.value);
  store.set('opacity', op.value);
  for (const o of overlays) o.setOpacity(state.opacity);
});
const dots = $('dots');
dots.checked = state.dots;
dots.addEventListener('change', () => { state.dots = dots.checked; store.set('dots', state.dots ? '1' : '0'); buildDots(); });

const infoToggle = $('info-toggle');
infoToggle.addEventListener('click', () => infoToggle.setAttribute('aria-expanded', String(infoToggle.getAttribute('aria-expanded') !== 'true')));

/* ------------------------------------------------------------------ boot */
async function boot() {
  initHeader({ active: 'analysis' });
  infoToggle.setAttribute('aria-expanded', narrowMQ.matches ? 'false' : 'true');
  const loading = $('loading');
  try {
    const meta = await loadMeta().catch(() => null);
    const doc = await getJSON('data/analysis/analysis.json', { optional: true });
    if (!doc || !doc.variables || !Object.keys(doc.variables).length) {
      setStatus(meta);
      loading.classList.add('err');
      loading.querySelector('span').textContent = 'No analysis yet — it is rebuilt every hour.';
      loading.querySelector('.spinner')?.remove();
      return;
    }
    state.doc = doc;
    if (!doc.variables[state.v]) state.v = Object.keys(doc.variables)[0];
    for (const b of $('var-seg').querySelectorAll('button')) {
      b.disabled = !doc.variables[b.dataset.v];
      b.setAttribute('aria-pressed', String(b.dataset.v === state.v));
    }
    setStatus(meta, `Analysis ${utcLabel(parseTime(doc.valid))}`);
    refresh();
    loading.classList.add('gone');
  } catch (e) {
    console.error(e);
    loading.classList.add('err');
    loading.querySelector('span').textContent = 'Could not load the analysis. Try again in a minute.';
    loading.querySelector('.spinner')?.remove();
  }
}
boot();
