// StationTruth — shared helpers for every page (map, scoreboard, about).

export const REPO = 'https://github.com/JerryWoof/StationTruth';
export const METARPARSE = 'https://jerrywoof.github.io/MetarParse/';

/* ------------------------------------------------------------------ models
 * One hue per model family (Okabe–Ito based, colour-blind safe):
 * ECMWF = blue, GFS = orange, Blend = bluish green.
 * Raw model = lighter tint (+ thinner / dashed in charts), corrected = full colour.
 */
export const MODELS = [
  { id: 'ecmwf', label: 'ECMWF', long: 'ECMWF IFS (raw)', raw: true, family: 'ecmwf' },
  { id: 'ecmwf_bc', label: 'ECMWF corr.', long: 'ECMWF, bias-corrected', raw: false, family: 'ecmwf' },
  { id: 'gfs', label: 'GFS', long: 'NOAA GFS (raw)', raw: true, family: 'gfs' },
  { id: 'gfs_bc', label: 'GFS corr.', long: 'GFS, bias-corrected', raw: false, family: 'gfs' },
  { id: 'blend', label: 'Blend', long: 'Blend of both corrected models', raw: false, family: 'blend' },
];
export const MODEL = Object.fromEntries(MODELS.map((m) => [m.id, m]));

export const VARS = {
  t: { label: 'Temperature', short: 'Temp', unit: '°C', range: 4, digits: 1 },
  td: { label: 'Dew point', short: 'Dew pt', unit: '°C', range: 4, digits: 1 },
  wspd: { label: 'Wind speed', short: 'Wind', unit: 'kt', range: 8, digits: 1 },
  gust: { label: 'Wind gust', short: 'Gust', unit: 'kt', range: 10, digits: 1 },
  mslp: { label: 'Pressure (MSL)', short: 'Pressure', unit: 'hPa', range: 3, digits: 1 },
  wdir: { label: 'Wind direction', short: 'Wind dir', unit: '°', range: 60, digits: 0 },
};

/* ------------------------------------------------------------------ theme */
export const darkMQ = matchMedia('(prefers-color-scheme: dark)');
export const isDark = () => darkMQ.matches;
export function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
export function modelColour(id) { return cssVar(`--m-${id}`) || '#888'; }
export function onThemeChange(fn) { darkMQ.addEventListener('change', fn); }

// Diverging blue (forecast too low) -> neutral -> red (forecast too high). x in [-1, 1].
const DIV = [
  [-1, [33, 102, 172]], [-0.6, [67, 147, 195]], [-0.3, [146, 197, 222]], [-0.1, [209, 229, 240]],
  [0, [236, 236, 236]],
  [0.1, [253, 219, 199]], [0.3, [244, 165, 130]], [0.6, [214, 96, 77]], [1, [178, 24, 43]],
];
export function divColour(x) {
  x = Math.max(-1, Math.min(1, x));
  for (let i = 1; i < DIV.length; i++) {
    const [x1, c1] = DIV[i];
    if (x <= x1) {
      const [x0, c0] = DIV[i - 1];
      const f = (x - x0) / (x1 - x0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return 'rgb(178,24,43)';
}
export const DIV_GRADIENT = `linear-gradient(90deg, ${DIV.map(([x, c]) => `rgb(${c.join(',')}) ${((x + 1) * 50).toFixed(0)}%`).join(', ')})`;

/* ------------------------------------------------------------------ format */
export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const fmt = (v, d = 1) => (isNum(v) ? v.toFixed(d).replace(/^-(0\.?0*)$/, '$1') : '–');
export const fmtSigned = (v, d = 1) => {
  if (!isNum(v)) return '–';
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return (v > 0 ? '+' : '−') + Math.abs(v).toFixed(d);
};
export const fmtInt = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : '–');
export const fmtCompact = (v) => {
  if (!isNum(v)) return '–';
  if (v >= 1e6) return `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`;
  if (v >= 1e4) return `${Math.round(v / 1e3)}k`;
  return fmtInt(v);
};
export const median = (arr) => {
  const a = arr.filter(isNum).sort((x, y) => x - y);
  if (!a.length) return null;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};
export const parseTime = (s) => (s ? new Date(s) : null);
// "2026101212" -> Date
export function parseRun(r) {
  if (!r || r.length < 10) return null;
  return new Date(Date.UTC(+r.slice(0, 4), +r.slice(4, 6) - 1, +r.slice(6, 8), +r.slice(8, 10)));
}
export function runLabel(r) {
  const d = parseRun(r);
  if (!d) return '–';
  return `${d.getUTCDate()} ${d.toLocaleString('en-GB', { month: 'short', timeZone: 'UTC' })} ${String(d.getUTCHours()).padStart(2, '0')}Z`;
}
export function ago(t, now = Date.now()) {
  const d = t instanceof Date ? t : parseTime(t);
  if (!d || Number.isNaN(+d)) return 'unknown';
  const m = Math.round((now - d) / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
}
export function dateLabel(t) {
  const d = t instanceof Date ? t : parseTime(t);
  if (!d) return '–';
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}
export function utcLabel(t) {
  const d = t instanceof Date ? t : parseTime(t);
  if (!d) return '–';
  return `${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')} UTC`;
}

/* ------------------------------------------------------------------ data */
let bust = String(Math.floor(Date.now() / 300000)); // 5-minute cache key until meta is known
export async function getJSON(path, { optional = false } = {}) {
  let r;
  try {
    r = await fetch(`${path}?v=${encodeURIComponent(bust)}`, { cache: 'no-cache' });
  } catch (e) {
    if (optional) return null;
    throw e;
  }
  if (!r.ok) {
    if (optional && r.status === 404) return null;
    throw new Error(`${path}: HTTP ${r.status}`);
  }
  return r.json();
}
export async function loadMeta() {
  const meta = await getJSON('data/meta.json', { optional: true });
  if (meta?.generated) bust = meta.generated;
  return meta;
}

// How much history we have; scores are considered "early" for the first 3 days.
export function progress(meta) {
  if (!meta || !meta.first_obs) return { days: 0, pairs: 0, early: true, empty: true };
  const first = parseTime(meta.first_obs);
  const gen = parseTime(meta.generated) || new Date();
  const days = Math.max(0, (gen - first) / 86400000);
  return { days, pairs: meta.pairs || 0, early: days < 3, empty: !meta.pairs, first };
}

// "Collecting data since …" banner; returns '' when we have enough history.
export function collectingHTML(meta, { always = false } = {}) {
  const p = progress(meta);
  if (!always && !p.early) return '';
  if (!meta) {
    return `<div class="collecting" role="status"><b>No data yet.</b> The first observations and forecasts are being collected — check back in an hour.</div>`;
  }
  const pct = Math.min(100, (p.days / 7) * 100);
  return `<div class="collecting" role="status">
    <div><b>Collecting data since ${esc(dateLabel(p.first))}</b> — scores become meaningful after a few days.</div>
    <div class="prog" aria-hidden="true"><span style="width:${pct.toFixed(1)}%"></span></div>
    <div class="prog-txt">${p.days < 1 ? `${Math.round(p.days * 24)} hours` : `${p.days.toFixed(1)} days`} collected · ${fmtInt(p.pairs)} forecast–observation pairs</div>
  </div>`;
}

export function setStatus(meta, extra = '') {
  const el = document.getElementById('updated');
  const st = document.getElementById('status');
  if (!el) return;
  if (!meta) { el.textContent = 'No data yet'; st?.classList.add('stale'); return; }
  const gen = parseTime(meta.generated);
  const stale = gen && Date.now() - gen > 3 * 3600000;
  el.textContent = `Updated ${ago(gen)}`;
  el.title = gen ? gen.toISOString().replace('.000', '') : '';
  st?.classList.toggle('stale', !!stale);
  const c = document.getElementById('count');
  if (c) c.textContent = extra || (meta.station_count ? `${fmtInt(meta.station_count)} stations` : '');
}

/* ------------------------------------------------------------------ uPlot helpers */
export function axisStyle() {
  return {
    stroke: cssVar('--muted'),
    grid: { stroke: cssVar('--grid'), width: 1 },
    ticks: { stroke: cssVar('--grid'), width: 1, size: 4 },
    font: '11px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  };
}
// Re-create uPlot charts when their container resizes.
export function observeWidth(el, fn) {
  let w = el.clientWidth;
  const ro = new ResizeObserver(() => {
    const nw = el.clientWidth;
    if (Math.abs(nw - w) > 2) { w = nw; fn(nw); }
  });
  ro.observe(el);
  return ro;
}

// Segmented control helper: buttons with data-v inside el.
export function seg(el, value, onChange) {
  const btns = [...el.querySelectorAll('button')];
  const set = (v) => btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
  set(value);
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || !el.contains(b) || b.disabled) return;
    set(b.dataset.v);
    onChange(b.dataset.v);
  });
  return set;
}

// dashRaw: raw models drawn dashed (scoreboard); on the map page dashes mean "future", so raw is thin instead
export function legendSwatch(id, dashRaw = true) {
  const m = MODEL[id];
  return `<i class="sw-line ${m?.raw ? (dashRaw ? 'raw' : 'thin') : ''}" style="--c:var(--m-${id})"></i>`;
}
