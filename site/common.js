// StationTruth — shared helpers for every page (map, analysis, scores, TAFs, about).
// Pages import this as './common.js?v=__BUILD__' (the deploy replaces __BUILD__; unreplaced it is a harmless literal).
// Every module on a page must use the same URL, or the browser loads common.js twice.

export const REPO = 'https://github.com/JerryWoof/StationTruth';
export const METARPARSE = 'https://jerrywoof.github.io/MetarParse/';

/* ------------------------------------------------------------------ models
 * REGISTRY mirrors src/stationtruth/registry.py (same ids, order and kind; checked by
 * tests/test_registry.py). Every raw model <id> gets a bias-corrected <id>_bc, plus
 * one 'blend' of all corrected models. Each family has its own hue (Okabe–Ito / Tol
 * based, colour-blind safe): ECMWF IFS blue, GFS orange, ICON olive, AIFS reddish
 * purple, AIGFS brown, Blend bluish green. Raw model = lighter tint (+ thinner /
 * dashed in charts), corrected = full colour. AI models carry an "AI" badge.
 * c: [raw light, corrected light, raw dark, corrected dark]
 */
const REGISTRY = [
  { id: 'ecmwf', kind: 'physics', label: 'ECMWF', long: 'ECMWF IFS', step: 3, c: ['#5ea3d9', '#0060a8', '#8cc4f2', '#3d9bf0'] },
  { id: 'gfs', kind: 'physics', label: 'GFS', long: 'NOAA GFS', step: 3, c: ['#e9a33b', '#c45000', '#f5c47a', '#ff8a3d'] },
  { id: 'icon', kind: 'physics', label: 'ICON', long: 'DWD ICON', step: 3, c: ['#cdbb45', '#7d6e00', '#e6d886', '#d2bd3a'] },
  { id: 'aifs', kind: 'ai', label: 'AIFS', long: 'ECMWF AIFS', step: 6, c: ['#dc8fbf', '#a3307a', '#f0b3d8', '#e070b4'] },
  { id: 'aigfs', kind: 'ai', label: 'AIGFS', long: 'NOAA AIGFS', step: 6, c: ['#c8a27c', '#87562a', '#dcc0a0', '#c99563'] },
];
const BLEND_COLOURS = ['#00876a', '#2cc79f'];

/** Every model the site knows, raw/corrected pairs per family, then the blend. */
const ALL_MODELS = [
  ...REGISTRY.flatMap((r) => {
    const base = { family: r.id, kind: r.kind, ai: r.kind === 'ai', step: r.step };
    return [
      { ...base, id: r.id, label: r.label, long: `${r.long} (raw)`, raw: true },
      { ...base, id: `${r.id}_bc`, label: `${r.label} corr.`, long: `${r.long}, bias-corrected`, raw: false },
    ];
  }),
  { id: 'blend', label: 'Blend', long: 'Blend of all corrected models', raw: false, family: 'blend', kind: 'blend', ai: false, step: 3 },
];
/** Models shown on this page: all known ones until useModels(meta) narrows them (in place). */
export const MODELS = [...ALL_MODELS];
export const MODEL = Object.fromEntries(ALL_MODELS.map((m) => [m.id, m]));
export const corrected = (id) => `${id}_bc`;
export const rawOf = (id) => (id.endsWith('_bc') ? id.slice(0, -3) : id);

/** Keep only the models the data has (meta.models), in display order; returns MODELS. */
export function useModels(meta) {
  const have = new Set(Array.isArray(meta?.models) ? meta.models : []);
  const keep = ALL_MODELS.filter((m) => have.has(m.id));
  if (keep.length) MODELS.splice(0, MODELS.length, ...keep);
  return MODELS;
}
/** Plain-text name for <option>s and canvas labels: "ECMWF AIFS (raw) · AI". */
export function modelText(id, long = false) {
  const m = MODEL[id];
  if (!m) return id;
  return `${long ? m.long : m.label}${m.ai ? ' · AI' : ''}`;
}
/** HTML name with an "AI" badge for machine-learning models. */
export function modelName(id, long = false) {
  const m = MODEL[id];
  if (!m) return esc(id);
  return `${esc(long ? m.long : m.label)}${m.ai ? '<span class="ai-badge" title="Machine-learning (AI) model">AI</span>' : ''}`;
}

// Colours and the AI badge style are generated from REGISTRY, so a new model needs no CSS edit.
(function injectModelStyle() {
  if (typeof document === 'undefined' || document.getElementById('model-style')) return;
  const vars = (dark) => [
    ...REGISTRY.flatMap((r) => [`--m-${r.id}:${r.c[dark ? 2 : 0]}`, `--m-${r.id}_bc:${r.c[dark ? 3 : 1]}`]),
    `--m-blend:${BLEND_COLOURS[dark ? 1 : 0]}`,
  ].join(';');
  const el = document.createElement('style');
  el.id = 'model-style';
  el.textContent = `:root{${vars(false)}}@media (prefers-color-scheme: dark){:root{${vars(true)}}}
.ai-badge{display:inline-block;margin-left:4px;padding:0 4px;border-radius:4px;border:1px solid currentColor;font-size:9px;font-weight:700;line-height:13px;letter-spacing:.04em;vertical-align:1px;color:var(--muted)}`;
  document.head.appendChild(el);
}());

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

/* ------------------------------------------------------------------ header
 * Every page shares the same top bar (see index.html for the markup):
 *   <header class="topbar"> <a class="brand">…<p class="status" id="status">
 *     <span class="live-dot"></span><span id="updated"></span><span class="sep">·</span><span id="count"></span></p></a>
 *   <nav class="nav" aria-label="Pages"></nav> … </header>
 *
 * renderNav(active)  fills the topbar's <nav class="nav"> with PAGES (creating it
 *                    after .brand when missing) and marks the page whose key is
 *                    `active` ('map' | 'analysis' | 'scores' | 'tafs' | 'about')
 *                    with aria-current="page". Keep a static copy of the links in
 *                    the HTML as a no-JS fallback; this call just rewrites it.
 * initHeader({ active, meta, extra })  renderNav(active) + setStatus(meta, extra)
 *                    when meta is given (or call setStatus later, after loadMeta()).
 * setStatus(meta, extra)  "Updated X min ago" = when the site was built
 *                    (meta.generated); the tooltip adds the latest observation time
 *                    (meta.last_obs). Re-renders itself every minute.
 */
export const PAGES = [
  { key: 'map', href: './', label: 'Map' },
  { key: 'forecast', href: 'forecast.html', label: 'Forecast' },
  { key: 'analysis', href: 'analysis.html', label: 'Analysis' },
  { key: 'scores', href: 'scores.html', label: 'Scores' },
  { key: 'tafs', href: 'tafs.html', label: 'TAFs' },
  { key: 'about', href: 'about.html', label: 'About' },
];
export function renderNav(active) {
  const bar = document.querySelector('.topbar');
  if (!bar) return null;
  let nav = bar.querySelector('nav.nav');
  if (!nav) {
    nav = document.createElement('nav');
    nav.className = 'nav';
    nav.setAttribute('aria-label', 'Pages');
    const brand = bar.querySelector('.brand');
    if (brand) brand.after(nav); else bar.prepend(nav);
  }
  nav.innerHTML = PAGES.map((p) => `<a href="${p.href}"${p.key === active ? ' aria-current="page"' : ''}>${esc(p.label)}</a>`).join('');
  // the top nav is full on phones: the status page is linked from the page footer
  const foot = document.querySelector('.foot p:last-of-type');
  if (foot && active !== 'status' && !foot.querySelector('a[href="status.html"]')) foot.insertAdjacentHTML('beforeend', ' · <a href="status.html">Pipeline status</a>');
  return nav;
}
export function initHeader({ active, meta, extra } = {}) {
  renderNav(active);
  if (meta !== undefined) setStatus(meta, extra);
}

const BUILD_STALE_MS = 3 * 3600000; // site not rebuilt for 3 h: the hourly job is stuck
const OBS_STALE_MS = 3 * 3600000; // newest observation 3 h older than the build: collection is stuck
let statusArgs = null;
let statusTimer = 0;
function fullTime(d) { return d ? `${dateLabel(d)} ${utcLabel(d)}` : '–'; }

export function setStatus(meta, extra = '') {
  statusArgs = [meta, extra];
  if (!statusTimer) statusTimer = setInterval(() => statusArgs && renderStatus(...statusArgs), 60000);
  renderStatus(meta, extra);
}
function renderStatus(meta, extra) {
  const el = document.getElementById('updated');
  const st = document.getElementById('status');
  if (!el) return;
  if (!meta) { el.textContent = 'No data yet'; st?.classList.add('stale'); return; }
  const gen = parseTime(meta.generated);
  const obs = parseTime(meta.last_obs);
  const now = Date.now();
  const buildStale = !gen || now - gen > BUILD_STALE_MS;
  const obsStale = !!(gen && obs && gen - obs > OBS_STALE_MS);
  el.textContent = `Updated ${ago(gen, now)}${obsStale ? ' · obs delayed' : ''}`;
  const tip = [
    `Site built ${fullTime(gen)} (${ago(gen, now)})`,
    obs ? `Latest obs ${fullTime(obs)} (${ago(obs, now)})` : 'No observations yet',
  ];
  if (buildStale) tip.push('The site has not been rebuilt for a while — data may be out of date.');
  if (obsStale) tip.push('Observation collection is behind the site build.');
  (st || el).title = tip.join('\n');
  el.removeAttribute('title');
  st?.classList.toggle('stale', buildStale || obsStale);
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
