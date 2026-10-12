// StationTruth — TAF vs reality page (data: data/taf_scores.json, schema in docs/TAF.md)
import {
  isNum, esc, fmt, fmtInt, fmtCompact, getJSON, loadMeta, initHeader, setStatus, dateLabel,
  cssVar, onThemeChange, axisStyle, observeWidth, seg,
} from './common.js?v=__BUILD__';

const $ = (id) => document.getElementById(id);
const CATS = ['VFR', 'MVFR', 'IFR', 'LIFR'];
const LEAD_X = { '0-6': 3, '6-12': 9, '12-24': 18, '24-30': 27 }; // bucket midpoints on the chart
const LEAD_MIN_N = 30; // buckets with fewer verified hours are left out of the charts
const RANK_MIN_EVENTS = 5; // a station needs this many hits+misses+false alarms to be ranked
const REGION_ORDER = ['Europe', 'North America', 'Central America & Caribbean', 'South America', 'Africa', 'Middle East', 'Asia', 'Oceania', 'Polar'];
const SCORES = [
  { k: 'csi', label: 'CSI', colour: '--s-csi' },
  { k: 'pod', label: 'POD', colour: '--s-pod' },
  { k: 'far', label: 'FAR', colour: '--s-far' },
];

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { win: p.get('win'), region: p.get('region'), obs: p.get('obs'), tempo: p.get('tempo') };
}
const h = readHash();
const state = {
  meta: null,
  data: null,
  win: h.win === '30d' ? '30d' : '7d',
  region: h.region || 'global',
  obs: h.obs === 'nominal' ? 'nominal' : 'worst',
  tempo: h.tempo === '1' ? 'tempo' : 'prev',
};
const charts = [];

function writeHash() {
  const p = new URLSearchParams();
  if (state.win !== '7d') p.set('win', state.win);
  if (state.region !== 'global') p.set('region', state.region);
  if (state.obs !== 'worst') p.set('obs', state.obs);
  if (state.tempo === 'tempo') p.set('tempo', '1');
  const s = p.toString();
  history.replaceState(null, '', s ? `#${s}` : location.pathname + location.search);
}

const win = () => state.data?.windows?.[state.win] || null;
function block() {
  const w = win();
  if (!w) return null;
  const r = state.region === 'global' ? w.global : w.regions?.[state.region];
  return r?.[state.obs] || null;
}
const evKey = () => (state.tempo === 'tempo' ? 'ifr_tempo' : 'ifr');
const regionLabel = (r) => (r === 'global' ? 'Global' : r);
const pct = (v, d = 0) => (isNum(v) ? `${(v * 100).toFixed(d)}%` : '–');
const ratio = (v) => (isNum(v) ? v.toFixed(2) : '–');

/* ------------------------------------------------------------------ banner */
function bannerHTML() {
  const d = state.data;
  const first = d?.first_issued ? dateLabel(d.first_issued) : null;
  if (!d || !d.taf_count) {
    return `<div class="collecting" role="status"><b>Collecting TAFs${first ? ` since ${esc(first)}` : ''}.</b> TAF collection has just started — the first scores appear once the forecast hours have been observed. Check back in a few hours.</div>`;
  }
  const w = win();
  if (!w) {
    return `<div class="collecting" role="status"><b>Collecting TAFs since ${esc(first)}.</b> ${fmtInt(d.taf_count)} TAFs from ${fmtInt(d.station_count)} airports stored; waiting for the forecast hours to be observed.</div>`;
  }
  const days = (new Date(w.end) - new Date(w.start)) / 86400000;
  if (days >= 3) return '';
  const pctDone = Math.min(100, (days / 7) * 100);
  return `<div class="collecting" role="status">
    <div><b>Collecting TAFs since ${esc(first)}</b> — scores settle down after a few days.</div>
    <div class="prog" aria-hidden="true"><span style="width:${pctDone.toFixed(1)}%"></span></div>
    <div class="prog-txt">${days < 1 ? `${Math.round(days * 24)} hours` : `${days.toFixed(1)} days`} verified · ${fmtInt(d.taf_count)} TAFs</div>
  </div>`;
}

/* ------------------------------------------------------------------ KPIs */
function renderKpis(b) {
  if (!b || !b.n) { $('kpis').innerHTML = ''; return; }
  const e = b[evKey()];
  const other = b[state.tempo === 'tempo' ? 'ifr' : 'ifr_tempo'];
  const events = e.a + e.c;
  const tempoNote = state.tempo === 'tempo' ? 'TEMPO/PROB counted' : 'prevailing forecast only';
  const cards = [
    `<div class="kpi"><div class="k">Right category</div><div class="v">${pct(b.hit_rate)}</div><div class="s">of ${fmtCompact(b.n)} forecast hours · Heidke skill ${ratio(b.hss)}</div></div>`,
    `<div class="kpi"><div class="k">IFR caught (POD)</div><div class="v">${pct(e.pod)}</div><div class="s">${fmtInt(e.a)} of ${fmtInt(events)} IFR-or-worse hours · ${tempoNote}</div></div>`,
    `<div class="kpi"><div class="k">False alarms (FAR)</div><div class="v">${pct(e.far)}</div><div class="s">${fmtInt(e.b)} of ${fmtInt(e.a + e.b)} forecast IFR hours were better</div></div>`,
    `<div class="kpi"><div class="k">CSI · IFR or worse</div><div class="v">${ratio(e.csi)}</div><div class="s">${state.tempo === 'tempo' ? 'without' : 'with'} TEMPO/PROB: ${ratio(other.csi)} · freq. bias ${ratio(e.bias)}</div></div>`,
  ];
  $('kpis').innerHTML = cards.join('');
}

/* ------------------------------------------------------------------ contingency table */
function renderTable(b) {
  if (!b || !b.n) {
    $('ct').innerHTML = '<div class="empty-note">No verified TAF hours for this selection yet.</div>';
    $('ct-note').textContent = '';
    return;
  }
  const t = b.table;
  const rowTot = t.map((r) => r.reduce((x, y) => x + y, 0));
  const colTot = CATS.map((_, j) => t.reduce((x, r) => x + r[j], 0));
  const accent = cssVar('--accent') || '#0b6bcb';
  const cell = (v, i, j) => {
    const share = rowTot[i] ? v / rowTot[i] : 0;
    const bg = `color-mix(in srgb, ${accent} ${Math.round(share * 55)}%, var(--surface-2))`;
    return `<td class="${i === j ? 'diag' : ''}" style="background:${bg}" title="Forecast ${CATS[i]}, observed ${CATS[j]}: ${fmtInt(v)} hours (${pct(share, 1)} of forecast ${CATS[i]})"><span class="c">${fmtCompact(v)}</span><span class="p">${rowTot[i] ? pct(share) : '–'}</span></td>`;
  };
  $('ct').innerHTML = `<table class="ct">
    <thead>
      <tr><th></th><th class="axis" colspan="4">Observed →</th><th class="totc"></th></tr>
      <tr><th class="axis">Forecast ↓</th>${CATS.map((c) => `<th scope="col"><span class="cat ${c}">${c}</span></th>`).join('')}<th scope="col" class="totc">Total</th></tr>
    </thead>
    <tbody>${t.map((r, i) => `<tr><th scope="row"><span class="cat ${CATS[i]}">${CATS[i]}</span></th>${r.map((v, j) => cell(v, i, j)).join('')}<td class="tot totc">${fmtCompact(rowTot[i])}</td></tr>`).join('')}
      <tr><th scope="row">Total</th>${colTot.map((v) => `<td class="tot">${fmtCompact(v)}</td>`).join('')}<td class="tot totc">${fmtCompact(b.n)}</td></tr>
    </tbody>
  </table>
  <div class="ct-legend"><span><i style="box-shadow:inset 0 0 0 2px var(--good)"></i>correct category</span><span>percentages: share of each forecast row (how often “forecast X” turned out to be each category)</span></div>`;
  $('ct-note').textContent = `${regionLabel(state.region)} · ${state.win === '7d' ? 'last 7' : 'last 30'} days · prevailing TAF category vs ${state.obs === 'worst' ? 'the worst category reported during the hour' : 'the routine METAR nearest the hour'}. All lead times.`;
}

/* ------------------------------------------------------------------ charts */
function leadSeries(b, get) {
  const rows = (b?.by_lead || []).filter((r) => LEAD_X[r.lead] != null);
  const x = rows.map((r) => LEAD_X[r.lead]);
  return { x, rows, y: (fn) => rows.map((r) => (r.n >= LEAD_MIN_N && isNum(fn(r)) ? fn(r) : null)), get };
}

function makeChart(el, b, series, tipFmt) {
  el.innerHTML = '';
  const { x, rows } = leadSeries(b);
  const ys = series.map((s) => rows.map((r) => (r.n >= LEAD_MIN_N && isNum(s.get(r)) ? s.get(r) : null)));
  if (!x.length || !ys.some((a) => a.some(isNum))) {
    el.innerHTML = '<div class="empty-note" style="margin-top:40px">Not enough verified hours yet.</div>';
    return null;
  }
  const ax = axisStyle();
  const tipEl = document.createElement('div');
  tipEl.className = 'chart-tip';
  tipEl.hidden = true;
  const opts = {
    width: el.clientWidth || 480,
    height: 250,
    legend: { show: false },
    cursor: { y: false, points: { size: 8 }, drag: { x: false, y: false } },
    scales: { x: { time: false, range: () => [0, 30] }, y: { range: () => [0, 1] } },
    axes: [
      { ...ax, size: 34, label: 'Hours after the TAF was issued', labelSize: 16, labelGap: 0, labelFont: ax.font, splits: () => [0, 6, 12, 24, 30], values: (u, v) => v.map((t) => `${t} h`) },
      { ...ax, size: 40, splits: () => [0, 0.25, 0.5, 0.75, 1], values: (u, v) => v.map((t) => t.toFixed(2)) },
    ],
    series: [{}, ...series.map((s) => ({
      label: s.label, stroke: cssVar(s.colour), width: s.dash ? 1.8 : 2.4, dash: s.dash ? [5, 4] : undefined,
      spanGaps: true, points: { show: true, size: 8, width: 1, fill: cssVar(s.colour), stroke: cssVar('--surface-solid') },
    }))],
    hooks: {
      setCursor: [(u) => {
        const i = u.cursor.idx;
        if (i == null) { tipEl.hidden = true; return; }
        const r = rows[i];
        const lines = series.map((s, k) => (isNum(u.data[k + 1][i])
          ? `<div class="tr"><span><i class="sw-line ${s.dash ? 'raw' : ''}" style="--c:var(${s.colour})"></i>${esc(s.label)}</span><b>${tipFmt(u.data[k + 1][i])}</b></div>` : '')).join('');
        if (!lines) { tipEl.hidden = true; return; }
        tipEl.innerHTML = `<div class="tt">Lead ${esc(r.lead)} h · ${fmtCompact(r.n)} hours</div>${lines}`;
        tipEl.hidden = false;
        const left = u.cursor.left + u.bbox.left / (devicePixelRatio || 1);
        const w = tipEl.offsetWidth; const W = el.clientWidth;
        tipEl.style.left = `${left + 14 + w > W ? Math.max(0, left - 14 - w) : left + 14}px`;
      }],
    },
  };
  const u = new uPlot(opts, [x, ...ys], el);
  el.appendChild(tipEl);
  el.addEventListener('mouseleave', () => { tipEl.hidden = true; });
  const ro = observeWidth(el, (w) => u.setSize({ width: w, height: opts.height }));
  return { u, ro };
}

function renderKeys() {
  $('ifr-key').innerHTML = SCORES.map((s) => `<span><i class="sw-line" style="--c:var(${s.colour})"></i>${s.label}</span>`).join('')
    + '<span><i class="solid"></i>prevailing</span><span><i class="dash"></i>with TEMPO/PROB</span>';
  $('hit-key').innerHTML = '<span><i class="sw-line" style="--c:var(--s-hit)"></i>Hit rate</span><span><i class="sw-line" style="--c:var(--s-hss)"></i>Heidke skill</span>';
}

function renderCharts(b) {
  while (charts.length) { const c = charts.pop(); c.ro.disconnect(); c.u.destroy(); }
  if (typeof uPlot === 'undefined') return;
  const ifrSeries = [];
  for (const s of SCORES) {
    ifrSeries.push({ ...s, label: `${s.label}`, get: (r) => r.ifr?.[s.k] });
    ifrSeries.push({ ...s, label: `${s.label} + TEMPO`, dash: true, get: (r) => r.ifr_tempo?.[s.k] });
  }
  const c1 = makeChart($('chart-ifr'), b, ifrSeries, (v) => v.toFixed(2));
  if (c1) charts.push(c1);
  const c2 = makeChart($('chart-hit'), b, [
    { label: 'Hit rate', colour: '--s-hit', get: (r) => r.hit_rate },
    { label: 'Heidke skill', colour: '--s-hss', get: (r) => r.hss },
  ], (v) => v.toFixed(2));
  if (c2) charts.push(c2);
  const ns = (b?.by_lead || []).map((r) => r.n);
  $('lead-note').textContent = ns.length && Math.max(...ns)
    ? `Verified hours per lead bucket: ${(b.by_lead || []).map((r) => `${r.lead} h: ${fmtCompact(r.n)}`).join(' · ')}. Buckets with fewer than ${LEAD_MIN_N} hours are left out. Most TAFs cover 24 or 30 hours, so the last bucket is smaller.`
    : '';
}

function renderLeadTable(b) {
  const rows = b?.by_lead || [];
  if (!rows.length || !b.n) { $('lead-table').innerHTML = '<div class="empty-note">No verified hours yet.</div>'; return; }
  const k = evKey();
  const line = (label, r, e) => `<tr class="${r.n < LEAD_MIN_N ? 'thin' : ''}"><td>${label}</td><td>${fmtInt(r.n)}</td><td>${pct(r.hit_rate)}</td><td>${ratio(r.hss)}</td><td>${fmtInt(e.a + e.c)}</td><td>${ratio(e.pod)}</td><td>${ratio(e.far)}</td><td>${ratio(e.csi)}</td><td>${ratio(e.bias)}</td></tr>`;
  $('lead-table').innerHTML = `<table class="tbl">
    <thead><tr><th scope="col">Lead</th><th scope="col">Hours</th><th scope="col">Hit rate</th><th scope="col">Heidke</th><th scope="col" title="Observed IFR-or-worse hours">IFR hrs</th><th scope="col">POD</th><th scope="col">FAR</th><th scope="col">CSI</th><th scope="col">Bias</th></tr></thead>
    <tbody>${rows.map((r) => line(`${esc(r.lead)} h`, r, r[k])).join('')}${line('<b>All</b>', b, b[k])}</tbody>
    <caption>IFR-or-worse scores ${state.tempo === 'tempo' ? 'counting TEMPO/PROB groups' : 'for the prevailing forecast only'} (switch above).</caption>
  </table>`;
}

/* ------------------------------------------------------------------ station ranking */
// Station rows are arrays in data.station_fields order; derive POD/FAR/CSI here.
function stationRows() {
  const d = state.data;
  const f = d?.station_fields || [];
  const ix = Object.fromEntries(f.map((k, i) => [k, i]));
  const tempo = state.tempo === 'tempo';
  return (win()?.stations || []).map((r) => {
    const a = r[ix[tempo ? 'a_t' : 'a']]; const b = r[ix[tempo ? 'b_t' : 'b']]; const c = r[ix[tempo ? 'c_t' : 'c']];
    const [name, country, region] = d.stations?.[r[ix.id]] || ['', '', null];
    return {
      id: r[ix.id], name, country, region, n: r[ix.n], a, b, c,
      pod: a + c ? a / (a + c) : null, far: a + b ? b / (a + b) : null, csi: a + b + c ? a / (a + b + c) : null,
    };
  });
}

function renderRanking() {
  const minN = state.data?.station_min_n || 48;
  const list = stationRows().filter((s) => (state.region === 'global' || s.region === state.region)
    && s.n >= minN && isNum(s.csi) && (s.a + s.b + s.c) >= RANK_MIN_EVENTS);
  $('rank-note').textContent = `Airports with at least ${minN} verified hours and ${RANK_MIN_EVENTS} IFR-or-worse hits, misses or false alarms (${fmtInt(list.length)} in ${regionLabel(state.region)}). Compared with the worst category in each hour; ${state.tempo === 'tempo' ? 'TEMPO/PROB groups counted' : 'prevailing forecast only'}. Small samples swing a lot — treat single airports as anecdotes.`;
  const table = (rows) => {
    if (!rows.length) return '<div class="empty-note">Not enough airports with IFR events yet.</div>';
    return `<table class="tbl"><thead><tr><th scope="col">Airport</th><th scope="col">Name</th><th scope="col" title="Observed IFR-or-worse hours">IFR hrs</th><th scope="col">POD</th><th scope="col">FAR</th><th scope="col">CSI</th></tr></thead><tbody>${rows.map((s) => `<tr><td><a class="icao" href="./#${encodeURIComponent(s.id)}" title="Open on the map">${esc(s.id)}</a></td><td title="${esc(s.name)}${s.country ? `, ${esc(s.country)}` : ''}">${esc(s.name || '–')}</td><td>${fmtInt(s.a + s.c)}</td><td>${ratio(s.pod)}</td><td>${ratio(s.far)}</td><td><b>${ratio(s.csi)}</b><span class="bar" style="width:${Math.round(s.csi * 40)}px" aria-hidden="true"></span></td></tr>`).join('')}</tbody></table>`;
  };
  const byCsi = [...list].sort((p, q) => q.csi - p.csi || (q.a + q.c) - (p.a + p.c));
  $('rank-top').innerHTML = table(byCsi.slice(0, 15));
  $('rank-bottom').innerHTML = byCsi.length > 15
    ? table(byCsi.slice(Math.max(15, byCsi.length - 15)).reverse())
    : '<div class="empty-note">Needs more than 15 ranked airports.</div>';
}

/* ------------------------------------------------------------------ render */
function render() {
  const b = block();
  $('collecting').innerHTML = bannerHTML();
  renderKpis(b);
  renderTable(b);
  renderCharts(b);
  renderLeadTable(b);
  renderRanking();
}

seg($('win-seg'), state.win, (v) => { state.win = v; writeHash(); render(); });
seg($('obs-seg'), state.obs, (v) => { state.obs = v; writeHash(); render(); });
seg($('tempo-seg'), state.tempo, (v) => { state.tempo = v; writeHash(); render(); });
const regionSel = $('region-sel');
regionSel.addEventListener('change', () => { state.region = regionSel.value; writeHash(); render(); });
onThemeChange(() => render());

async function boot() {
  initHeader({ active: 'tafs' });
  renderKeys();
  try {
    state.meta = await loadMeta();
    state.data = await getJSON('data/taf_scores.json', { optional: true });
  } catch (err) {
    console.error(err);
    $('collecting').innerHTML = `<div class="collecting" role="alert">Couldn't load the TAF scores (${esc(err.message)}).</div>`;
  }
  setStatus(state.meta, state.data?.taf_count ? `${fmtInt(state.data.station_count)} TAF airports` : '');
  const regions = new Set();
  for (const w of Object.values(state.data?.windows || {})) for (const r of Object.keys(w.regions || {})) regions.add(r);
  const list = [...regions].sort((a, b) => (REGION_ORDER.indexOf(a) + 1 || 99) - (REGION_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b));
  regionSel.innerHTML = `<option value="global">Global</option>${list.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join('')}`;
  if (state.region !== 'global' && !regions.has(state.region)) state.region = 'global';
  regionSel.value = state.region;
  render();
}
boot();
