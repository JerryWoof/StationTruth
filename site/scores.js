// StationTruth — scoreboard page
import {
  MODELS, MODEL, VARS, isNum, esc, fmt, fmtSigned, fmtInt, fmtCompact,
  getJSON, loadMeta, collectingHTML, setStatus, cssVar, modelColour, onThemeChange, axisStyle, observeWidth, seg, legendSwatch,
} from './common.js';

const $ = (id) => document.getElementById(id);
const LEAD_MIN_N = 30; // leads with fewer pairs are left out of the charts
const ROW_MIN_N = 100; // summary rows with fewer pairs are greyed out
const SCORE_VARS = ['t', 'td', 'wspd', 'gust', 'mslp', 'wdir'];

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { win: p.get('win'), region: p.get('region'), v: p.get('var') };
}
const h = readHash();
const state = {
  meta: null,
  scores: null,
  win: h.win === '30d' ? '30d' : '7d',
  region: h.region || 'global',
  v: SCORE_VARS.includes(h.v) ? h.v : 't',
  hidden: new Set(),
};
const charts = [];

function writeHash() {
  const p = new URLSearchParams();
  if (state.win !== '7d') p.set('win', state.win);
  if (state.region !== 'global') p.set('region', state.region);
  if (state.v !== 't') p.set('var', state.v);
  const s = p.toString();
  history.replaceState(null, '', s ? `#${s}` : location.pathname + location.search);
}

function block() {
  const w = state.scores?.windows?.[state.win];
  if (!w) return null;
  return state.region === 'global' ? w.global : w.regions?.[state.region];
}

/* ------------------------------------------------------------------ KPIs */
function renderKpis(b) {
  const V = VARS[state.v];
  const all = (m) => b?.[m]?.[state.v]?.all;
  const ok = MODELS.filter((m) => { const a = all(m.id); return a && a.n >= ROW_MIN_N && isNum(a.rmse); });
  const cards = [];
  if (ok.length) {
    const best = ok.reduce((x, y) => (all(y.id).rmse < all(x.id).rmse ? y : x));
    cards.push(`<div class="kpi"><div class="k">Most accurate</div><div class="v">${legendSwatch(best.id)}${esc(best.label)}</div><div class="s">RMSE ${fmt(all(best.id).rmse)} ${esc(V.unit)}, all lead times</div></div>`);
    for (const [raw, bc] of [['ecmwf', 'ecmwf_bc'], ['gfs', 'gfs_bc']]) {
      const a = all(raw); const c = all(bc);
      if (a && c && a.n >= ROW_MIN_N && c.n >= ROW_MIN_N && a.rmse > 0) {
        const pct = ((c.rmse - a.rmse) / a.rmse) * 100;
        const good = pct < -0.5;
        cards.push(`<div class="kpi"><div class="k">Correction · ${esc(MODEL[raw].label)}</div><div class="v" style="color:${good ? 'var(--good)' : 'var(--text)'}">${pct <= 0 ? '−' : '+'}${Math.abs(pct).toFixed(0)}% RMSE</div><div class="s">${fmt(a.rmse)} → ${fmt(c.rmse)} ${esc(V.unit)}${good ? '' : ' · not helping yet'}</div></div>`);
      }
    }
  }
  const n = Math.max(0, ...MODELS.map((m) => all(m.id)?.n || 0));
  cards.push(`<div class="kpi"><div class="k">Sample</div><div class="v">${fmtCompact(n)} pairs</div><div class="s">${esc(regionLabel(state.region))} · ${esc(V.label.toLowerCase())} · ${state.win === '7d' ? '7' : '30'} days</div></div>`);
  $('kpis').innerHTML = cards.join('');
}
const regionLabel = (r) => (r === 'global' ? 'Global' : r);

/* ------------------------------------------------------------------ table */
function renderTable(b) {
  const V = VARS[state.v];
  const rows = MODELS.map((m) => [m, b?.[m.id]?.[state.v]?.all]);
  const ok = rows.filter(([, a]) => a && a.n >= ROW_MIN_N);
  if (!rows.some(([, a]) => a && a.n)) {
    $('summary').innerHTML = `<div class="empty-note">No verified pairs for ${esc(V.label.toLowerCase())} in ${esc(regionLabel(state.region))} over this window yet.</div>`;
    return;
  }
  const best = (k, f = (x) => x) => (ok.length ? Math.min(...ok.map(([, a]) => (isNum(a[k]) ? f(a[k]) : Infinity))) : NaN);
  const bB = best('bias', Math.abs); const bM = best('mae'); const bR = best('rmse');
  const cell = (a, k, bv, f = (x) => x, signed = false) => {
    if (!a || !isNum(a[k])) return '<td class="muted">–</td>';
    const isBest = a.n >= ROW_MIN_N && Math.abs(f(a[k]) - bv) < 1e-9;
    return `<td class="${isBest ? 'best' : ''}">${signed ? fmtSigned(a[k], V.digits) : fmt(a[k], V.digits)}</td>`;
  };
  const vsRaw = (m, a) => {
    if (m.raw || !a || !isNum(a.rmse)) return '<td class="muted">–</td>';
    const refs = m.id === 'blend' ? ['ecmwf', 'gfs'] : [m.id.replace('_bc', '')];
    const ref = Math.min(...refs.map((r) => b?.[r]?.[state.v]?.all?.rmse).filter(isNum));
    if (!Number.isFinite(ref) || ref <= 0) return '<td class="muted">–</td>';
    const pct = ((a.rmse - ref) / ref) * 100;
    return `<td style="color:${pct < -0.5 ? 'var(--good)' : pct > 0.5 ? 'var(--warm)' : 'var(--muted)'};font-weight:600">${pct <= 0 ? '−' : '+'}${Math.abs(pct).toFixed(1)}%</td>`;
  };
  $('summary').innerHTML = `<table class="tbl">
    <thead><tr><th scope="col">Model</th><th scope="col">n</th><th scope="col">Bias</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col" title="RMSE change vs the raw model (blend: vs the better raw model)">vs raw</th></tr></thead>
    <tbody>${rows.map(([m, a]) => `<tr class="${!a || a.n < ROW_MIN_N ? 'thin' : ''}">
      <td><span class="mname">${legendSwatch(m.id)}<span class="lname">${esc(m.long)}</span><span class="sname">${esc(m.label)}</span></span></td><td>${fmtInt(a?.n ?? 0)}</td>
      ${cell(a, 'bias', bB, Math.abs, true)}${cell(a, 'mae', bM)}${cell(a, 'rmse', bR)}${vsRaw(m, a)}</tr>`).join('')}</tbody>
    <caption>${esc(regionLabel(state.region))} · ${esc(V.label)} in ${esc(V.unit)} · all lead times 0–72 h · ★ best (smallest |bias|, MAE, RMSE among rows with ≥ ${ROW_MIN_N} pairs).</caption>
  </table>`;
}

/* ------------------------------------------------------------------ charts */
function renderKey() {
  $('model-key').innerHTML = MODELS.map((m) => `<button type="button" class="key-btn" data-m="${m.id}" aria-pressed="${!state.hidden.has(m.id)}">${legendSwatch(m.id)}${esc(m.long)}</button>`).join('');
}
$('model-key').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-m]');
  if (!btn) return;
  const m = btn.dataset.m;
  if (state.hidden.has(m)) state.hidden.delete(m); else state.hidden.add(m);
  btn.setAttribute('aria-pressed', String(!state.hidden.has(m)));
  const i = MODELS.findIndex((x) => x.id === m) + 1;
  for (const c of charts) c.u.setSeries(i, { show: !state.hidden.has(m) });
});

function leadData(b, key) {
  const leads = new Set();
  for (const m of MODELS) for (const r of b?.[m.id]?.[state.v]?.by_lead || []) leads.add(r.lead_h);
  const x = [...leads].sort((p, q) => p - q);
  const ys = MODELS.map((m) => {
    const by = new Map((b?.[m.id]?.[state.v]?.by_lead || []).map((r) => [r.lead_h, r]));
    return x.map((l) => { const r = by.get(l); return r && r.n >= LEAD_MIN_N && isNum(r[key]) ? r[key] : null; });
  });
  const ns = MODELS.map((m) => {
    const by = new Map((b?.[m.id]?.[state.v]?.by_lead || []).map((r) => [r.lead_h, r.n]));
    return x.map((l) => by.get(l) ?? 0);
  });
  return { x, ys, ns };
}

function makeChart(el, b, key) {
  const V = VARS[state.v];
  const { x, ys, ns } = leadData(b, key);
  el.innerHTML = '';
  if (!x.length || !ys.some((a) => a.some(isNum))) {
    el.innerHTML = '<div class="empty-note" style="margin-top:40px">Not enough pairs yet for this selection.</div>';
    return null;
  }
  const ax = axisStyle();
  const zeroCol = cssVar('--border-strong');
  const tipEl = document.createElement('div');
  tipEl.className = 'chart-tip';
  tipEl.hidden = true;
  const opts = {
    width: el.clientWidth || 480,
    height: 260,
    legend: { show: false },
    cursor: { y: false, points: { size: 7 }, drag: { x: false, y: false } },
    scales: { x: { time: false, range: (u, lo, hi) => [0, Math.max(hi, 24)] }, y: key === 'rmse' ? { range: (u, lo, hi) => [0, hi * 1.08 || 1] } : { range: (u, lo, hi) => { const m = Math.max(Math.abs(lo), Math.abs(hi), 0.2) * 1.15; return [-m, m]; } } },
    axes: [
      { ...ax, size: 34, label: 'Lead time (hours after model start)', labelSize: 16, labelGap: 0, labelFont: ax.font, incrs: [3, 6, 12, 24], values: (u, v) => v.map((t) => `${t} h`) },
      { ...ax, size: 46, label: V.unit, labelSize: 14, labelFont: ax.font, values: (u, v) => v.map((t) => (key === 'bias' ? fmtSigned(t, V.digits === 0 ? 0 : 1) : fmt(t, V.digits === 0 ? 0 : 1))) },
    ],
    series: [{}, ...MODELS.map((m) => ({
      label: m.label, stroke: modelColour(m.id), width: m.raw ? 1.6 : 2.4, dash: m.raw ? [5, 4] : undefined,
      spanGaps: true, show: !state.hidden.has(m.id), points: { show: true, size: m.raw ? 4 : 5, width: 1, fill: modelColour(m.id) },
    }))],
    hooks: {
      draw: key === 'bias' ? [(u) => {
        const y0 = u.valToPos(0, 'y', true); const { left, width } = u.bbox; const ctx = u.ctx;
        ctx.save(); ctx.strokeStyle = zeroCol; ctx.lineWidth = 1.5 * (devicePixelRatio || 1);
        ctx.beginPath(); ctx.moveTo(left, y0); ctx.lineTo(left + width, y0); ctx.stroke(); ctx.restore();
      }] : [],
      setCursor: [(u) => {
        const i = u.cursor.idx;
        if (i == null) { tipEl.hidden = true; return; }
        const rows = MODELS.map((m, k) => (u.series[k + 1].show && isNum(u.data[k + 1][i])
          ? `<div class="tr"><span>${legendSwatch(m.id)}${esc(m.label)}</span><b>${key === 'bias' ? fmtSigned(u.data[k + 1][i]) : fmt(u.data[k + 1][i])} <span class="muted" style="font-weight:500">n ${fmtCompact(ns[k][i])}</span></b></div>` : '')).join('');
        if (!rows) { tipEl.hidden = true; return; }
        tipEl.innerHTML = `<div class="tt">+${x[i]} h · ${esc(V.unit)}</div>${rows}`;
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

function renderCharts(b) {
  while (charts.length) { const c = charts.pop(); c.ro.disconnect(); c.u.destroy(); }
  if (typeof uPlot === 'undefined') return;
  for (const [id, key] of [['chart-rmse', 'rmse'], ['chart-bias', 'bias']]) {
    const c = makeChart($(id), b, key);
    if (c) charts.push(c);
  }
  // lead-time sample note
  let minN = Infinity; let maxN = 0;
  for (const m of MODELS) for (const r of b?.[m.id]?.[state.v]?.by_lead || []) { minN = Math.min(minN, r.n); maxN = Math.max(maxN, r.n); }
  $('lead-note').textContent = maxN
    ? `Pairs per lead time: ${fmtInt(minN)}–${fmtInt(maxN)} per model. Hover a chart for exact values and sample sizes.`
    : '';
}

function render() {
  const b = block();
  const V = VARS[state.v];
  $('sub-bias').textContent = `Mean of forecast − observed, ${V.unit}. Above zero: model ${state.v === 't' || state.v === 'td' ? 'too warm / moist' : 'too high'}; below: ${state.v === 't' || state.v === 'td' ? 'too cold / dry' : 'too low'}.${state.v === 'wdir' ? ' Positive = forecast direction veered (clockwise).' : ''}`;
  $('sub-rmse').textContent = `Root-mean-square error, ${V.unit}. Usually grows as the forecast reaches further ahead.`;
  renderKpis(b);
  renderCharts(b);
  renderTable(b);
}

/* ------------------------------------------------------------------ controls */
seg($('win-seg'), state.win, (v) => { state.win = v; writeHash(); render(); });
seg($('var-seg'), state.v, (v) => { state.v = v; writeHash(); render(); });
const regionSel = $('region-sel');
regionSel.addEventListener('change', () => { state.region = regionSel.value; writeHash(); render(); });
onThemeChange(() => render());

async function boot() {
  renderKey();
  try {
    state.meta = await loadMeta();
    setStatus(state.meta);
    $('collecting').innerHTML = collectingHTML(state.meta);
    state.scores = await getJSON('data/scores.json', { optional: true });
  } catch (err) {
    console.error(err);
    $('collecting').innerHTML = `<div class="collecting" role="alert">Couldn't load the scores (${esc(err.message)}).</div>`;
  }
  const regions = new Set();
  for (const w of Object.values(state.scores?.windows || {})) for (const r of Object.keys(w.regions || {})) regions.add(r);
  const order = ['Europe', 'North America', 'Central America & Caribbean', 'South America', 'Africa', 'Middle East', 'Asia', 'Oceania', 'Polar'];
  const list = [...regions].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99) || a.localeCompare(b));
  regionSel.innerHTML = `<option value="global">Global</option>${list.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join('')}`;
  if (state.region !== 'global' && !regions.has(state.region)) state.region = 'global';
  regionSel.value = state.region;
  if (!state.scores) {
    $('kpis').innerHTML = '';
    $('collecting').innerHTML = collectingHTML(state.meta, { always: true });
  }
  render();
}
boot();
