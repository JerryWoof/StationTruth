// StationTruth — scoreboard "Head to head" section (data/compare.json, see docs/DATA.md).
// Model ids are read from the data, so new models appear without code changes.
import {
  MODELS, VARS, isNum, esc, fmt, fmtInt, fmtCompact, getJSON, cssVar, modelText, modelName,
  onThemeChange, axisStyle, observeWidth, seg, legendSwatch,
} from './common.js?v=__BUILD__';

const $ = (id) => document.getElementById(id);
const H2H_VARS = ['t', 'td', 'wspd', 'gust', 'mslp'];
const BUCKET_LABEL = { '0-24': '0–24 h', '27-48': '27–48 h', '51-72': '51–72 h', all: '0–72 h' };
const REGION_ORDER = ['Europe', 'North America', 'Central America & Caribbean', 'South America', 'Africa', 'Middle East', 'Asia', 'Oceania', 'Polar'];
const FEW_DAYS = 7; // below this many days with data, intervals are flagged as rough

const state = { data: null, a: null, b: null, v: 't', win: '7d', region: 'global', metric: 'rmse' };
let chart = null;

/* ------------------------------------------------------------------ data access */
const F = {}; // field name -> index, from compare.json "fields"

function winBlock() {
  const w = state.data?.windows?.[state.win];
  if (!w) return null;
  return state.region === 'global' ? w.global : w.regions?.[state.region];
}

// Row for A vs B as an object with A-minus-B deltas; the file stores each pair once.
function pairRows(a, b, v = state.v) {
  const blk = winBlock();
  if (!blk || a === b) return null;
  let rows = blk[`${a}|${b}`]?.[v];
  let flip = false;
  if (!rows) { rows = blk[`${b}|${a}`]?.[v]; flip = true; }
  if (!rows) return null;
  return rows.map((r) => (r ? decode(r, flip) : null));
}
function decode(r, flip) {
  const g = (k) => r[F[k]];
  const neg = (x) => (isNum(x) ? -x : x);
  const o = {
    n: g('n'), stations: g('stations'), days: g('days'),
    rmse_a: g('rmse_a'), rmse_b: g('rmse_b'), d_rmse: g('d_rmse'), d_rmse_lo: g('d_rmse_lo'), d_rmse_hi: g('d_rmse_hi'),
    mae_a: g('mae_a'), mae_b: g('mae_b'), d_mae: g('d_mae'), d_mae_lo: g('d_mae_lo'), d_mae_hi: g('d_mae_hi'),
    frac: g('frac_a_better'),
  };
  if (!flip) return o;
  return {
    ...o,
    rmse_a: o.rmse_b, rmse_b: o.rmse_a, d_rmse: neg(o.d_rmse), d_rmse_lo: neg(o.d_rmse_hi), d_rmse_hi: neg(o.d_rmse_lo),
    mae_a: o.mae_b, mae_b: o.mae_a, d_mae: neg(o.d_mae), d_mae_lo: neg(o.d_mae_hi), d_mae_hi: neg(o.d_mae_lo),
    frac: isNum(o.frac) ? 1 - o.frac : o.frac,
  };
}
// 'win' (A significantly better), 'lose', 'tie' (CI includes 0) or 'na' (no CI)
function verdictOf(r, metric = 'rmse') {
  if (!r) return 'na';
  const lo = r[`d_${metric}_lo`]; const hi = r[`d_${metric}_hi`];
  if (!isNum(lo) || !isNum(hi)) return 'na';
  if (hi < 0) return 'win';
  if (lo > 0) return 'lose';
  return 'tie';
}

const unit = () => VARS[state.v]?.unit || '';
const varLabel = () => (VARS[state.v]?.label || state.v).toLowerCase();
const dig = () => 2;
const fmtD = (x) => {
  if (!isNum(x)) return '–';
  const a = Math.abs(x).toFixed(dig());
  return Number(a) === 0 ? a : `${x > 0 ? '+' : '−'}${a}`;
};
const fmtAbs = (x) => (isNum(x) ? Math.abs(x).toFixed(dig()) : '–');
const pct = (x) => (isNum(x) ? `${Math.round(x * 100)}%` : '–');
const regionText = () => (state.region === 'global' ? 'worldwide' : `in ${state.region}`);

/* ------------------------------------------------------------------ verdict */
function renderVerdict() {
  const rows = pairRows(state.a, state.b);
  const all = rows?.[rows.length - 1];
  const A = modelName(state.a); const B = modelName(state.b);
  const sub = $('h2h-verdict-sub');
  if (!all) {
    $('h2h-verdict').innerHTML = `No common forecasts of ${A} and ${B} for ${esc(varLabel())} ${esc(regionText())} in this window yet.`;
    sub.textContent = '';
    return;
  }
  const m = state.metric;
  const d = all[`d_${m}`]; const lo = all[`d_${m}_lo`]; const hi = all[`d_${m}_hi`];
  const v = verdictOf(all, m);
  const M = m.toUpperCase();
  const n = `${fmtCompact(all.n)} common pairs`;
  const st = isNum(all.frac) ? `; better at ${pct(all.frac)} of ${fmtInt(all.stations)} airports` : '';
  let html;
  if (v === 'win' || v === 'lose') {
    const [W, L] = v === 'win' ? [A, B] : [B, A];
    const [cl, ch] = v === 'win' ? [-hi, -lo] : [lo, hi];
    const frac = v === 'win' ? all.frac : (isNum(all.frac) ? 1 - all.frac : null);
    const stw = isNum(frac) ? `; better at ${pct(frac)} of ${fmtInt(all.stations)} airports` : '';
    html = `<b class="win">${W}</b> beats <b>${L}</b> by <b>${fmtAbs(d)} ${esc(unit())}</b> ${M} (95% CI ${fmtAbs(cl)}–${fmtAbs(ch)}) on ${n}${stw}.`;
  } else if (v === 'tie') {
    html = `<b>No significant difference</b> between ${A} and ${B}: ${M} A − B = ${fmtD(d)} ${esc(unit())} (95% CI ${fmtD(lo)} to ${fmtD(hi)}) on ${n}${st}.`;
  } else {
    html = `${A} − ${B}: ${M} difference ${fmtD(d)} ${esc(unit())} on ${n}${st} — too few days of common data for a confidence interval.`;
  }
  $('h2h-verdict').innerHTML = html;
  const bits = [`${VARS[state.v]?.label || state.v}, all lead times 0–72 h, ${regionText()}, last ${state.win === '7d' ? '7' : '30'} days`];
  bits.push(`RMSE ${fmt(all.rmse_a, 2)} vs ${fmt(all.rmse_b, 2)} · MAE ${fmt(all.mae_a, 2)} vs ${fmt(all.mae_b, 2)} ${unit()}`);
  if (isNum(all.days) && all.days < FEW_DAYS) bits.push(`only ${all.days} day${all.days === 1 ? '' : 's'} of common data — the interval is rough`);
  const lag = state.data?.windows?.[state.win]?.lag1?.[`${state.a}|${state.b}`]?.[state.v] ?? state.data?.windows?.[state.win]?.lag1?.[`${state.b}|${state.a}`]?.[state.v];
  if (isNum(lag) && lag > 0.5) bits.push(`day-to-day correlation of the difference is ${lag.toFixed(2)}: the interval may be somewhat too narrow`);
  sub.textContent = bits.join(' · ');
}

/* ------------------------------------------------------------------ chart */
function renderChart() {
  const el = $('h2h-chart');
  if (chart) { chart.ro.disconnect(); chart.u.destroy(); chart = null; }
  el.innerHTML = '';
  const rows = pairRows(state.a, state.b);
  const m = state.metric;
  $('h2h-chart-sub').textContent = `${m.toUpperCase()} of A minus ${m.toUpperCase()} of B on their common sample, ${unit()}. Below zero: A (${modelText(state.a)}) is more accurate. Whiskers: 95% confidence interval.`;
  if (!rows || !rows.slice(0, 3).some((r) => r && isNum(r[`d_${m}`]))) {
    el.innerHTML = '<div class="empty-note" style="margin-top:40px">No common pairs for this selection.</div>';
    return;
  }
  if (typeof uPlot === 'undefined') return;
  const labels = state.data.buckets.slice(0, 3);
  const x = [0, 1, 2];
  const pts = rows.slice(0, 3);
  const y = pts.map((r) => (r && isNum(r[`d_${m}`]) ? r[`d_${m}`] : null));
  const lo = pts.map((r) => (r && isNum(r[`d_${m}_lo`]) ? r[`d_${m}_lo`] : null));
  const hi = pts.map((r) => (r && isNum(r[`d_${m}_hi`]) ? r[`d_${m}_hi`] : null));
  const col = { win: cssVar('--good'), lose: cssVar('--warm'), tie: cssVar('--muted'), na: cssVar('--muted') };
  const colours = pts.map((r) => col[verdictOf(r, m)]);
  const ext = [...y, ...lo, ...hi].filter(isNum);
  const span = Math.max(Math.abs(Math.min(0, ...ext)), Math.abs(Math.max(0, ...ext)), 0.05) * 1.2;
  const ymin = Math.min(0, ...ext) < 0 ? -span : -span * 0.25;
  const ymax = Math.max(0, ...ext) > 0 ? span : span * 0.25;
  const ax = axisStyle();
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  tip.hidden = true;
  const dpr = devicePixelRatio || 1;
  const opts = {
    width: el.clientWidth || 420,
    height: 240,
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false } },
    scales: { x: { time: false, range: () => [-0.5, 2.5] }, y: { range: () => [ymin, ymax] } },
    axes: [
      { ...ax, size: 34, label: 'Lead time', labelSize: 16, labelGap: 0, labelFont: ax.font, splits: () => x, values: (u, v) => v.map((i) => BUCKET_LABEL[labels[i]] || ''), grid: { show: false } },
      { ...ax, size: 50, label: `A − B, ${unit()}`, labelSize: 14, labelFont: ax.font, values: (u, v) => v.map((t) => fmtD(t)) },
    ],
    series: [{}, { label: 'A − B', stroke: 'transparent', paths: () => null, points: { show: false } }],
    hooks: {
      draw: [(u) => {
        const ctx = u.ctx;
        ctx.save();
        // zero line
        const y0 = u.valToPos(0, 'y', true);
        ctx.strokeStyle = cssVar('--border-strong');
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath(); ctx.moveTo(u.bbox.left, y0); ctx.lineTo(u.bbox.left + u.bbox.width, y0); ctx.stroke();
        for (let i = 0; i < 3; i++) {
          if (!isNum(y[i])) continue;
          const cx = u.valToPos(i, 'x', true);
          ctx.strokeStyle = colours[i];
          ctx.fillStyle = colours[i];
          ctx.lineWidth = 2 * dpr;
          if (isNum(lo[i]) && isNum(hi[i])) {
            const a = u.valToPos(lo[i], 'y', true); const b = u.valToPos(hi[i], 'y', true);
            const w = 7 * dpr;
            ctx.beginPath();
            ctx.moveTo(cx, a); ctx.lineTo(cx, b);
            ctx.moveTo(cx - w, a); ctx.lineTo(cx + w, a);
            ctx.moveTo(cx - w, b); ctx.lineTo(cx + w, b);
            ctx.stroke();
          }
          const cy = u.valToPos(y[i], 'y', true);
          ctx.beginPath(); ctx.arc(cx, cy, 5.5 * dpr, 0, 2 * Math.PI); ctx.fill();
          ctx.lineWidth = 2 * dpr; ctx.strokeStyle = cssVar('--surface-solid'); ctx.stroke();
        }
        ctx.restore();
      }],
      setCursor: [(u) => {
        const left = u.cursor.left;
        if (left == null || left < 0) { tip.hidden = true; return; }
        const i = Math.round(u.posToVal(left, 'x'));
        const r = pts[i];
        if (i < 0 || i > 2 || !r) { tip.hidden = true; return; }
        tip.innerHTML = `<div class="tt">${esc(BUCKET_LABEL[labels[i]])} · ${esc(unit())}</div>
          <div class="tr"><span>A − B</span><b>${fmtD(r[`d_${m}`])}</b></div>
          <div class="tr"><span>95% CI</span><b>${isNum(r[`d_${m}_lo`]) ? `${fmtD(r[`d_${m}_lo`])} … ${fmtD(r[`d_${m}_hi`])}` : '–'}</b></div>
          <div class="tr"><span>${esc(modelText(state.a))}</span><b>${fmt(r[`${m}_a`], 2)}</b></div>
          <div class="tr"><span>${esc(modelText(state.b))}</span><b>${fmt(r[`${m}_b`], 2)}</b></div>
          <div class="tr"><span>Common pairs</span><b>${fmtCompact(r.n)}</b></div>`;
        tip.hidden = false;
        const px = u.valToPos(i, 'x') + u.bbox.left / dpr;
        const w = tip.offsetWidth; const W = el.clientWidth;
        tip.style.left = `${px + 14 + w > W ? Math.max(0, px - 14 - w) : px + 14}px`;
      }],
    },
  };
  const u = new uPlot(opts, [x, y], el);
  el.appendChild(tip);
  el.addEventListener('mouseleave', () => { tip.hidden = true; });
  const ro = observeWidth(el, (w) => u.setSize({ width: w, height: opts.height }));
  chart = { u, ro };
}

/* ------------------------------------------------------------------ tables */
function sigBadge(r, m = 'rmse') {
  const v = verdictOf(r, m);
  if (v === 'win') return '<span class="sig win">A better</span>';
  if (v === 'lose') return '<span class="sig lose">B better</span>';
  if (v === 'tie') return '<span class="sig tie">no sig. diff.</span>';
  return '<span class="sig tie">no CI</span>';
}
function renderTable() {
  const rows = pairRows(state.a, state.b);
  const m = state.metric; const M = m.toUpperCase();
  if (!rows) { $('h2h-table').innerHTML = '<div class="empty-note">No common pairs for this selection.</div>'; return; }
  const body = state.data.buckets.map((bk, i) => {
    const r = rows[i];
    if (!r) return `<tr class="thin"><td>${esc(BUCKET_LABEL[bk] || bk)}</td><td colspan="6" class="muted" style="text-align:center">no common pairs</td></tr>`;
    const ci = isNum(r[`d_${m}_lo`]) ? `${fmtD(r[`d_${m}_lo`])} … ${fmtD(r[`d_${m}_hi`])}` : '–';
    return `<tr${bk === 'all' ? ' style="font-weight:700"' : ''}><td>${esc(BUCKET_LABEL[bk] || bk)}</td><td>${fmtCompact(r.n)}</td><td>${fmt(r[`${m}_a`], 2)}</td><td>${fmt(r[`${m}_b`], 2)}</td>
      <td>${fmtD(r[`d_${m}`])}<div class="ci">${ci}</div></td><td>${sigBadge(r, m)}</td><td title="${fmtInt(r.stations)} airports with ≥ ${state.data.method?.station_min_n ?? 10} common pairs">${pct(r.frac)}</td></tr>`;
  }).join('');
  $('h2h-table').innerHTML = `<table class="tbl h2h-tbl">
    <thead><tr><th scope="col">Lead</th><th scope="col">n</th><th scope="col">${M} A</th><th scope="col">${M} B</th><th scope="col">A − B</th><th scope="col" aria-label="Significance"></th><th scope="col" title="Share of airports where A has the lower RMSE">A wins</th></tr></thead>
    <tbody>${body}</tbody>
    <caption>A = ${esc(modelText(state.a, true))}, B = ${esc(modelText(state.b, true))} · ${esc(unit())} · interval under each difference.</caption>
  </table>`;
}

function renderMatrix() {
  const ids = models();
  const m = state.metric;
  const hasAny = ids.some((a) => ids.some((b) => pairRows(a, b)?.[3]));
  if (!hasAny) { $('h2h-matrix').innerHTML = '<div class="empty-note">No comparisons for this selection.</div>'; return; }
  const head = ids.map((b) => `<th scope="col" title="${esc(modelText(b, true))}">${legendSwatch(b)}<br>${modelName(b)}</th>`).join('');
  const body = ids.map((a) => `<tr><th scope="row" title="${esc(modelText(a, true))}">${modelName(a)} ${legendSwatch(a)}</th>${ids.map((b) => {
    if (a === b) return '<td class="self">—</td>';
    const r = pairRows(a, b)?.[3];
    if (!r) return '<td class="na">·</td>';
    const v = verdictOf(r, m);
    const sel = a === state.a && b === state.b ? ' sel' : '';
    const desc = `${modelText(a)} minus ${modelText(b)}: ${fmtD(r[`d_${m}`])} ${unit()}${isNum(r[`d_${m}_lo`]) ? ` (95% CI ${fmtD(r[`d_${m}_lo`])} to ${fmtD(r[`d_${m}_hi`])})` : ''}, ${fmtCompact(r.n)} pairs`;
    return `<td class="${v === 'na' ? 'tie' : v}${sel}" data-a="${esc(a)}" data-b="${esc(b)}" title="${esc(desc)}" tabindex="0" role="button" aria-label="${esc(desc)}">${fmtD(r[`d_${m}`])}<small>${fmtCompact(r.n)}</small></td>`;
  }).join('')}</tr>`).join('');
  $('h2h-matrix').innerHTML = `<table class="mx"><thead><tr><th scope="col"><span class="sr-only">Row model</span></th>${head}</tr></thead><tbody>${body}</tbody>
    <caption class="note" style="caption-side:bottom;text-align:left;padding-top:6px">${m.toUpperCase()} difference row − column in ${esc(unit())}; small number = common pairs.</caption></table>`;
}

function renderCommon() {
  const c = state.data?.windows?.[state.win]?.common?.[state.v];
  const el = $('h2h-common');
  const blk = c && (state.region === 'global' ? c.global : c.regions?.[state.region]);
  if (!blk || !Object.keys(blk).length) { el.innerHTML = '<div class="empty-note">No forecasts verified by all models for this selection.</div>'; $('h2h-common-note').textContent = 'Every model scored on exactly the same forecasts.'; return; }
  const bk = state.data.buckets; const ia = bk.length - 1;
  const ids = models().filter((id) => blk[id]);
  ids.sort((p, q) => (blk[p][ia][3] ?? Infinity) - (blk[q][ia][3] ?? Infinity));
  const best = blk[ids[0]]?.[ia]?.[3];
  const n = blk[ids[0]]?.[ia]?.[0];
  const exc = (c.excluded || []).map((id) => modelText(id));
  $('h2h-common-note').textContent = `Every model scored on exactly the same ${fmtCompact(n)} forecasts: the hours, airports and lead times that all of them verified (${unit()}, ${regionText()}, last ${state.win === '7d' ? '7' : '30'} days).${exc.length ? ` Left out because they have too little data yet: ${exc.join(', ')}.` : ''} Differences here have no interval; use the pairs above for significance.`;
  el.innerHTML = `<table class="tbl">
    <thead><tr><th scope="col">#</th><th scope="col" style="text-align:left">Model</th><th scope="col">Bias</th><th scope="col">MAE</th><th scope="col">RMSE</th>${bk.slice(0, 3).map((b) => `<th scope="col">RMSE<span class="u">${esc(BUCKET_LABEL[b])}</span></th>`).join('')}</tr></thead>
    <tbody>${ids.map((id, i) => { const r = blk[id]; const a = r[ia]; return `<tr><td>${i + 1}</td><td style="text-align:left"><span class="mname">${legendSwatch(id)}${modelName(id)}</span></td>
      <td>${fmtD(a[1])}</td><td>${fmt(a[2], 2)}</td><td class="${a[3] === best ? 'best' : ''}">${fmt(a[3], 2)}</td>${r.slice(0, 3).map((x) => `<td>${fmt(x[3], 2)}</td>`).join('')}</tr>`; }).join('')}</tbody>
  </table>`;
}

/* ------------------------------------------------------------------ controls */
function models() {
  const have = state.data?.models || [];
  const order = MODELS.map((m) => m.id);
  return [...have].sort((p, q) => (order.indexOf(p) + 1 || 99) - (order.indexOf(q) + 1 || 99));
}
function fillModelSelects() {
  const ids = models();
  const opts = (sel) => ids.map((id) => `<option value="${esc(id)}"${id === sel ? ' selected' : ''}>${esc(modelText(id, true))}</option>`).join('');
  $('h2h-a').innerHTML = opts(state.a);
  $('h2h-b').innerHTML = opts(state.b);
}
function render() {
  if (!state.data) return;
  fillModelSelects();
  renderVerdict();
  renderChart();
  renderTable();
  renderMatrix();
  renderCommon();
}

function setPair(a, b) {
  if (a === b) { const other = models().find((id) => id !== a); if (state.a === a) b = state.b !== a ? state.b : other; else a = state.a !== b ? state.a : other; }
  state.a = a; state.b = b;
  render();
}
$('h2h-a').addEventListener('change', (e) => setPair(e.target.value, state.b === e.target.value ? state.a : state.b));
$('h2h-b').addEventListener('change', (e) => setPair(state.a === e.target.value ? state.b : state.a, e.target.value));
$('h2h-swap').addEventListener('click', () => setPair(state.b, state.a));
seg($('h2h-var'), state.v, (v) => { state.v = v; render(); });
seg($('h2h-win'), state.win, (v) => { state.win = v; render(); });
seg($('h2h-metric'), state.metric, (v) => { state.metric = v; render(); });
$('h2h-region').addEventListener('change', (e) => { state.region = e.target.value; render(); });
function pick(e) {
  const td = e.target.closest('td[data-a]');
  if (!td) return;
  if (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ') return;
  e.preventDefault();
  setPair(td.dataset.a, td.dataset.b);
  $('h2h').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
$('h2h-matrix').addEventListener('click', pick);
$('h2h-matrix').addEventListener('keydown', pick);
onThemeChange(() => render());

async function boot() {
  try {
    // scores.js has loaded meta.json first (it sets the cache-busting key); a fresh
    // 5-minute key is fine too
    state.data = await getJSON('data/compare.json', { optional: true });
  } catch (err) {
    console.error(err);
  }
  if (!state.data) {
    $('h2h-verdict').textContent = 'Head-to-head comparisons are not available yet (they appear after the next site build).';
    for (const id of ['h2h-chart', 'h2h-table', 'h2h-matrix', 'h2h-common']) $(id).innerHTML = '';
    return;
  }
  (state.data.fields || []).forEach((f, i) => { F[f] = i; });
  const ids = models();
  state.a = ids.includes('ecmwf') ? 'ecmwf' : ids[0];
  state.b = ids.includes('gfs') && state.a !== 'gfs' ? 'gfs' : ids.find((id) => id !== state.a);
  const regs = new Set();
  for (const w of Object.values(state.data.windows || {})) for (const [r, blk] of Object.entries(w.regions || {})) if (Object.keys(blk).length) regs.add(r);
  const list = [...regs].sort((p, q) => (REGION_ORDER.indexOf(p) + 1 || 99) - (REGION_ORDER.indexOf(q) + 1 || 99) || p.localeCompare(q));
  $('h2h-region').innerHTML = `<option value="global">Global</option>${list.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join('')}`;
  if (!ids.length || !state.b) {
    $('h2h-verdict').textContent = 'Not enough models with verified forecasts to compare yet.';
    return;
  }
  render();
}
if (!H2H_VARS.includes(state.v)) state.v = 't';
boot();
