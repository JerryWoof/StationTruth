// StationTruth — pipeline status page (data/health.json, see docs/DATA.md).
import {
  esc, fmt, fmtInt, isNum, getJSON, loadMeta, setStatus, renderNav, cssVar, ago, dateLabel, utcLabel, parseTime,
  runLabel, modelText, onThemeChange, axisStyle, observeWidth,
} from './common.js?v=__BUILD__';

const $ = (id) => document.getElementById(id);
const BUILD_STALE_H = 3; // health.json is written by the hourly build: older means the build is stuck
const LEVEL = { info: 0, degraded: 1, down: 2 };
const TITLE = { ok: 'All systems normal', degraded: 'Degraded', down: 'Down' };
let H = null;
let chart = null;

const when = (t) => { const d = parseTime(t); return d ? `${dateLabel(d)} ${utcLabel(d)}` : '–'; };
const whenAgo = (t) => (t ? `${ago(t)} <span class="muted">(${esc(when(t))})</span>` : '–');

/* ------------------------------------------------------------------ summary */
function effective(h, live) {
  const reasons = [...(h.reasons || [])];
  const gen = parseTime(h.generated);
  const ageH = gen ? (Date.now() - gen) / 3600000 : Infinity;
  if (ageH > BUILD_STALE_H) {
    reasons.unshift({ level: 'down', component: 'build', msg: `The site has not been rebuilt for ${ageH < 48 ? `${Math.round(ageH)} h` : `${Math.round(ageH / 24)} days`}: everything below may be out of date.` });
  }
  // analysis / TAF scores are built after health.json in the hourly job: use the live files
  if (live.analysis) {
    const i = reasons.findIndex((r) => r.component === 'analysis');
    if (i >= 0 && h.analysis?.status === 'not_built') reasons.splice(i, 1);
    const a = parseTime(live.analysis.generated);
    if (a && Date.now() - a > BUILD_STALE_H * 3600000) reasons.push({ level: 'degraded', component: 'analysis', msg: `Analysis map last built ${ago(a)}.` });
  }
  reasons.sort((p, q) => LEVEL[q.level] - LEVEL[p.level]);
  const worst = Math.max(0, ...reasons.map((r) => LEVEL[r.level] ?? 0));
  return { status: ['ok', 'degraded', 'down'][worst], reasons };
}

function renderSummary(h, live) {
  const { status, reasons } = effective(h, live);
  $('light').className = `light ${status}`;
  $('light-title').textContent = TITLE[status];
  $('light-sub').innerHTML = `Checked at the last site build, ${whenAgo(h.generated)}.`;
  const list = reasons.filter((r) => r.level !== 'info' || reasons.length <= 8);
  $('reasons').innerHTML = list.length
    ? list.map((r) => `<li><span class="lv ${esc(r.level)}">${esc(r.level === 'info' ? 'note' : r.level)}</span><span>${esc(r.msg)}</span></li>`).join('')
    : '<li><span class="lv info">ok</span><span>Observations, model runs, TAFs and the build are all on time.</span></li>';
}

function renderKpis(h) {
  const o = h.obs || {};
  const runs = h.runs?.models || {};
  const newest = Object.entries(runs).map(([m, r]) => `${esc(modelText(m))} ${esc(runLabel(r.latest))}`).join(' · ');
  const cards = [
    ['Latest observation', o.last_obs ? ago(o.last_obs) : 'none', o.last_obs ? when(o.last_obs) : ''],
    ['Airports reporting', fmtInt(o.stations_3h), `last 3 h · ${fmtInt(o.stations_24h)} in 24 h`],
    ['Site build', h.build?.generated ? ago(h.build.generated) : '–', isNum(h.build?.seconds) ? `took ${fmt(h.build.seconds, 0)} s` : ''],
    ['Newest model runs', Object.keys(runs).length ? `${Object.keys(runs).length} models` : 'none', newest],
  ];
  $('kpis').innerHTML = cards.map(([k, v, s]) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${s}</div></div>`).join('');
}

/* ------------------------------------------------------------------ obs chart */
function renderObs(h) {
  const el = $('obs-chart');
  if (chart) { chart.ro.disconnect(); chart.u.destroy(); chart = null; }
  el.innerHTML = '';
  const hr = h.obs?.hours;
  if (!hr || !hr.reports?.length) { el.innerHTML = '<div class="empty-note" style="margin-top:40px">No observations.</div>'; return; }
  $('obs-sub').textContent = `By observation time (UTC). Median ${fmtInt(h.obs.median_reports)} reports per complete hour${(h.obs.gaps || []).length ? `; ${h.obs.gaps.length} flagged period${h.obs.gaps.length > 1 ? 's' : ''}` : ''}.`;
  if (typeof uPlot === 'undefined') return;
  const t0 = parseTime(hr.start) / 1000;
  const x = hr.reports.map((_, i) => t0 + i * hr.step * 60 + 1800); // bar at the middle of its hour
  const pick = (want) => hr.reports.map((v, i) => (want.includes(hr.state[i]) ? v : null));
  const ok = pick(['ok']); const bad = pick(['gap', 'low']); const part = pick(['partial']);
  // zero-report hours get a small stub so a gap is visible
  const ymax = Math.max(...hr.reports, 1);
  const badShown = bad.map((v) => (v === 0 ? ymax * 0.03 : v));
  const ax = axisStyle();
  const bars = uPlot.paths.bars({ size: [0.82, 40], gap: 1 });
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  tip.hidden = true;
  const opts = {
    width: el.clientWidth || 600,
    height: 220,
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false } },
    scales: { x: { time: false, range: () => [t0, t0 + hr.reports.length * 3600] }, y: { range: () => [0, ymax * 1.1] } },
    axes: [
      {
        ...ax, size: 30,
        // UTC-aligned ticks every 6 h (12 h on narrow screens)
        splits: (u) => {
          const step = (u.bbox.width / (devicePixelRatio || 1)) < 480 ? 43200 : 21600;
          const out = [];
          for (let t = Math.ceil(t0 / step) * step; t <= t0 + hr.reports.length * 3600; t += step) out.push(t);
          return out;
        },
        values: (u, v) => v.map((ts) => {
          const d = new Date(ts * 1000);
          const h = d.getUTCHours();
          return h === 0 ? `${d.getUTCDate()} ${d.toLocaleString('en-GB', { month: 'short', timeZone: 'UTC' })}` : `${String(h).padStart(2, '0')}Z`;
        }),
      },
      { ...ax, size: 52, values: (u, v) => v.map((n) => (n >= 1000 ? `${(n / 1000).toFixed(n % 1000 ? 1 : 0)}k` : n)) },
    ],
    series: [
      {},
      { label: 'normal', fill: cssVar('--accent'), stroke: cssVar('--accent'), width: 0, paths: bars, points: { show: false } },
      { label: 'flagged', fill: cssVar('--warm'), stroke: cssVar('--warm'), width: 0, paths: bars, points: { show: false } },
      { label: 'filling', fill: cssVar('--surface-3'), stroke: cssVar('--border-strong'), width: 1, paths: bars, points: { show: false } },
    ],
    hooks: {
      setCursor: [(u) => {
        const i = u.cursor.idx;
        if (i == null) { tip.hidden = true; return; }
        const d = new Date((x[i] - 1800) * 1000);
        const st = { ok: 'normal', gap: 'no reports', low: 'unusually few', partial: 'still filling' }[hr.state[i]] || hr.state[i];
        tip.innerHTML = `<div class="tt">${esc(dateLabel(d))} ${esc(utcLabel(d))}</div>
          <div class="tr"><span>Reports</span><b>${fmtInt(hr.reports[i])}</b></div>
          <div class="tr"><span>Airports</span><b>${fmtInt(hr.stations?.[i])}</b></div>
          <div class="tr"><span>State</span><b>${esc(st)}</b></div>`;
        tip.hidden = false;
        const left = u.valToPos(x[i], 'x') + u.bbox.left / (devicePixelRatio || 1);
        const w = tip.offsetWidth; const W = el.clientWidth;
        tip.style.left = `${left + 14 + w > W ? Math.max(0, left - 14 - w) : left + 14}px`;
      }],
    },
  };
  const u = new uPlot(opts, [x, ok, badShown, part], el);
  el.appendChild(tip);
  el.addEventListener('mouseleave', () => { tip.hidden = true; });
  chart = { u, ro: observeWidth(el, (w) => u.setSize({ width: w, height: opts.height })) };
}

/* ------------------------------------------------------------------ runs */
function renderRuns(h) {
  const R = h.runs || {};
  const models = Object.entries(R.models || {});
  if (!models.length) { $('runs').innerHTML = '<div class="empty-note">No model runs found.</div>'; return; }
  const cycles = models[0][1].cycles.map((c) => c.run);
  const mark = { ok: '✓', missing: '✗', pending: '…', before: '·' };
  const word = { ok: 'stored', missing: 'missing', pending: 'not due yet', before: 'before this model was collected' };
  const head = cycles.map((r) => { const d = runLabel(r).split(' '); return `<th scope="col" class="c">${esc(d[0])} ${esc(d[1])}<br>${esc(d[2])}</th>`; }).join('');
  const rows = models.map(([m, r]) => {
    const cells = cycles.map((rid) => {
      const c = r.cycles.find((x) => x.run === rid) || { status: 'before' };
      const late = isNum(c.late_h) ? `, stored ${c.late_h} h after start (${when(c.stored)})` : '';
      const t = `${modelText(m)} ${runLabel(rid)}: ${word[c.status] || c.status}${late}`;
      return `<td class="c ${esc(c.status)}" title="${esc(t)}" aria-label="${esc(t)}">${mark[c.status] || '?'}</td>`;
    }).join('');
    const late = isNum(r.median_late_h) ? `${fmt(r.median_late_h, 1)} h` : '–';
    return `<tr><th scope="row" style="text-align:left">${esc(modelText(m))}</th>${cells}<td>${r.stored}/${r.expected}</td><td class="lt">${late}</td></tr>`;
  }).join('');
  $('runs').innerHTML = `<table class="tbl runs"><thead><tr><th scope="col">Model</th>${head}<th scope="col">Stored</th><th scope="col" title="Median time from model start to the run being stored">Delay</th></tr></thead><tbody>${rows}</tbody></table>`;
  $('runs-note').textContent = `✓ stored · ✗ missing · … not due yet · · before the model was collected. Expected: ${(R.cycle_hours || [0, 12]).map((x) => `${String(x).padStart(2, '0')}Z`).join(' and ')} runs, due ${models[0][1].grace_h ?? 12} h after their start. Delay = median time from model start to storage${R.listing ? '' : ' (unknown: no asset listing in this build)'}; backfilled runs show long delays.`;
}

/* ------------------------------------------------------------------ products */
function card(title, status, facts) {
  return `<section class="panel"><h3><span class="dot ${esc(status)}" aria-hidden="true"></span>${esc(title)}</h3><dl class="facts">${facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl></section>`;
}
function renderCards(h, live) {
  const out = [];
  const b = h.build || {};
  const errs = Object.entries(b.errors || {});
  out.push(card('Site build', errs.length ? 'degraded' : 'ok', [
    ['Built', whenAgo(b.generated)],
    ['Duration', isNum(b.seconds) ? `${fmt(b.seconds, 1)} s` : '–'],
    ['Steps', b.timings ? esc(Object.entries(b.timings).filter(([k]) => k !== 'total').map(([k, v]) => `${k} ${fmt(v, 1)} s`).join(', ')) : '–'],
    ['Peak memory', isNum(b.peak_rss_mb) ? `${fmtInt(b.peak_rss_mb)} MB` : '–'],
    ['Pairs', fmtInt(b.pairs)],
    ['Errors', errs.length ? errs.map(([k, v]) => `<b>${esc(k)}</b>: ${esc(v)}`).join('<br>') : 'none'],
  ]));
  const a = live.analysis || (h.analysis?.status !== 'not_built' ? h.analysis : null);
  const aAge = a?.generated ? Date.now() - parseTime(a.generated) : Infinity;
  out.push(card('Analysis map', !a ? 'down' : aAge > BUILD_STALE_H * 3600000 ? 'degraded' : 'ok', [
    ['Built', a ? whenAgo(a.generated) : 'not available'],
    ['Valid', a?.valid ? esc(when(a.valid)) : '–'],
    ['Runtime', isNum(a?.runtime?.seconds ?? a?.seconds) ? `${fmt(a.runtime?.seconds ?? a.seconds, 1)} s` : '–'],
  ]));
  const t = h.tafs || {};
  const ts = live.taf || (h.taf_scores?.status === 'ok' ? h.taf_scores : null);
  const tafStale = !isNum(t.age_h) || t.age_h > 8;
  out.push(card('TAFs', tafStale ? 'degraded' : 'ok', [
    ['Newest TAF', whenAgo(t.last_issued)],
    ['Files (7 days)', (t.files || []).length ? esc(t.files.map((f) => `${f.date.slice(5)}: ${isNum(f.rows) ? `${Math.round(f.rows / 1000)}k` : '?'}`).join(' · ')) : 'none'],
    ['TAF scores', ts ? whenAgo(ts.generated) : 'not built'],
  ]));
  const o = h.obs || {};
  out.push(card('Observation files', (h.reasons || []).some((r) => r.component === 'obs' && r.level !== 'info') ? 'degraded' : 'ok', [
    ['Last collection', o.last_collect ? whenAgo(o.last_collect) : '–'],
    ['Files (7 days)', (o.files || []).length ? esc(o.files.map((f) => `${f.date.slice(5)}: ${isNum(f.rows) ? `${Math.round(f.rows / 1000)}k` : '?'}`).join(' · ')) : 'none'],
  ]));
  $('cards').innerHTML = out.join('');
}

/* ------------------------------------------------------------------ boot */
let live = {};
function render() {
  if (!H) return;
  renderSummary(H, live);
  renderKpis(H);
  renderObs(H);
  renderRuns(H);
  renderCards(H, live);
}
onThemeChange(() => render());

async function boot() {
  renderNav('status');
  let meta = null;
  try {
    meta = await loadMeta();
    setStatus(meta);
    [H, live.analysis, live.taf] = await Promise.all([
      getJSON('data/health.json', { optional: true }),
      getJSON('data/analysis/analysis.json', { optional: true }).catch(() => null),
      getJSON('data/taf_scores.json', { optional: true }).catch(() => null),
    ]);
  } catch (err) {
    console.error(err);
  }
  if (!H) {
    $('light').className = 'light';
    $('light-title').textContent = 'No status yet';
    $('light-sub').textContent = meta ? 'health.json has not been built yet; it appears after the next site build.' : 'The site has no data yet.';
    return;
  }
  render();
}
boot();
