# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localhost-only FastAPI dashboard: ALT similarity, neighbor thumbnails, action RMS bars."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from gr00t.eval.inference_viz_state import InferenceVizState

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ALT / VLA inference</title>
  <style>
    :root {
      font-family: system-ui, sans-serif;
      background: #0f1115;
      color: #e8eaed;
    }
    body { margin: 0; padding: 12px; }
    h1 { font-size: 1rem; font-weight: 600; margin: 0 0 12px 0; color: #9aa0a6; }
    .main-layout {
      display: flex;
      flex-direction: row;
      gap: 16px;
      align-items: flex-start;
      max-width: 1600px;
      margin: 0 auto;
    }
    .col-left {
      flex: 1;
      min-width: 0;
    }
    .col-right {
      width: 400px;
      flex-shrink: 0;
      position: sticky;
      top: 12px;
      max-height: calc(100vh - 24px);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    @media (max-width: 960px) {
      .main-layout { flex-direction: column; }
      .col-right {
        width: 100%;
        position: static;
        max-height: none;
      }
    }
    .score-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin-bottom: 4px; }
    .ood-big { font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800; letter-spacing: -0.03em;
      font-variant-numeric: tabular-nums; line-height: 1; }
    .ood-sub { font-size: 0.8rem; color: #9aa0a6; line-height: 1.35; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 700;
      font-size: 0.85rem; }
    .badge-id { background: #1b4332; color: #95d5b2; }
    .badge-ood { background: #5c1010; color: #ff8a8a; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .panel { background: #1a1d24; border-radius: 10px; padding: 10px; border: 1px solid #2d323c; }
    .panel h2 { margin: 0 0 8px 0; font-size: 0.75rem; color: #9aa0a6; font-weight: 600; line-height: 1.3; }
    .img-slot {
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 6px;
      overflow: hidden;
      background: #000;
    }
    .img-slot img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .cap { font-size: 0.72rem; color: #80868b; margin-top: 6px; }
    .bar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; font-size: 0.68rem; }
    .bar-name {
      width: 140px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #bdc1c6;
      flex-shrink: 0;
    }
    .bar-track { flex: 1; height: 10px; background: #2d323c; border-radius: 4px; overflow: hidden; min-width: 0; }
    .bar-fill { height: 100%; background: #8ab4f8; border-radius: 4px; }
    .bar-fill-diff { height: 100%; background: #f4b400; border-radius: 4px; }
    .bar-fill-alt0 { height: 100%; background: #81c995; border-radius: 4px; }
    .waiting { color: #80868b; padding: 24px; text-align: center; }
    .compare-note { font-size: 0.68rem; color: #80868b; margin-top: 4px; }
    .err-grid-wrap { overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
    .err-grid { font-size: 0.62rem; border-collapse: collapse; width: 100%; }
    .err-grid th, .err-grid td {
      border: 1px solid #3c4043;
      padding: 3px 5px;
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .err-grid thead th { background: #252830; color: #bdc1c6; position: sticky; top: 0; z-index: 1; }
    .err-grid tbody th { background: #1e2128; color: #9aa0a6; text-align: center; }
    .err-grid td { color: #e8eaed; }
  </style>
</head>
<body>
  <h1>ALT similarity &mdash; live inference</h1>
  <div class="main-layout">
    <div class="col-left">
      <div class="grid" id="imgGrid">
        <div class="waiting" id="placeholder">Waiting for <code>POST /act</code> on the inference server&hellip;</div>
      </div>
    </div>
    <div class="col-right">
      <div class="panel">
        <div class="score-row">
          <div class="ood-big" id="oodScore">&mdash;</div>
          <div>
            <div id="oodBadge" class="badge badge-id">waiting</div>
            <div class="ood-sub">Higher = closer to pretraining (NN cosine). Low ⇒ more OOD.</div>
          </div>
        </div>
      </div>
      <div class="panel" id="barPanelGroot" style="display:none;">
        <h2 id="barTitleGroot">GR00T: per-joint RMS (|action| over time)</h2>
        <div id="barListGroot"></div>
      </div>
      <div class="panel" id="barPanelGrootT0" style="display:none;">
        <h2 id="barTitleGrootT0">GR00T: action at t = 0 (per joint)</h2>
        <div id="barListGrootT0"></div>
      </div>
      <div class="panel" id="barPanelAltT0" style="display:none;">
        <h2 id="barTitleAltT0">ALT rank-1: lookup action at t = 0 (per joint)</h2>
        <p class="compare-note" style="margin:0 0 6px 0;">From LeRobot parquet <code>action</code> at the matched frame, stored as-is in the table (absolute joint targets if your dataset exported them that way). Not <code>observation.state</code>.</p>
        <div id="barListAltT0"></div>
      </div>
      <div class="panel" id="barPanelDiff" style="display:none;">
        <h2 id="barTitleDiff">GR00T vs ALT rank-1: mean Δ per joint (deg)</h2>
        <div id="compareNote" class="compare-note"></div>
        <div id="barListDiff"></div>
      </div>
      <div class="panel" id="errGridPanel" style="display:none;">
        <h2 id="errGridTitle">Signed Δ (°) vs time: GR00T − ALT</h2>
        <p class="compare-note" style="margin:0 0 6px 0;">Rows = step index in chunk (0 … T−1); columns = joints. Values are <strong>not</strong> absolute.</p>
        <div id="errGridWrap" class="err-grid-wrap"></div>
      </div>
    </div>
  </div>
  <script>
    let lastSeq = -1;
    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }
    function setThumb(img, kind, neighbor, cam, seq) {
      const u = kind === 'q'
        ? '/thumb?kind=query&cam=' + cam + '&seq=' + seq
        : '/thumb?kind=match&neighbor=' + neighbor + '&cam=' + cam + '&seq=' + seq;
      img.loading = 'lazy';
      img.src = u;
    }
    function fillBarList(el, norms, fillClass) {
      let maxV = 0;
      norms.forEach(function (x) { if (x.rms > maxV) maxV = x.rms; });
      if (maxV < 1e-9) maxV = 1;
      el.innerHTML = norms.map(function (x) {
        const pct = Math.min(100, (x.rms / maxV) * 100);
        return '<div class="bar-row"><span class="bar-name" title="' + esc(x.label) + '">' +
          esc(x.label) + '</span><div class="bar-track"><div class="' + fillClass + '" style="width:' +
          pct.toFixed(1) + '%"></div></div><span>' + x.rms.toFixed(3) + '°</span></div>';
      }).join('');
    }
    function fillBarListAbsScale(el, norms, fillClass) {
      let maxA = 0;
      norms.forEach(function (x) {
        const a = Math.abs(x.rms);
        if (a > maxA) maxA = a;
      });
      if (maxA < 1e-9) maxA = 1;
      el.innerHTML = norms.map(function (x) {
        const a = Math.abs(x.rms);
        const pct = Math.min(100, (a / maxA) * 100);
        return '<div class="bar-row"><span class="bar-name" title="' + esc(x.label) + '">' +
          esc(x.label) + '</span><div class="bar-track"><div class="' + fillClass + '" style="width:' +
          pct.toFixed(1) + '%"></div></div><span>' + x.rms.toFixed(3) + '°</span></div>';
      }).join('');
    }
    function shortJointThLabel(full, idx) {
      const i = full.lastIndexOf('[');
      const j = full.lastIndexOf(']');
      if (i >= 0 && j > i) return 'j' + full.slice(i + 1, j);
      return 'j' + idx;
    }
    function errCellBg(v, maxAbs) {
      const a = Math.abs(v);
      const t = maxAbs > 1e-12 ? Math.min(1, a / maxAbs) : 0;
      const al = 0.14 + t * 0.48;
      if (v < -1e-9) return 'background:rgba(138,180,248,' + al + ')';
      if (v > 1e-9) return 'background:rgba(244,180,0,' + al + ')';
      return 'background:rgba(154,160,166,0.12)';
    }
    function renderErrGrid(wrapEl, gridDeg, colLabels) {
      if (!gridDeg || !gridDeg.length || !colLabels || !colLabels.length) {
        wrapEl.innerHTML = '';
        return;
      }
      let maxAbs = 0;
      for (let ti = 0; ti < gridDeg.length; ti++) {
        const row = gridDeg[ti];
        for (let ji = 0; ji < row.length; ji++) {
          const a = Math.abs(row[ji]);
          if (a > maxAbs) maxAbs = a;
        }
      }
      if (maxAbs < 1e-12) maxAbs = 1;
      let html = '<table class="err-grid"><thead><tr><th>t</th>';
      for (let ji = 0; ji < colLabels.length; ji++) {
        const full = colLabels[ji];
        html += '<th title="' + esc(full) + '">' + esc(shortJointThLabel(full, ji)) + '</th>';
      }
      html += '</tr></thead><tbody>';
      for (let ti = 0; ti < gridDeg.length; ti++) {
        html += '<tr><th>' + ti + '</th>';
        const row = gridDeg[ti];
        for (let ji = 0; ji < row.length; ji++) {
          const v = row[ji];
          html += '<td style="' + errCellBg(v, maxAbs) + '">' + v.toFixed(2) + '</td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
      wrapEl.innerHTML = html;
    }
    function render(d) {
      if (!d || d.seq === undefined) return;
      if (d.seq === lastSeq) return;
      lastSeq = d.seq;
      const ph = document.getElementById('placeholder');
      if (ph) ph.remove();
      document.getElementById('oodScore').textContent =
        (d.ood_score !== undefined && d.ood_score !== null)
          ? Number(d.ood_score).toFixed(4) : '—';
      const bd = document.getElementById('oodBadge');
      if (d.is_ood === true) {
        bd.textContent = 'OOD';
        bd.className = 'badge badge-ood';
      } else if (d.is_ood === false) {
        bd.textContent = 'in-distribution';
        bd.className = 'badge badge-id';
      } else {
        bd.textContent = 'no threshold';
        bd.className = 'badge badge-id';
      }
      const grid = document.getElementById('imgGrid');
      grid.innerHTML = '';
      const nCams = (d.alt_video_keys && d.alt_video_keys.length) || 0;
      const nNei = (d.alt_neighbors && d.alt_neighbors.length) || 0;
      function addPanel(title, inner) {
        const p = document.createElement('div');
        p.className = 'panel';
        p.innerHTML = '<h2>' + title + '</h2>' + inner;
        grid.appendChild(p);
      }
      for (let c = 0; c < nCams; c++) {
        const vk = d.alt_video_keys[c];
        const inner = '<div class="img-slot"><img alt="query cam' + c + '"/></div><div class="cap">' +
          esc(vk) + '</div>';
        addPanel('Query cam ' + c, inner);
        const img = grid.lastChild.querySelector('img');
        setThumb(img, 'q', 0, c, d.seq);
      }
      for (let n = 0; n < nNei; n++) {
        const nb = d.alt_neighbors[n];
        const cap = 'ep ' + nb.episode_index + ' · frame ' + nb.frame_index +
          ' · cos ' + Number(nb.cosine_similarity).toFixed(4);
        let inner = '';
        for (let c = 0; c < nCams; c++) {
          inner += '<div class="img-slot" style="margin-bottom:6px;"><img alt="n' + n + 'c' + c +
            '"/></div>';
        }
        inner += '<div class="cap">' + esc(cap) + '</div>';
        addPanel('Match #' + (n + 1), inner);
        const imgs = grid.lastChild.querySelectorAll('img');
        for (let c = 0; c < nCams; c++) {
          setThumb(imgs[c], 'm', n, c, d.seq);
        }
      }
      const jpG = document.getElementById('barPanelGroot');
      const blG = document.getElementById('barListGroot');
      const btG = document.getElementById('barTitleGroot');
      const norms = d.joint_action_rms || [];
      if (norms.length) {
        jpG.style.display = 'block';
        if (btG) {
          btG.textContent = 'GR00T: per-joint RMS over horizon — ' + (d.joint_rms_description || '');
        }
        fillBarList(blG, norms, 'bar-fill');
      } else {
        jpG.style.display = 'none';
      }
      const jpGt0 = document.getElementById('barPanelGrootT0');
      const blGt0 = document.getElementById('barListGrootT0');
      const btGt0 = document.getElementById('barTitleGrootT0');
      const t0g = d.groot_action_t0_deg || [];
      if (jpGt0 && blGt0 && t0g.length) {
        jpGt0.style.display = 'block';
        if (btGt0) {
          btGt0.textContent = 'GR00T: action at t = 0 — ' + (d.action_t0_description || '');
        }
        fillBarListAbsScale(blGt0, t0g, 'bar-fill');
      } else if (jpGt0) {
        jpGt0.style.display = 'none';
      }
      const jpAt0 = document.getElementById('barPanelAltT0');
      const blAt0 = document.getElementById('barListAltT0');
      const btAt0 = document.getElementById('barTitleAltT0');
      const t0a = d.alt_lookup_action_t0_deg || [];
      if (jpAt0 && blAt0 && t0a.length) {
        jpAt0.style.display = 'block';
        if (btAt0) {
          btAt0.textContent = 'ALT rank-1: lookup action at t = 0 — ' + (d.action_t0_description || '');
        }
        fillBarListAbsScale(blAt0, t0a, 'bar-fill-alt0');
      } else if (jpAt0) {
        jpAt0.style.display = 'none';
      }
      const jpD = document.getElementById('barPanelDiff');
      const blD = document.getElementById('barListDiff');
      const btD = document.getElementById('barTitleDiff');
      const cn = document.getElementById('compareNote');
      const diffs = d.joint_alt_vs_groot_diff_rms || [];
      const ci = d.joint_compare_info || {};
      if (diffs.length && ci.ok) {
        jpD.style.display = 'block';
        if (btD) {
          btD.textContent = 'GR00T vs ALT rank-1 chunk — ' + (d.joint_diff_description || '');
        }
        if (cn) {
          cn.textContent = 'Aligned T=' + ci.T_compare + ' steps, D=' + ci.D_compare +
            ' joints (min of GR00T T=' + ci.T_gr00t + '/ALT T=' + ci.T_alt +
            ', D=' + ci.D_gr00t + '/' + ci.D_alt + ').';
        }
        fillBarListAbsScale(blD, diffs, 'bar-fill-diff');
      } else {
        jpD.style.display = 'none';
        if (cn) cn.textContent = '';
      }
      const egp = document.getElementById('errGridPanel');
      const egw = document.getElementById('errGridWrap');
      const gdeg = d.err_time_joint_deg;
      const clabs = d.err_joint_col_labels;
      if (egp && egw && gdeg && gdeg.length && clabs && clabs.length) {
        egp.style.display = 'block';
        renderErrGrid(egw, gdeg, clabs);
      } else if (egp) {
        egp.style.display = 'none';
        if (egw) egw.innerHTML = '';
      }
    }
    const es = new EventSource('/stream');
    es.onmessage = function (ev) {
      try { render(JSON.parse(ev.data)); } catch (e) { console.warn(e); }
    };
    es.onerror = function () { /* reconnects automatically */ };
    fetch('/api/latest').then(function (r) { return r.json(); }).then(render).catch(function () {});
  </script>
</body>
</html>
"""


def _wait_seq_after(viz_state: InferenceVizState, after: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = viz_state.get_seq()
        if s > after:
            return s
        time.sleep(0.025)
    return viz_state.get_seq()


def create_viz_app(viz_state: InferenceVizState) -> FastAPI:
    app = FastAPI(title="GR00T ALT viz", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _HTML_PAGE

    @app.get("/api/latest")
    async def api_latest() -> dict[str, Any]:
        seq = viz_state.get_seq()
        meta = viz_state.get_meta_copy()
        if meta is None:
            return {"seq": seq}
        out = dict(meta)
        out["seq"] = seq
        return out

    @app.get("/thumb")
    async def thumb(
        kind: str,
        cam: int,
        seq: int,
        neighbor: int = 0,
    ) -> Response:
        if kind == "query":
            data = viz_state.get_jpeg_query(cam)
        elif kind == "match":
            data = viz_state.get_jpeg_match(neighbor, cam)
        else:
            raise HTTPException(400, "kind must be query or match")
        if not data:
            raise HTTPException(404, "no image")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            last = -1
            while True:
                new_seq = await asyncio.to_thread(_wait_seq_after, viz_state, last, 45.0)
                meta = viz_state.get_meta_copy()
                if meta is None:
                    payload = json.dumps({"seq": new_seq})
                else:
                    payload_obj = dict(meta)
                    payload_obj["seq"] = new_seq
                    payload = json.dumps(payload_obj)
                yield f"data: {payload}\n\n"
                last = new_seq

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
