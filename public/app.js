// Clime — forecast reliability dashboard logic
// All requests use relative paths (/api/...) so this works identically on
// localhost and on Vercel. If any request fails, the UI falls back to a
// cached in-memory demo snapshot rather than showing a blank page.

const API = ""; // relative — same-origin
let state = {
  day: 4,
  region: "Rajasthan",
  explainRegion: "Rajasthan",
  explainDay: 4,
  regionsList: [],
  map: null,
  markers: [],
  charts: {},
  riskFilter: "all",
  regionSearch: "",
  liveSnapshot: null,
  liveFetchedAt: 0,
  liveRegionDetails: new Map(),
};

const LIVE_CLIENT_TTL_MS = 14 * 60 * 1000;

// ---------------------------------------------------------------
// Fetch helper with graceful fallback (spec #40: never show a blank page)
// ---------------------------------------------------------------
async function apiGet(path, fallback = null) {
  try {
    const res = await fetch(API + path);
    if (!res.ok) throw new Error("HTTP " + res.status);
    return await res.json();
  } catch (e) {
    console.warn("API fallback used for", path, e);
    return fallback;
  }
}

async function apiPost(path, payload = {}, fallback = null) {
  try {
    const res = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    return await res.json();
  } catch (e) {
    console.warn("API fallback used for", path, e);
    return fallback;
  }
}

function fmtPct(n) { return (n === null || n === undefined) ? "—" : `${n}%`; }

function riskClass(level) {
  if (level === "HIGH") return "red";
  if (level === "MODERATE") return "orange";
  return "green";
}

function updateLiveStatus(data, isDemo = false) {
  const status = document.getElementById("liveStatusText");
  const updated = document.getElementById("liveUpdated");
  const badge = document.getElementById("liveStatus");
  if (!status || !updated || !badge) return;
  const dataTag = document.querySelector(".data-tag");

  badge.classList.toggle("stale", data?.freshness === "stale" || isDemo);
  if (isDemo) {
    status.textContent = "DEMO DATA · LIVE FEED UNAVAILABLE";
    updated.textContent = "Showing synthetic prototype snapshot";
    if (dataTag) dataTag.textContent = "Prototype / Demonstration Data";
    return;
  }

  const freshness = data.freshness === "stale" ? "STALE LIVE DATA" : "LIVE OPEN-METEO";
  status.textContent = freshness;
  if (dataTag) dataTag.textContent = "Live NWP · Experimental risk";
  const retrieved = data.retrieved_at ? new Date(data.retrieved_at).toLocaleString() : "time unavailable";
  updated.textContent = `${data.model_resolution} · retrieved ${retrieved} · experimental risk model`;
}

async function loadLiveSnapshot(force = false) {
  if (!force && state.liveSnapshot && Date.now() - state.liveFetchedAt < LIVE_CLIENT_TTL_MS) {
    return state.liveSnapshot;
  }
  const path = force ? "/api/live-forecast?refresh=true" : "/api/live-forecast";
  const data = await apiGet(path, null);
  if (data?.mode === "live" && Array.isArray(data.regions)) {
    if (state.liveSnapshot?.retrieved_at !== data.retrieved_at) state.liveRegionDetails.clear();
    state.liveSnapshot = data;
    state.liveFetchedAt = Date.now();
    updateLiveStatus(data);
    return data;
  }
  if (state.liveSnapshot) {
    state.liveSnapshot = { ...state.liveSnapshot, freshness: "stale" };
    updateLiveStatus(state.liveSnapshot);
    return state.liveSnapshot;
  }
  updateLiveStatus(null, true);
  return null;
}

function findLiveCell(region, day) {
  return state.liveSnapshot?.regions.find(cell => cell.region === region && cell.day === day) || null;
}

// ---------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------
function initNav() {
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  document.getElementById("menuToggle")?.addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
  });
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  const viewEl = document.getElementById("view-" + view);
  if (viewEl) viewEl.classList.add("active");
  document.getElementById("hero").style.display = view === "dashboard" ? "block" : "none";
  document.getElementById("sidebar").classList.remove("open");

  if (view === "decision") loadDecisionCenterView();
  if (view === "region") loadRegionView();
  if (view === "explain") loadExplainView();
  if (view === "replay") loadReplayView();
  if (view === "performance") loadPerformanceView();
  if (view === "api") loadApiView();
  window.scrollTo({ top: 0, behavior: "instant" });
}

function refreshMap() {
  if (state.mapViewMode === "district") {
    loadDistrictMap(state.day);
  } else {
    loadMap(state.day);
  }
}

function initDashboardFilters() {
  const searchInput = document.getElementById("regionSearch");
  if (!searchInput) return;

  searchInput.addEventListener("input", (event) => {
    state.regionSearch = event.target.value.trim().toLowerCase();
    refreshMap();
  });

  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.toggle("active", b === btn));
      state.riskFilter = btn.dataset.level || "all";
      refreshMap();
    });
  });
}

function filterMapRegions(regions) {
  return regions.filter(region => {
    const riskMatch = state.riskFilter === "all" || region.risk_level === state.riskFilter;
    const searchMatch = !state.regionSearch || region.region.toLowerCase().includes(state.regionSearch);
    return riskMatch && searchMatch;
  });
}

// ---------------------------------------------------------------
// DASHBOARD
// ---------------------------------------------------------------
async function loadDashboard(force = false) {
  const snapshot = await loadLiveSnapshot(force);
  await Promise.all([loadKPIs(snapshot), loadMap(state.day, snapshot), loadTrendCharts(snapshot), loadAlerts(snapshot)]);
}

async function loadKPIs(snapshot = state.liveSnapshot) {
  if (snapshot?.mode === "live") {
    const cells = snapshot.regions;
    const averageConfidence = cells.reduce((total, cell) => total + cell.confidence, 0) / cells.length;
    const highRisk = new Set(cells.filter(cell => cell.risk_level === "HIGH").map(cell => cell.region));
    const errorProne = cells.filter(cell => cell.error_prone_area).length;
    const highest = cells.reduce((best, cell) => cell.bust_probability > best.bust_probability ? cell : best, cells[0]);
    const byDay = Array.from({ length: 10 }, (_, index) => {
      const dayCells = cells.filter(cell => cell.day === index + 1);
      return { day: index + 1, confidence: dayCells.reduce((total, cell) => total + cell.confidence, 0) / (dayCells.length || 1) };
    });
    const uncertainDay = byDay.reduce((worst, day) => day.confidence < worst.confidence ? day : worst, byDay[0]);
    document.getElementById("kpiGrid").innerHTML = `
      <div class="kpi-card"><div class="kpi-label">Average Confidence</div><div class="kpi-value">${averageConfidence.toFixed(1)}%</div></div>
      <div class="kpi-card"><div class="kpi-label">High Bust-Risk Regions</div><div class="kpi-value">${highRisk.size}</div><div class="kpi-sub">of ${new Set(cells.map(cell => cell.region)).size} monitored</div></div>
      <div class="kpi-card"><div class="kpi-label">Error-Prone Region-Days</div><div class="kpi-value">${errorProne}</div><div class="kpi-sub">across Day 1–10</div></div>
      <div class="kpi-card"><div class="kpi-label">Highest Bust Probability</div><div class="kpi-value">${highest.bust_probability}%</div><div class="kpi-sub">${highest.region} · Day ${highest.day}</div></div>
      <div class="kpi-card"><div class="kpi-label">Least-Confident Lead</div><div class="kpi-value">Day ${uncertainDay.day}</div></div>
    `;
    return;
  }
  const kpi = await apiGet("/api/summary");
  const grid = document.getElementById("kpiGrid");
  if (!kpi) { grid.innerHTML = `<div class="kpi-card">Demo data unavailable</div>`; return; }
  grid.innerHTML = `
    <div class="kpi-card"><div class="kpi-label">Average Confidence</div><div class="kpi-value">${kpi.average_confidence}%</div></div>
    <div class="kpi-card"><div class="kpi-label">High-Risk Regions</div><div class="kpi-value">${kpi.high_risk_regions_count}</div><div class="kpi-sub">of ${kpi.regions_monitored} monitored</div></div>
    <div class="kpi-card"><div class="kpi-label">Error-prone cells</div><div class="kpi-value">${kpi.error_prone_cells ?? "—"}</div><div class="kpi-sub">region × lead time</div></div>
    <div class="kpi-card"><div class="kpi-label">Highest Bust Probability</div><div class="kpi-value">${kpi.highest_bust_probability}%</div><div class="kpi-sub">${kpi.highest_bust_region} • Day ${kpi.highest_bust_day}</div></div>
    <div class="kpi-card"><div class="kpi-label">Most Uncertain Lead Time</div><div class="kpi-value">${kpi.most_uncertain_lead_time}</div></div>
  `;
}

function buildDaySelector() {
  const wrap = document.getElementById("daySelector");
  wrap.innerHTML = "";
  for (let d = 1; d <= 10; d++) {
    const btn = document.createElement("button");
    btn.className = "day-btn" + (d === state.day ? " active" : "");
    btn.textContent = "DAY " + d;
    btn.addEventListener("click", () => {
      state.day = d;
      document.querySelectorAll(".day-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (state.mapViewMode === "district") {
        loadDistrictMap(d);
        if (state.districtCode) showDistrictDetail(state.districtCode, d);
      } else {
        loadMap(d);
        if (state.region) showRegionDetail(state.region, d);
      }
    });
    wrap.appendChild(btn);
  }
}

async function loadMap(day, snapshot = state.liveSnapshot) {
  if (!snapshot && !state.liveSnapshot) snapshot = await loadLiveSnapshot();
  const data = snapshot?.mode === "live"
    ? { regions: snapshot.regions.filter(region => region.day === day) }
    : await apiGet(`/api/forecast-map?day=${day}`);
  if (!state.map) {
    state.map = L.map("map", { scrollWheelZoom: false }).setView([22.5, 80], 4.6);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
      subdomains: ["a", "b", "c"],
    }).addTo(state.map);
  }
  state.markers.forEach(m => state.map.removeLayer(m));
  state.markers = [];

  if (!data) return;

  const visibleRegions = filterMapRegions(data.regions);
  const colorFor = (level) => level === "HIGH" ? "#D6425C" : level === "MODERATE" ? "#C97A0F" : "#1C9A6C";

  visibleRegions.forEach(r => {
    const radius = 9 + (100 - r.confidence) / 6;
    const marker = L.circleMarker([r.lat, r.lon], {
      radius,
      color: "#fff",
      weight: 2,
      fillColor: colorFor(r.risk_level),
      fillOpacity: 0.85,
      className: "custom-pin",
    }).addTo(state.map);

    marker.bindTooltip(`<strong>${r.region}</strong><br>Bust probability: ${r.bust_probability}%<br>Confidence: ${r.confidence}%<br>Risk: ${r.risk_level}<br>Error-prone: ${r.error_prone_area ? "YES" : "NO"}`, { direction: "top" });
    marker.on("click", () => showRegionDetail(r.region, day));
    state.markers.push(marker);
  });

  const emptyState = document.getElementById("regionDetailEmpty");
  if (visibleRegions.length === 0 && emptyState) {
    emptyState.style.display = "block";
    emptyState.innerHTML = "<p>No regions match the current risk filter or search.</p>";
    document.getElementById("regionDetailContent").style.display = "none";
  } else if (emptyState) {
    emptyState.style.display = visibleRegions.length ? "none" : "block";
  }

  const proneList = document.getElementById("errorProneList");
  if (proneList) {
    const prone = visibleRegions.filter(r => r.error_prone_area).sort((a, b) => b.bust_probability - a.bust_probability);
    proneList.innerHTML = prone.length
      ? prone.slice(0, 8).map(r => `<button class="error-prone-item" data-region="${r.region}">${r.region}<span>${r.bust_probability}%</span></button>`).join("")
      : "<p class='muted'>No error-prone flags at this lead time among the current filter.</p>";
    proneList.querySelectorAll(".error-prone-item").forEach(btn => {
      btn.addEventListener("click", () => showRegionDetail(btn.dataset.region, day));
    });
  }
}

// ---------------------------------------------------------------
// DISTRICT MAP (added — was never wired into this "Clime" rebuild)
// ---------------------------------------------------------------
let districtGeoJSON = null;
let districtLayer = null;

function colorForRisk(level) {
  return level === "HIGH" ? "#D6425C" : level === "MODERATE" ? "#C97A0F" : "#1C9A6C";
}

async function loadDistrictGeoJSON() {
  if (districtGeoJSON) return districtGeoJSON;
  try {
    const res = await fetch("/india-districts.geojson");
    if (!res.ok) throw new Error("HTTP " + res.status);
    districtGeoJSON = await res.json();
  } catch (e) {
    console.warn("District boundaries unavailable at /india-districts.geojson", e);
    districtGeoJSON = { type: "FeatureCollection", features: [] };
  }
  return districtGeoJSON;
}

async function loadDistrictMap(day) {
  const [geo, data] = await Promise.all([
    loadDistrictGeoJSON(),
    apiGet(`/api/district-map?day=${day}`),
  ]);

  if (!state.map) {
    state.map = L.map("map", { scrollWheelZoom: false }).setView([22.5, 80], 4.6);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
      subdomains: ["a", "b", "c"],
    }).addTo(state.map);
  }

  // Clear whichever layer type was showing before
  state.markers.forEach(m => state.map.removeLayer(m));
  state.markers = [];
  if (districtLayer) { state.map.removeLayer(districtLayer); districtLayer = null; }

  const emptyState = document.getElementById("regionDetailEmpty");
  const content = document.getElementById("regionDetailContent");

  if (!geo.features.length) {
    if (emptyState) {
      emptyState.style.display = "block";
      emptyState.innerHTML = "<p><strong>District boundaries not found.</strong> Expected a file at <code>/india-districts.geojson</code> — check it exists in <code>public/</code>.</p>";
    }
    return;
  }
  if (!data || !data.districts) {
    if (emptyState) {
      emptyState.style.display = "block";
      emptyState.innerHTML = "<p><strong>No district data returned</strong> from /api/district-map for this day.</p>";
    }
    return;
  }

  // Defensive lookups: field names are read from whatever the backend actually
  // sends, trying the most likely key first, since district_snapshot.json's
  // exact schema wasn't visible when writing this.
  const byCode = {};
  data.districts.forEach(d => { byCode[d.district_code] = d; });

  districtLayer = L.geoJSON(geo, {
    style: (feature) => {
      const props = feature.properties || {};
      const code = props.district_code || props.dt_code || props.code;
      const d = byCode[code];
      const matchesRisk = state.riskFilter === "all" || (d && d.risk_level === state.riskFilter);
      const matchesSearch = !state.regionSearch || (d && (d.district.toLowerCase().includes(state.regionSearch) || d.state.toLowerCase().includes(state.regionSearch)));
      const visible = matchesRisk && matchesSearch;
      return {
        fillColor: d ? colorForRisk(d.risk_level) : "#ccc",
        fillOpacity: visible ? 0.75 : 0.08,
        color: visible ? "#ffffff" : "#444444",
        weight: visible ? 0.6 : 0.2,
      };
    },
    onEachFeature: (feature, layer) => {
      const props = feature.properties || {};
      const code = props.district_code || props.dt_code || props.code;
      const d = byCode[code];
      if (!d) return;
      const districtName = d.district || props.district || "Unknown district";
      const stateName = d.state || props.state || "";
      layer.bindTooltip(
        `<strong>${districtName}, ${stateName}</strong><br>Bust probability: ${d.bust_probability}%<br>Confidence: ${d.confidence}%<br>Risk: ${d.risk_level}`,
        { sticky: true }
      );
      layer.on({
        mouseover: (e) => e.target.setStyle({ weight: 2, color: "#10233F" }),
        mouseout: (e) => districtLayer.resetStyle(e.target),
        click: () => showDistrictDetail(code, day),
      });
    },
  }).addTo(state.map);

  if (emptyState) emptyState.style.display = "none";

  // Populate error-prone / high-risk district list in dashboard sidebar
  const proneList = document.getElementById("errorProneList");
  if (proneList && data.districts) {
    const visibleDistricts = data.districts.filter(d => {
      const matchesRisk = state.riskFilter === "all" || d.risk_level === state.riskFilter;
      const matchesSearch = !state.regionSearch || (d.district.toLowerCase().includes(state.regionSearch) || d.state.toLowerCase().includes(state.regionSearch));
      return matchesRisk && matchesSearch;
    });
    const prone = visibleDistricts.filter(d => d.error_prone_area || d.risk_level === "HIGH").sort((a, b) => b.bust_probability - a.bust_probability);
    proneList.innerHTML = prone.length
      ? prone.slice(0, 8).map(d => `<button class="error-prone-item" data-code="${d.district_code}">${d.district} (${d.state})<span>${d.bust_probability}%</span></button>`).join("")
      : "<p class='muted'>No high-risk or error-prone districts match the current filter.</p>";
    proneList.querySelectorAll(".error-prone-item").forEach(btn => {
      btn.addEventListener("click", () => showDistrictDetail(btn.dataset.code, day));
    });
  }
}

async function showDistrictDetail(districtCode, day) {
  state.districtCode = districtCode;
  const detail = await apiGet(`/api/district/${encodeURIComponent(districtCode)}?day=${day}`);
  const emptyState = document.getElementById("regionDetailEmpty");
  const content = document.getElementById("regionDetailContent");
  if (emptyState) emptyState.style.display = "none";
  content.style.display = "block";
  if (!detail) { content.innerHTML = "<p class='muted'>District details unavailable.</p>"; return; }

  const contributors = (detail.shap_contributors && detail.shap_contributors.length)
    ? detail.shap_contributors.map(c => c.label || c)
    : (detail.top_contributors || []);

  content.innerHTML = `
    <div class="ops-card">
      <div class="ops-title">${(detail.district || "").toUpperCase()}, ${(detail.state || "").toUpperCase()} — DAY ${detail.day}</div>
      <div class="ops-row"><span class="ops-label">Bust Probability</span><span>${detail.bust_probability}%</span></div>
      <div class="ops-row"><span class="ops-label">Confidence</span><span>${detail.confidence}%</span></div>
      <div class="ops-row"><span class="ops-label">Risk</span><span class="risk-badge ${detail.risk_level}">${detail.risk_level}</span></div>
      <div class="ops-factors">Contributing factors:
        <ul>${contributors.slice(0, 5).map(c => `<li>${c}</li>`).join("")}</ul>
      </div>
    </div>
    <div style="margin-top:12px; display:flex; flex-direction:column; gap:8px;">
      <a class="rd-link" id="viewDistrictInDecision" style="cursor:pointer; font-weight:600;">⚡ Open in AI Decision Center →</a>
      <a class="rd-link" id="viewDistrictInRegion" style="cursor:pointer; font-weight:600;">📊 Open in Regional Analysis →</a>
    </div>
    <p class="muted" style="margin-top:10px; font-size:11px;">Operational district reliability (554 Districts / 18 States).</p>
  `;

  content.querySelector("#viewDistrictInDecision")?.addEventListener("click", async () => {
    switchView("decision");
    const stateSel = document.getElementById("decisionState");
    const daySel = document.getElementById("decisionDay");
    if (stateSel) {
      stateSel.value = detail.state;
      await updateDecisionDistricts(detail.district_code);
    }
    if (daySel) daySel.value = String(day);
    runDecisionAnalysis();
  });

  content.querySelector("#viewDistrictInRegion")?.addEventListener("click", async () => {
    state.region = detail.state;
    state.regionDistrict = detail.district_code;
    switchView("region");
  });
}

function initMapViewToggle() {
  const toggle = document.getElementById("mapViewToggle");
  if (!toggle) return;
  toggle.querySelectorAll("[data-mapview]").forEach(btn => {
    btn.addEventListener("click", () => {
      toggle.querySelectorAll("[data-mapview]").forEach(b => b.classList.toggle("active", b === btn));
      state.mapViewMode = btn.dataset.mapview;
      if (state.mapViewMode === "district") {
        loadDistrictMap(state.day);
      } else {
        loadMap(state.day);
      }
    });
  });
}

async function showRegionDetail(region, day) {
  const detail = findLiveCell(region, day) || await apiGet(`/api/forecast/${encodeURIComponent(region)}?day=${day}`);
  document.getElementById("regionDetailEmpty").style.display = "none";
  const content = document.getElementById("regionDetailContent");
  content.style.display = "block";
  if (!detail) { content.innerHTML = "<p class='muted'>Details unavailable.</p>"; return; }

  const contributors = (detail.shap_contributors && detail.shap_contributors.length)
    ? detail.shap_contributors.map(c => c.label || c)
    : (detail.top_contributors || []);
  const prone = detail.error_prone_area ? "YES" : "NO";

  content.innerHTML = `
    <div class="ops-card">
      <div class="ops-title">${detail.region.toUpperCase()} — DAY ${detail.day}</div>
      <div class="ops-row"><span class="ops-label">Forecast Bust Probability</span><span>${detail.bust_probability}%</span></div>
      <div class="ops-row"><span class="ops-label">Forecast Confidence</span><span>${detail.confidence}%</span></div>
      <div class="ops-row"><span class="ops-label">Bust Risk</span><span class="risk-badge ${detail.risk_level}">${detail.risk_level}</span></div>
      <div class="ops-row"><span class="ops-label">Error-Prone Area</span><span>${prone}</span></div>
      <div class="ops-factors">Major model contributors:
        <ul>${contributors.slice(0, 4).map(c => `<li>${c}</li>`).join("")}</ul>
      </div>
    </div>
    <div class="rd-metric"><span>NWP precipitation</span><span class="val">${detail.forecast_precipitation} mm</span></div>
    <div class="rd-metric"><span>Valid date</span><span class="val">${detail.valid_date || "Demo"}</span></div>
    <div class="rd-metric"><span>Historical precip error (mean)</span><span class="val">${detail.mean_historical_precip_error ?? "—"} mm</span></div>
    <p class="explain-copy" style="margin-top:12px;">${detail.explanation || ""}</p>
    <a class="rd-link" id="viewDetailedAnalysis">View Detailed Analysis →</a>
  `;
  document.getElementById("viewDetailedAnalysis").addEventListener("click", () => {
    state.region = region;
    state.explainRegion = region;
    state.explainDay = day;
    switchView("region");
  });

  const currentDay = document.getElementById("simLeadTime");
  if (currentDay) {
    currentDay.value = String(day);
    document.getElementById("simLeadTimeVal").textContent = `Day ${day}`;
  }

  const detailPayload = {
    precipitation: detail.forecast_precipitation || 38,
    temperature: detail.forecast_temperature || 28,
    pressure: detail.forecast_pressure || 1012,
    humidity: detail.forecast_humidity || 74,
    wind_speed: detail.forecast_wind_speed || 18,
    wind_direction: 180,
    lead_time: day,
    latitude: detail.latitude || 22.5,
    longitude: detail.longitude || 78.5,
    spatial_gradient: detail.spatial_gradient || 5.5,
    historical_error: detail.historical_error || 8,
    pressure_variation: detail.pressure_variation || 4.2,
    precipitation_variability: detail.precipitation_variability || 12,
  };

  Object.entries(detailPayload).forEach(([key, value]) => {
    const el = document.getElementById(`sim${key.charAt(0).toUpperCase()}${key.slice(1)}`);
    if (el && !Number.isNaN(Number(value))) {
      el.value = Number(value);
      const label = document.getElementById(`${el.id}Val`);
      if (label) {
        if (key === "lead_time") label.textContent = `Day ${value}`;
        else if (key === "precipitation") label.textContent = `${value} mm`;
        else if (key === "temperature") label.textContent = `${value}°C`;
        else if (key === "humidity") label.textContent = `${value}%`;
        else if (key === "wind_speed") label.textContent = `${value} km/h`;
        else label.textContent = value;
      }
    }
  });
}


// ---------------------------------------------------------------
// AI WEATHER DECISION CENTER
// ---------------------------------------------------------------
async function loadDecisionCenterView() {
  const stateSel = document.getElementById("decisionState");
  const distSel = document.getElementById("decisionDistrict");
  const daySel = document.getElementById("decisionDay");
  const analyzeBtn = document.getElementById("analyzeDecisionBtn");

  if (!stateSel) return;

  const regions = await ensureRegionsList();
  if (!stateSel.dataset.filled) {
    stateSel.innerHTML = '<option value="">Select a state…</option>' +
      regions.map(r => `<option value="${r}">${r}</option>`).join("");
    stateSel.dataset.filled = "1";

    stateSel.addEventListener("change", async () => {
      await updateDecisionDistricts();
      runDecisionAnalysis();
    });

    distSel?.addEventListener("change", () => {
      runDecisionAnalysis();
    });

    daySel?.addEventListener("change", () => {
      runDecisionAnalysis();
    });

    analyzeBtn?.addEventListener("click", () => {
      runDecisionAnalysis();
    });
  }

  // Pre-select current state if none selected
  if (!stateSel.value && regions.length) {
    stateSel.value = state.region || regions[0];
    await updateDecisionDistricts();
  }

  runDecisionAnalysis();
}

async function updateDecisionDistricts(selectedCode = null) {
  const stateSel = document.getElementById("decisionState");
  const distSel = document.getElementById("decisionDistrict");
  const daySel = document.getElementById("decisionDay");
  if (!stateSel || !distSel) return;

  const selectedState = stateSel.value;
  if (!selectedState) {
    distSel.innerHTML = '<option value="">All districts (state level)</option>';
    distSel.disabled = true;
    return;
  }

  distSel.disabled = false;
  distSel.innerHTML = '<option value="">All districts (state level)</option>';

  const day = daySel ? parseInt(daySel.value) : (state.day || 4);
  try {
    const data = await apiGet(`/api/district-map?day=${day}&state=${encodeURIComponent(selectedState)}`, { districts: [] });
    const districts = (data.districts || []).sort((a, b) => a.district.localeCompare(b.district));
    districts.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.district_code || d.district;
      opt.textContent = `${d.district} (Code: ${d.district_code || '—'})`;
      distSel.appendChild(opt);
    });
    if (selectedCode) {
      distSel.value = selectedCode;
    }
  } catch (err) {
    console.warn("Could not fetch districts for", selectedState, err);
  }
}

async function runDecisionAnalysis() {
  const stateSel = document.getElementById("decisionState");
  const distSel = document.getElementById("decisionDistrict");
  const daySel = document.getElementById("decisionDay");
  const loadingEl = document.getElementById("decisionLoading");
  const errorEl = document.getElementById("decisionError");
  const resultsEl = document.getElementById("decisionResults");
  const analyzeBtn = document.getElementById("analyzeDecisionBtn");

  if (!stateSel || !stateSel.value) {
    if (resultsEl) resultsEl.style.display = "none";
    return;
  }

  const stateVal = stateSel.value;
  const distVal = distSel ? distSel.value : "";
  const dayVal = daySel ? parseInt(daySel.value) : 4;

  if (loadingEl) loadingEl.style.display = "block";
  if (errorEl) errorEl.style.display = "none";
  if (resultsEl) resultsEl.style.display = "none";
  if (analyzeBtn) analyzeBtn.disabled = true;

  try {
    let url = `/api/decision-center?state=${encodeURIComponent(stateVal)}&day=${dayVal}`;
    if (distVal) {
      url += `&district=${encodeURIComponent(distVal)}`;
    }
    const res = await apiGet(url);
    if (!res || res.detail) {
      throw new Error(res?.detail || "No decision data available");
    }

    // Populate Results
    document.getElementById("bustProbability").textContent = `${res.bust_probability}%`;
    document.getElementById("forecastConfidence").textContent = `${res.confidence}%`;
    const riskEl = document.getElementById("decisionRiskLevel");
    riskEl.textContent = res.risk_level;
    riskEl.className = `decision-value risk-badge ${res.risk_level}`;
    riskEl.style.display = "inline-block";

    document.getElementById("decisionErrorProneTag").textContent = res.error_prone_area
      ? "⚠ Historical high-error terrain/regime"
      : "Area Status: Normal Terrain";
    document.getElementById("decisionLeadTime").textContent = `Day ${res.day || dayVal}`;
    document.getElementById("decisionLocation").textContent = res.district
      ? `${res.district}, ${res.state}`
      : `${res.state} (State aggregate)`;

    // Weather Context
    const wx = res.weather_context || {};
    document.getElementById("dwPrecip").textContent = wx.precipitation != null ? `${wx.precipitation} mm` : "--";
    document.getElementById("dwTemp").textContent = wx.temperature != null ? `${wx.temperature} °C` : "--";
    document.getElementById("dwPressure").textContent = wx.pressure != null ? `${wx.pressure} hPa` : "--";
    document.getElementById("dwWind").textContent = wx.wind_speed != null ? `${wx.wind_speed} km/h` : "--";

    // Feature drivers
    const contribList = document.getElementById("decisionContributors");
    const drivers = res.feature_drivers || [];
    if (drivers.length) {
      contribList.innerHTML = drivers.map(d => {
        const sign = d.direction === "increases_risk" ? "🔺 increases uncertainty" : "🔹 stabilizes forecast";
        const val = d.importance ? `(${Math.round(d.importance * 100)}% relative weight)` : "";
        return `<li><strong>${d.feature}</strong>: ${sign} ${val}</li>`;
      }).join("");
    } else {
      contribList.innerHTML = `<li class="muted">Standard variance profile; no anomalous drivers.</li>`;
    }

    // Narrative
    document.getElementById("decisionExplanation").textContent = res.narrative || res.explanation || "No explanation available.";

    // Recommendations
    const recList = document.getElementById("decisionRecommendations");
    const recs = res.recommendations || [];
    if (recs.length) {
      recList.innerHTML = recs.map(r => `<li>${r}</li>`).join("");
    } else {
      recList.innerHTML = `<li>Standard operating procedures apply; uncertainty is within normal bounds.</li>`;
    }

    if (resultsEl) resultsEl.style.display = "block";
  } catch (err) {
    if (errorEl) {
      errorEl.textContent = `Error loading decision analysis: ${err.message || err}`;
      errorEl.style.display = "block";
    }
  } finally {
    if (loadingEl) loadingEl.style.display = "none";
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

async function loadTrendCharts(snapshot = state.liveSnapshot) {
  // Aggregate average confidence & bust probability across all regions, per day
  const days = [1,2,3,4,5,6,7,8,9,10];
  const conf = [], bust = [];
  if (snapshot?.mode === "live") {
    days.forEach(day => {
      const cells = snapshot.regions.filter(cell => cell.day === day);
      const average = cells.reduce((total, cell) => total + cell.confidence, 0) / (cells.length || 1);
      conf.push(Number(average.toFixed(1)));
      bust.push(Number((100 - average).toFixed(1)));
    });
    drawLineChart("confidenceChart", days.map(d => "Day " + d), [{ label: "Avg Confidence %", data: conf, color: "#1B5FE0" }]);
    drawLineChart("bustChart", days.map(d => "Day " + d), [{ label: "Avg Bust Probability %", data: bust, color: "#D6425C" }]);
    return;
  }
  const results = await Promise.all(days.map(d => apiGet(`/api/day-analysis/${d}`)));
  results.forEach((r, i) => {
    conf.push(r ? r.average_confidence : null);
    bust.push(r ? Math.round((100 - r.average_confidence) * 10) / 10 : null);
  });
  drawLineChart("confidenceChart", days.map(d => "Day " + d), [{ label: "Avg Confidence %", data: conf, color: "#1B5FE0" }]);
  drawLineChart("bustChart", days.map(d => "Day " + d), [{ label: "Avg Bust Probability %", data: bust, color: "#D6425C" }]);
}

async function loadAlerts(snapshot = state.liveSnapshot) {
  if (snapshot?.mode === "live") {
    const bestByRegion = new Map();
    snapshot.regions.forEach(cell => {
      const current = bestByRegion.get(cell.region);
      if (!current || cell.bust_probability > current.bust_probability) bestByRegion.set(cell.region, cell);
    });
    const alerts = [...bestByRegion.values()]
      .sort((first, second) => second.bust_probability - first.bust_probability)
      .slice(0, 6)
      .map(cell => ({
        region: cell.region,
        day: cell.day,
        bust_probability: cell.bust_probability,
        level: cell.risk_level === "HIGH" ? "HIGH RISK" : cell.risk_level === "MODERATE" ? "MODERATE RISK" : "LOW RISK",
      }));
    renderAlerts(alerts);
    return;
  }
  const data = await apiGet("/api/alerts", { alerts: [] });
  renderAlerts(data.alerts);
}

function renderAlerts(alerts) {
  const list = document.getElementById("alertsList");
  if (!alerts.length) { list.innerHTML = "<p class='muted'>No alerts available.</p>"; return; }
  list.innerHTML = alerts.map(a => {
    const cls = a.level === "HIGH RISK" ? "high" : a.level === "MODERATE RISK" ? "moderate" : "low";
    const icon = a.level === "HIGH RISK" ? "🔴" : a.level === "MODERATE RISK" ? "🟠" : "🟢";
    return `<div class="alert-row ${cls}" data-region="${a.region}" data-day="${a.day}">
      <span class="alert-icon">${icon}</span>
      <span class="alert-region">${a.region} — Day ${a.day}</span>
      <span class="alert-badge">${a.level}</span>
      <span class="alert-prob">${a.bust_probability}%</span>
    </div>`;
  }).join("");
  list.querySelectorAll(".alert-row").forEach(row => {
    row.addEventListener("click", () => {
      state.region = row.dataset.region;
      state.explainRegion = row.dataset.region;
      state.explainDay = parseInt(row.dataset.day);
      switchView("region");
    });
  });
}

// ---------------------------------------------------------------
// POPULATE DISTRICT DROPDOWN
// ---------------------------------------------------------------
// POPULATE DISTRICT DROPDOWN
// ---------------------------------------------------------------
async function populateRegionDistricts(region, selectedCode = "", day = null) {
  const distSel = document.getElementById("regionDistrictSelect");
  if (!distSel) return;

  distSel.innerHTML = `
    <option value="">All districts (state level)</option>
  `;

  if (!region) return;

  try {
    const dayVal = day || document.getElementById("regionDaySelect")?.value || state.day;
    const data = await apiGet(`/api/district-map?day=${dayVal}&state=${encodeURIComponent(region)}`, {
      districts: []
    });

    const districts = Array.isArray(data?.districts) ? data.districts : [];

    const sortedDistricts = districts.slice().sort((a, b) =>
      String(a.district || "").localeCompare(String(b.district || ""))
    );

    sortedDistricts.forEach(d => {
      const option = document.createElement("option");
      option.value = d.district_code;
      option.textContent = `${d.district} (Code: ${d.district_code})`;
      distSel.appendChild(option);
    });

    if (
      selectedCode &&
      [...distSel.options].some(option => option.value === selectedCode)
    ) {
      distSel.value = selectedCode;
    } else {
      distSel.value = "";
    }
  } catch (error) {
    console.error("Failed to load districts:", error);
  }
}

// ---------------------------------------------------------------
// REGION ANALYSIS
// ---------------------------------------------------------------
async function ensureRegionsList() {
  if (state.regionsList.length) return state.regionsList;
  const data = await apiGet("/api/regions", { regions: [] });
  state.regionsList = data.regions;
  return state.regionsList;
}

async function loadRegionView() {
  const regions = await ensureRegionsList();
  const sel = document.getElementById("regionSelect");
  const daySel = document.getElementById("regionDaySelect");
  const distSel = document.getElementById("regionDistrictSelect");
  if (!sel.dataset.filled) {
    sel.innerHTML = regions.map(r => `<option ${r === state.region ? "selected" : ""}>${r}</option>`).join("");
    daySel.innerHTML = Array.from({length:10}, (_,i) => i+1).map(d => `<option value="${d}" ${d === state.day ? "selected" : ""}>Day ${d}</option>`).join("");
    sel.dataset.filled = "1";
    sel.addEventListener("change", async () => {
      state.region = sel.value;
      await populateRegionDistricts(sel.value, "", parseInt(daySel.value));
      renderRegionAnalysis(parseInt(daySel.value));
    });
    daySel.addEventListener("change", async () => {
      const currentDistrict = distSel.value;
      const dayVal = parseInt(daySel.value);
      await populateRegionDistricts(sel.value, currentDistrict, dayVal);
      renderRegionAnalysis(dayVal);
    });
    distSel.addEventListener("change", () => renderRegionAnalysis(parseInt(daySel.value)));
  } else {
    sel.value = state.region;
    daySel.value = state.explainDay || state.day;
  }
  await populateRegionDistricts(sel.value, state.regionDistrict, parseInt(daySel.value));
  state.regionDistrict = "";
  renderRegionAnalysis(parseInt(daySel.value));
}

async function renderRegionAnalysis(day) {
  day = day || state.day;
  const region = document.getElementById("regionSelect").value;
  if (!state.liveSnapshot) await loadLiveSnapshot();
  const districtCode = document.getElementById("regionDistrictSelect")?.value;

if (districtCode && districtCode !== "all") {
  return renderDistrictAnalysis(districtCode, day);
}
  const detail = findLiveCell(region, day) || await apiGet(`/api/forecast/${encodeURIComponent(region)}?day=${day}`);
  if (!detail) return;
  const liveWeather = state.liveSnapshot?.mode === "live"
    ? state.liveSnapshot.locations.find(location => location.region === region)
    : null;
  const liveDetail = liveWeather ? await loadLiveRegionDetail(region) : null;
  if (document.getElementById("regionSelect").value !== region) return;
  renderLiveWeather(liveWeather, liveDetail);

  document.getElementById("regionForecastCards").innerHTML = `
    <div class="metric-card"><div class="m-label">Temperature</div><div class="m-value">${detail.forecast_temperature}°C</div></div>
    <div class="metric-card"><div class="m-label">Precipitation</div><div class="m-value">${detail.forecast_precipitation} mm</div></div>
    <div class="metric-card"><div class="m-label">Pressure</div><div class="m-value">${detail.forecast_pressure} hPa</div></div>
    <div class="metric-card"><div class="m-label">Humidity</div><div class="m-value">${detail.forecast_humidity}%</div></div>
    <div class="metric-card"><div class="m-label">Wind Speed</div><div class="m-value">${detail.forecast_wind_speed} km/h</div></div>
    <div class="metric-card"><div class="m-label">Forecast Horizon</div><div class="m-value">Day ${detail.day}</div></div>
  `;

  document.getElementById("reliabilityBlock").innerHTML = `
    <div class="big-stat">
      <div class="num">${detail.bust_probability}%</div>
      <div class="lbl">BUST PROBABILITY</div>
    </div>
    <div class="rd-metric"><span>Confidence</span><span class="val">${detail.confidence}%</span></div>
    <div class="rd-metric"><span>Bust Risk</span><span class="val"><span class="risk-badge ${detail.risk_level}">${detail.risk_level}</span></span></div>
    <div class="rd-metric"><span>Error-prone area</span><span class="val">${detail.error_prone_area ? "YES" : "NO"}</span></div>
  `;

  const ops = document.getElementById("regionOpsCard");
  if (ops) {
    const contributors = (detail.shap_contributors || []).map(c => c.label);
    ops.innerHTML = `
      <div class="ops-title">${detail.region.toUpperCase()} — DAY ${detail.day}</div>
      <div class="ops-row"><span class="ops-label">Forecast Bust Probability</span><span>${detail.bust_probability}%</span></div>
      <div class="ops-row"><span class="ops-label">Forecast Confidence</span><span>${detail.confidence}%</span></div>
      <div class="ops-row"><span class="ops-label">Bust Risk</span><span class="risk-badge ${detail.risk_level}">${detail.risk_level}</span></div>
      <div class="ops-row"><span class="ops-label">Error-Prone Area</span><span>${detail.error_prone_area ? "YES" : "NO"}</span></div>
      <div class="ops-factors">Major model contributors:
        <ul>${(contributors.length ? contributors : detail.top_contributors).slice(0,4).map(c => `<li>${c}</li>`).join("")}</ul>
      </div>
    `;
  }
  const expl = document.getElementById("regionExplanation");
  if (expl) expl.textContent = detail.explanation || "";

  document.getElementById("regionContributors").innerHTML = (detail.shap_contributors || detail.top_contributors.map(label => ({label}))).map((c, i) =>
    `<li><span class="rank">${i+1}</span>${c.label || c}${c.shap != null ? ` <span class="muted">(${c.shap > 0 ? "+" : ""}${c.shap})</span>` : ""}</li>`
  ).join("");

  const trend = state.liveSnapshot?.mode === "live"
    ? state.liveSnapshot.regions.filter(cell => cell.region === region).sort((a, b) => a.day - b.day)
    : detail.reliability_trend;
  drawLineChart("regionTrendChart", trend.map(t => "Day " + t.day),
    [{ label: "Bust Probability %", data: trend.map(t => t.bust_probability), color: "#D6425C" }]);
}

// ---------------------------------------------------------------
// DISTRICT-LEVEL RELIABILITY ANALYSIS
// ---------------------------------------------------------------
async function renderDistrictAnalysis(districtCode, day) {
  const detail = await apiGet(
    `/api/district/${encodeURIComponent(districtCode)}?day=${day}`
  );

  if (!detail) {
    console.error("District details unavailable:", districtCode);
    return;
  }

  // Update forecast cards
  const forecastCards = document.getElementById("regionForecastCards");

  if (forecastCards) {
    forecastCards.innerHTML = `
      <div class="metric-card">
        <div class="m-label">District</div>
        <div class="m-value">${detail.district || "—"}</div>
      </div>

      <div class="metric-card">
        <div class="m-label">State</div>
        <div class="m-value">${detail.state || "—"}</div>
      </div>

      <div class="metric-card">
        <div class="m-label">Bust Probability</div>
        <div class="m-value">${detail.bust_probability ?? "—"}%</div>
      </div>

      <div class="metric-card">
        <div class="m-label">Confidence</div>
        <div class="m-value">${detail.confidence ?? "—"}%</div>
      </div>

      <div class="metric-card">
        <div class="m-label">Risk Level</div>
        <div class="m-value">${detail.risk_level || "—"}</div>
      </div>

      <div class="metric-card">
        <div class="m-label">Forecast Horizon</div>
        <div class="m-value">Day ${detail.day ?? day}</div>
      </div>
    `;
  }

  // Update reliability section
  const reliability = document.getElementById("reliabilityBlock");

  if (reliability) {
    reliability.innerHTML = `
      <div class="big-stat">
        <div class="num">${detail.bust_probability ?? "—"}%</div>
        <div class="lbl">BUST PROBABILITY</div>
      </div>

      <div class="rd-metric">
        <span>Confidence</span>
        <span class="val">${detail.confidence ?? "—"}%</span>
      </div>

      <div class="rd-metric">
        <span>Bust Risk</span>
        <span class="val">
          <span class="risk-badge ${detail.risk_level || ""}">
            ${detail.risk_level || "—"}
          </span>
        </span>
      </div>
    `;
  }

  // Update operations card
  const ops = document.getElementById("regionOpsCard");

  if (ops) {
    const contributors =
      detail.shap_contributors?.length
        ? detail.shap_contributors.map(c => c.label || c)
        : detail.top_contributors || [];

    ops.innerHTML = `
      <div class="ops-title">
        ${(detail.district || "").toUpperCase()},
        ${(detail.state || "").toUpperCase()} — DAY ${detail.day ?? day}
      </div>

      <div class="ops-row">
        <span class="ops-label">Bust Probability</span>
        <span>${detail.bust_probability ?? "—"}%</span>
      </div>

      <div class="ops-row">
        <span class="ops-label">Confidence</span>
        <span>${detail.confidence ?? "—"}%</span>
      </div>

      <div class="ops-row">
        <span class="ops-label">Risk</span>
        <span class="risk-badge ${detail.risk_level || ""}">
          ${detail.risk_level || "—"}
        </span>
      </div>

      <div class="ops-factors">
        Contributing factors:
        <ul>
          ${contributors.slice(0, 5).map(c => `<li>${c}</li>`).join("")}
        </ul>
      </div>
    `;
  }

  // Update explanation
  const explanation = document.getElementById("regionExplanation");

  if (explanation) {
    explanation.textContent = detail.explanation || "";
  }

  // Update contributors list
  const contributorList = document.getElementById("regionContributors");

  if (contributorList) {
    const contributors =
      detail.shap_contributors?.length
        ? detail.shap_contributors
        : (detail.top_contributors || []).map(label => ({ label }));

    contributorList.innerHTML = contributors.map((c, i) => `
      <li>
        <span class="rank">${i + 1}</span>
        ${c.label || c}
        ${c.shap != null ? ` <span class="muted">(${c.shap > 0 ? "+" : ""}${c.shap})</span>` : ""}
      </li>
    `).join("");
  }

  // Update trend chart
  if (detail.reliability_trend && detail.reliability_trend.length) {
    drawLineChart("regionTrendChart", detail.reliability_trend.map(t => "Day " + t.day),
      [{ label: "Bust Probability %", data: detail.reliability_trend.map(t => t.bust_probability), color: "#D6425C" }]);
  }

  // Update district terrain profile
  const patternBox = document.getElementById("regionErrorPattern");
  if (patternBox) {
    patternBox.innerHTML = `
      <div style="font-weight:700; margin-bottom:6px;">${detail.district} District Reliability Profile</div>
      <p class="muted" style="margin-bottom:8px;">Terrain classification: ${detail.error_prone_area ? "High-variance / Complex Terrain" : "Standard Regional Terrain"}. Lead-time sensitivity: ${detail.day >= 7 ? "Significant degradation past Day 6" : "Nominal uncertainty trajectory"}.</p>
      <div class="rd-metric"><span>Error-Prone Terrain Flag</span><span class="val">${detail.error_prone_area ? "YES" : "NO"}</span></div>
    `;
  }

  const livePanel = document.getElementById("liveWeatherPanel");
  if (livePanel) livePanel.style.display = "none";
} 

async function loadLiveRegionDetail(region) {
  if (state.liveRegionDetails.has(region)) return state.liveRegionDetails.get(region);
  const detail = await apiGet(`/api/live-forecast/${encodeURIComponent(region)}`);
  if (detail) state.liveRegionDetails.set(region, detail);
  return detail;
}

function renderLiveWeather(summary, detail) {
  const panel = document.getElementById("liveWeatherPanel");
  if (!panel) return;
  panel.style.display = summary ? "block" : "none";
  if (!summary) return;

  const current = summary.current || {};
  const currentItems = [
    ["Temperature", current.temperature_2m, "°C"],
    ["Feels like", current.apparent_temperature, "°C"],
    ["Relative humidity", current.relative_humidity_2m, "%"],
    ["Wind speed", current.wind_speed_10m, " km/h"],
    ["Wind gusts", current.wind_gusts_10m, " km/h"],
    ["Precipitation", current.precipitation, " mm"],
    ["Rain", current.rain, " mm"],
    ["Cloud cover", current.cloud_cover, "%"],
  ];
  document.getElementById("liveCurrentConditions").innerHTML = currentItems.map(([label, value, unit]) =>
    `<div class="live-current-item"><span>${label}</span><strong>${value == null ? "—" : `${value}${unit}`}</strong></div>`
  ).join("");

  const retrieved = state.liveSnapshot?.retrieved_at;
  document.getElementById("liveRegionUpdated").textContent = retrieved
    ? `Updated ${new Date(retrieved).toLocaleTimeString()}`
    : "Live data";

  const daily = summary.daily || {};
  const dailyFields = [
    ["weather_code", "—"], ["temperature_2m_min", "°"], ["temperature_2m_max", "°"],
    ["precipitation_sum", " mm"], ["precipitation_probability_max", "%"],
    ["wind_speed_10m_max", " km/h"], ["wind_gusts_10m_max", " km/h"], ["uv_index_max", ""],
  ];
  const valueAt = (key, index, unit) => {
    const value = daily[key]?.[index];
    if (key === "weather_code") return value == null ? "—" : value;
    return value == null ? "—" : `${value}${unit}`;
  };
  document.getElementById("liveDailyRows").innerHTML = (daily.time || []).map((date, index) =>
    `<tr><td>${date}</td>${dailyFields.map(([key, unit]) => {
      const value = daily[key]?.[index];
      const displayValue = key === "weather_code" || value == null || Number.isInteger(value)
        ? valueAt(key, index, unit)
        : `${Number(value).toFixed(1)}${unit}`;
      return `<td>${displayValue}</td>`;
    }).join("")}</tr>`
  ).join("");

  const hourly = detail?.hourly || {};
  const times = hourly.time || [];
  const startTime = current.time || "";
  const nextHours = times.map((time, index) => ({ time, index }))
    .filter(point => !startTime || point.time >= startTime)
    .slice(0, 12);
  const hourlyValue = (key, index, unit) => {
    const value = hourly[key]?.[index];
    return value == null ? "—" : `${value}${unit}`;
  };
  document.getElementById("liveHourlyRows").innerHTML = nextHours.length
    ? nextHours.map(({ time, index }) => `<tr>
        <td>${time.slice(5, 16).replace("T", " ")}</td>
        <td>${hourlyValue("temperature_2m", index, "°")}</td>
        <td>${hourlyValue("precipitation", index, " mm")}</td>
        <td>${hourlyValue("precipitation_probability", index, "%")}</td>
        <td>${hourlyValue("wind_speed_10m", index, " km/h")}</td>
      </tr>`).join("")
    : "<tr><td colspan=\"5\">Hourly details unavailable.</td></tr>";
}

// ---------------------------------------------------------------
// EXPLAINABLE AI
// ---------------------------------------------------------------
async function loadExplainView() {
  const regions = await ensureRegionsList();
  const sel = document.getElementById("explainRegionSelect");
  const daySel = document.getElementById("explainDaySelect");
  if (!sel.dataset.filled) {
    sel.innerHTML = regions.map(r => `<option ${r === state.explainRegion ? "selected" : ""}>${r}</option>`).join("");
    daySel.innerHTML = Array.from({length:10}, (_,i) => i+1).map(d => `<option value="${d}" ${d === state.explainDay ? "selected" : ""}>Day ${d}</option>`).join("");
    sel.dataset.filled = "1";
    sel.addEventListener("change", renderExplain);
    daySel.addEventListener("change", renderExplain);
    document.querySelectorAll(".pct-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pct-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderExplain();
      });
    });
  } else {
    sel.value = state.explainRegion;
    daySel.value = state.explainDay;
  }
  renderExplain();
}

async function renderExplain() {
  const region = document.getElementById("explainRegionSelect").value;
  const day = parseInt(document.getElementById("explainDaySelect").value);
  if (!state.liveSnapshot) await loadLiveSnapshot();
  const pctBtn = document.querySelector(".pct-btn.active");
  const percentile = pctBtn ? pctBtn.dataset.pct : "90";
  const data = findLiveCell(region, day) || await apiGet(`/api/explain/${encodeURIComponent(region)}/${day}?percentile=${percentile}`);
  if (!data) return;

  document.getElementById("explainBust").textContent = data.bust_probability + "%";
  document.getElementById("explainMeta").textContent = `DAY ${data.day} • ${region.toUpperCase()}`;
  document.getElementById("explainConfidence").innerHTML = `Confidence: ${data.confidence}% · <span class="risk-badge ${data.risk_level}">${data.risk_level} BUST RISK</span> · Error-prone: ${data.error_prone_area ? "YES" : "NO"}`;

  const narrative = document.getElementById("explainNarrative");
  if (narrative) narrative.textContent = data.explanation || "";
  const thr = document.getElementById("explainThresholdNote");
  if (thr) {
    const threshold = data.statistical_threshold_precip ?? data[`bust_threshold_precip_p${percentile}`] ?? "—";
    thr.textContent = `Statistical precipitation error threshold at the ${percentile}th percentile for this region and lead time: ${threshold} mm (mean historical error ${data.mean_historical_precip_error ?? "—"} mm).`;
  }

  const shap = data.shap_contributors || [];
  const strength = contributor => Math.abs(contributor.shap ?? contributor.importance ?? 0);
  const maxAbs = Math.max(...shap.map(strength), 0.0001);
  const bars = shap.map((c) => {
    const mag = strength(c);
    const pct = Math.max(8, Math.round((mag / maxAbs) * 100));
    const down = (c.shap || 0) < 0;
    const value = c.shap == null ? `${((c.importance || 0) * 100).toFixed(1)}%` : `${c.shap > 0 ? "+" : ""}${c.shap}`;
    return `<div class="fbar-row">
      <div class="fbar-name">${c.label}</div>
      <div class="fbar-track"><div class="fbar-fill ${down ? "down" : ""}" style="width:${pct}%"></div></div>
      <div class="fbar-pct">${value}</div>
    </div>`;
  }).join("");
  document.getElementById("featureBars").innerHTML = bars +
    `<p class="muted" style="margin-top:10px;">${data.disclaimer || state.liveSnapshot?.disclaimer || "Prototype feature attribution; not causal proof."} Method: ${data.explanation_method || "Random Forest Feature Attribution"}</p>`;
}

// ---------------------------------------------------------------
// HISTORICAL EVENT REPLAY
// ---------------------------------------------------------------
let eventsCache = null;
async function loadReplayView() {
  if (!eventsCache) {
    const data = await apiGet("/api/events", { events: [] });
    eventsCache = data.events;
    const tabs = document.getElementById("eventTabs");
    tabs.innerHTML = eventsCache.map((e, i) =>
      `<button class="event-tab ${i === 0 ? "active" : ""}" data-id="${e.id}">${e.name}</button>`
    ).join("");
    tabs.querySelectorAll(".event-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        tabs.querySelectorAll(".event-tab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderEvent(btn.dataset.id);
      });
    });
  }
  if (eventsCache.length) renderEvent(eventsCache[0].id);
}

function renderEvent(id) {
  const e = eventsCache.find(ev => ev.id === id);
  if (!e) return;
  const verifiedBust = Boolean(e.forecast_bust);
  const resultClass = verifiedBust ? "HIGH-RISK" : "LOW-RISK";
  document.getElementById("eventFlow").innerHTML = `
    <div style="margin-bottom:10px;"><span class="event-tag">${e.note}</span></div>
    <div class="event-flow-grid">
      <div class="flow-step"><div class="fs-label">REGION</div><div class="fs-value" style="font-size:14px;">${e.region}</div></div>
      <div class="flow-step"><div class="fs-label">LEAD TIME</div><div class="fs-value">Day ${e.lead_time}</div></div>
      <div class="flow-step"><div class="fs-label">FORECAST</div><div class="fs-value">${e.forecast_value}${e.unit}</div></div>
      <div class="flow-step"><div class="fs-label">REFERENCE / OBSERVED</div><div class="fs-value">${e.reference_value}${e.unit}</div></div>
      <div class="flow-step"><div class="fs-label">FORECAST ERROR</div><div class="fs-value">${e.forecast_error}${e.unit}</div></div>
    </div>
    <div class="panel" style="text-align:center;">
      <div class="fs-label" style="margin-bottom:6px;">CLIME RELIABILITY ESTIMATE</div>
      <div style="font-size:40px; font-weight:800; color:var(--primary-dark);">${e.bust_probability}%</div>
      <div class="rd-metric"><span>Verified bust</span><span class="val">${verifiedBust ? "YES" : "NO"}</span></div>
      <div class="rd-metric"><span>P${e.threshold_percentile} bust threshold</span><span class="val">${e.bust_threshold} ${e.unit}</span></div>
      <div class="rd-metric"><span>Model confidence</span><span class="val">${e.confidence}%</span></div>
      <div class="rd-metric"><span>Predicted risk</span><span class="risk-badge ${e.risk_level}">${e.risk_level}</span></div>
      <p class="muted" style="margin-top:10px;">${e.why || ""}</p>
    </div>
    <div class="event-result ${resultClass}">${e.result}</div>
  `;
}

// ---------------------------------------------------------------
// MODEL PERFORMANCE
// ---------------------------------------------------------------
let perfLoaded = false;
async function loadPerformanceView() {
  if (perfLoaded) return;
  const data = await apiGet("/api/model-performance");
  if (!data) return;
  perfLoaded = true;

  document.getElementById("metricsGrid").innerHTML = `
    <div class="kpi-card"><div class="kpi-label">Model</div><div class="kpi-value" style="font-size:18px;">${data.model_type || "Random Forest"}</div></div>
    <div class="kpi-card"><div class="kpi-label">Accuracy</div><div class="kpi-value">${(data.accuracy*100).toFixed(1)}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Precision</div><div class="kpi-value">${(data.precision*100).toFixed(1)}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Recall</div><div class="kpi-value">${(data.recall*100).toFixed(1)}%</div></div>
    <div class="kpi-card"><div class="kpi-label">ROC-AUC</div><div class="kpi-value">${data.roc_auc.toFixed(3)}</div></div>
  `;

  drawLineChart("rocChart", data.roc_curve.fpr.map(v => v.toFixed(2)),
    [{ label: "ROC Curve (AUC " + data.roc_auc.toFixed(3) + ")", data: data.roc_curve.tpr, color: "#1B5FE0" }], "False Positive Rate", "True Positive Rate");

  drawLineChart("prChart", data.pr_curve.recall.map(v => v.toFixed(2)),
    [{ label: "PR Curve (AP " + data.pr_auc.toFixed(3) + ")", data: data.pr_curve.precision, color: "#1C9A6C" }], "Recall", "Precision");

  drawLineChart("calChart", data.calibration_curve.mean_predicted.map(v => v.toFixed(2)),
    [{ label: "Observed frequency", data: data.calibration_curve.fraction_positive, color: "#C97A0F" }], "Mean predicted probability", "Fraction of positives");

  const cm = data.confusion_matrix; // [[TN,FP],[FN,TP]]
  document.getElementById("confusionMatrix").innerHTML = `
    <div class="cm-cell tn"><div class="cm-num">${cm[0][0]}</div><div class="cm-label">True Negative</div></div>
    <div class="cm-cell fp"><div class="cm-num">${cm[0][1]}</div><div class="cm-label">False Positive</div></div>
    <div class="cm-cell fn"><div class="cm-num">${cm[1][0]}</div><div class="cm-label">False Negative</div></div>
    <div class="cm-cell tp"><div class="cm-num">${cm[1][1]}</div><div class="cm-label">True Positive</div></div>
  `;

  const lead = data.performance_by_lead_time;
  drawLineChart("leadPerfChart", lead.map(l => "Day " + l.day),
    [
      { label: "Accuracy", data: lead.map(l => l.accuracy), color: "#1B5FE0" },
      { label: "Avg Bust Probability", data: lead.map(l => l.avg_bust_probability), color: "#D6425C" },
    ]);
}

// ---------------------------------------------------------------
// CHART HELPER
// ---------------------------------------------------------------
function drawLineChart(canvasId, labels, series, xLabel, yLabel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  if (state.charts[canvasId]) {
    const previous = state.charts[canvasId];
    if (previous && typeof previous.destroy === "function") previous.destroy();
  }
  state.charts[canvasId] = { type: "custom" };

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 420;
  const height = rect.height || 220;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pad = { top: 20, right: 18, bottom: 42, left: 42 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const flat = series.flatMap(s => s.data.filter(v => v !== null && v !== undefined && Number.isFinite(v)));
  const minValue = flat.length ? Math.min(...flat) : 0;
  const maxValue = flat.length ? Math.max(...flat) : 100;
  const range = maxValue - minValue || 1;

  const yScale = (v) => pad.top + ((maxValue - v) / range) * plotHeight;
  const xScale = (idx) => pad.left + (plotWidth * idx) / Math.max(labels.length - 1, 1);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#EEF3FA";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (plotHeight * i) / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#5B7192";
  ctx.font = "11px Inter, sans-serif";
  if (xLabel) {
    ctx.fillText(xLabel, width / 2 - 20, height - 8);
  }
  if (yLabel) {
    ctx.save();
    ctx.translate(12, height / 2 + 20);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  labels.forEach((label, idx) => {
    const x = xScale(idx);
    ctx.fillStyle = "#5B7192";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(String(label), x - 14, height - 16);
  });

  series.forEach((entry) => {
    if (!entry.data || !entry.data.length) return;
    ctx.beginPath();
    entry.data.forEach((value, idx) => {
      const safeValue = value === null || value === undefined || !Number.isFinite(value) ? minValue : value;
      const x = xScale(idx);
      const y = yScale(safeValue);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = entry.color || "#1B5FE0";
    ctx.lineWidth = 2;
    ctx.stroke();

    const fillColor = (entry.color || "#1B5FE0") + "22";
    if (entry.data.some(v => v !== null && v !== undefined && Number.isFinite(v))) {
      ctx.beginPath();
      entry.data.forEach((value, idx) => {
        const safeValue = value === null || value === undefined || !Number.isFinite(value) ? minValue : value;
        const x = xScale(idx);
        const y = yScale(safeValue);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.lineTo(xScale(Math.max(labels.length - 1, 0)), height - pad.bottom);
      ctx.lineTo(xScale(0), height - pad.bottom);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    entry.data.forEach((value, idx) => {
      if (value === null || value === undefined || !Number.isFinite(value)) return;
      const x = xScale(idx);
      const y = yScale(value);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = entry.color || "#1B5FE0";
      ctx.fill();
    });
  });
}

// ---------------------------------------------------------------
// INTERACTIVE FORECAST SIMULATOR
// ---------------------------------------------------------------
function bindRangeLabels() {
  const ranges = [
    { id: "simPrecipitation", suffix: " mm", type: "number" },
    { id: "simTemperature", suffix: "°C" },
    { id: "simHumidity", suffix: "%" },
    { id: "simWindSpeed", suffix: " km/h" },
    { id: "simLeadTime", prefix: "Day " },
  ];

  ranges.forEach(({ id, suffix, prefix }) => {
    const input = document.getElementById(id);
    if (!input) return;
    const output = document.getElementById(`${id}Val`);
    const update = () => {
      const value = Number(input.value);
      if (!output) return;
      output.textContent = prefix ? `${prefix}${value}` : `${value}${suffix}`;
    };
    input.addEventListener("input", update);
    update();
  });
}

async function runPrediction() {
  const payload = {
    precipitation: Number(document.getElementById("simPrecipitation").value),
    temperature: Number(document.getElementById("simTemperature").value),
    pressure: 1012.6,
    humidity: Number(document.getElementById("simHumidity").value),
    wind_speed: Number(document.getElementById("simWindSpeed").value),
    wind_direction: 180,
    lead_time: Number(document.getElementById("simLeadTime").value),
    latitude: 22.5,
    longitude: 78.5,
    spatial_gradient: 5.5,
    historical_error: 8.5,
    pressure_variation: 4.2,
    precipitation_variability: 14,
  };

  const result = await apiPost("/api/predict", payload, null);
  const outcome = document.getElementById("predictOutcome");

  if (!result) {
    outcome.innerHTML = "<strong>Model unavailable.</strong> The dashboard is currently using demo data only.";
    return;
  }

  const confidenceText = result.confidence ? `${result.confidence}% confidence` : "confidence unavailable";
  outcome.innerHTML = `
    <strong>${result.risk_level} bust risk</strong> • ${result.bust_probability}% bust probability • ${confidenceText}
    ${result.error_prone_area ? " • Error-prone: YES" : ""}<br>
    ${result.explanation || ""}<br>
    <span class="muted">${result.disclaimer}</span>
  `;
}

async function loadApiView() {
  const health = await apiGet("/api/health", { status: "offline" });
  const integration = await apiGet("/api/integration", { endpoints: [] });
  document.getElementById("apiHealth").textContent = JSON.stringify(health, null, 2);
  document.getElementById("apiTable").innerHTML = (integration.endpoints || []).map(e =>
    `<div class="api-row"><span class="api-method">${e.method}</span><code>${e.path}</code><span>${e.use}</span></div>`
  ).join("");
}

// ---------------------------------------------------------------
// AI DECISION CENTER BUTTON
// ---------------------------------------------------------------
function initDecisionCenter() {
  document.getElementById("decisionBtn")?.addEventListener("click", () => {
    switchView("decision");
  });
}

// ---------------------------------------------------------------
// INIT
// ---------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initDashboardFilters();
  bindRangeLabels();

  document
    .getElementById("runPredictionBtn")
    ?.addEventListener("click", runPrediction);

  document
    .getElementById("refreshLive")
    ?.addEventListener("click", () => loadDashboard(true));

  // AI Decision Center
  initDecisionCenter();

  initMapViewToggle();
  buildDaySelector();
  loadDashboard();

  window.setInterval(() => loadDashboard(true), 15 * 60 * 1000);
});