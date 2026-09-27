const byId = (id) => document.getElementById(id);
const statusNames = {
  completed: "완료", pending: "대기", running: "진행 중", failed: "실패",
  excluded: "평가 제외", unavailable: "기록 없음", not_performed: "미평가",
};
const dimensionNames = {
  believability: "Believability", relationship: "Relationship", knowledge: "Knowledge",
  secret: "Secret", social_rules: "Social rules",
  financial_and_material_benefits: "Financial & material", goal: "Goal",
  overall_score: "Overall score",
};
const fieldNames = {
  first_name: "이름", last_name: "성", age: "나이", occupation: "직업",
  gender: "성별", gender_pronoun: "대명사", public_info: "공개 정보",
  personality_and_values: "성격과 가치관", secret: "비밀", scenario: "상황",
  agent_goals: "목표 (Agent 1 / Agent 2)", relationship: "관계 유형",
  background_story: "관계 배경", big_five: "Big Five", moral_values: "도덕적 가치",
  schwartz_personal_values: "개인적 가치", decision_making_style: "의사결정 방식",
};
let request = null;
let episodes = [];
let runs = [];
let selectedExperiment = "";
let selectedRun = "";
let selectedEpisode = "";
let player = null;

function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null && text !== undefined) node.textContent = String(text);
  return node;
}

function statusBadge(status) {
  const node = el("span", "badge", statusNames[status] || status);
  if (["completed", "failed", "pending", "running", "excluded"].includes(status)) {
    node.classList.add(status);
  }
  return node;
}

function message(text, className = "notice") {
  return el("p", className, text);
}

function warnings(items) {
  const box = el("div", "notices");
  [...new Set(items.filter(Boolean))].forEach((text) => box.append(message(text)));
  return box;
}

function shortModel(model) {
  return (model || "").split("@")[0].replace(/^custom\/structured-/, "")
    .replace(/^talktopia-(?:agent|evaluator)-/, "").replace(/-gpu\d+$/, "");
}

function timeLabel(time) {
  if (!Number.isFinite(time)) return "—";
  return Math.floor(time / 60) + ":" + String(Math.floor(time % 60)).padStart(2, "0");
}

function apiBase() {
  return "/api/runs/" + encodeURIComponent(selectedRun);
}

function episodeBase() {
  return apiBase() + "/episodes/" + encodeURIComponent(selectedEpisode);
}

async function getJSON(url, signal) {
  const response = await fetch(url, { signal, cache: "no-store" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "결과를 불러올 수 없습니다.");
  return data;
}

function beginRequest() {
  request?.abort();
  request = new AbortController();
  if (player) {
    player.pause();
    player.removeAttribute("src");
    player.load();
    player = null;
  }
  byId("detail").replaceChildren(el("div", "empty", "에피소드를 불러오는 중입니다."));
  byId("detail").setAttribute("aria-busy", "true");
  return request.signal;
}

function showError(error) {
  if (error.name === "AbortError") return;
  byId("detail").replaceChildren(message(error.message, "empty error"));
  byId("detail").removeAttribute("aria-busy");
}

function renderList() {
  const list = byId("episode-list");
  const fragment = document.createDocumentFragment();
  for (const episode of episodes) {
    const button = el("button", "episode-item");
    button.type = "button";
    button.dataset.episode = episode.id;
    button.setAttribute("aria-current", String(episode.id === selectedEpisode));
    const top = el("span", "episode-item-top");
    top.append(el("span", "episode-number", episode.id.replace("episode_", "#")), statusBadge(episode.status));
    button.append(top, el("span", "episode-scenario", episode.codename || "시나리오 정보 없음"));
    button.addEventListener("click", () => selectEpisode(episode.id));
    fragment.append(button);
  }
  list.replaceChildren(fragment);
  byId("episode-count").textContent = episodes.length;
}

async function loadRun(signal) {
  const data = await getJSON(apiBase() + "/episodes", signal);
  episodes = data.episodes;
  byId("catalog-warning").replaceChildren(warnings(data.warnings));
  if (!episodes.some((episode) => episode.id === selectedEpisode)) {
    selectedEpisode = episodes.find((episode) => episode.status === "completed")?.id
      || episodes[0]?.id || "";
  }
  renderList();
  if (!selectedEpisode) {
    byId("detail").replaceChildren(el("div", "empty", "이 실행에 에피소드 기록이 없습니다."));
    byId("detail").removeAttribute("aria-busy");
    return;
  }
  await loadEpisode(signal);
}

function renderSelectors() {
  const tags = [...new Set(runs.map((run) => run.experiment_tag))];
  if (!tags.includes(selectedExperiment)) selectedExperiment = tags[0] || "";
  const experiments = byId("experiment-select");
  experiments.replaceChildren(...tags.map((tag) => {
    const option = el("option", "", tag);
    option.value = tag;
    return option;
  }));
  experiments.value = selectedExperiment;
  experiments.title = selectedExperiment;
  experiments.disabled = !tags.length;

  const pairs = runs.filter((run) => run.experiment_tag === selectedExperiment);
  if (!pairs.some((run) => run.id === selectedRun)) {
    selectedRun = pairs[0]?.id || "";
    selectedEpisode = "";
  }
  const select = byId("run-select");
  select.replaceChildren(...pairs.map((run) => {
    const option = el("option", "", run.pair_label);
    option.value = run.id;
    return option;
  }));
  select.value = selectedRun;
  select.title = pairs.find((run) => run.id === selectedRun)?.pair_label || "";
  select.disabled = !pairs.length;
}

async function refresh() {
  const signal = beginRequest();
  try {
    runs = await getJSON("/api/runs", signal);
    selectedExperiment = runs.find((run) => run.id === selectedRun)?.experiment_tag || selectedExperiment;
    renderSelectors();
    episodes = [];
    renderList();
    byId("catalog-warning").replaceChildren();
    if (!selectedRun) {
      byId("detail").replaceChildren(el("div", "empty", "설정 파일의 runs에 시뮬레이션 경로를 추가해 주세요."));
      byId("detail").removeAttribute("aria-busy");
      return;
    }
    await loadRun(signal);
  } catch (error) { showError(error); }
}

async function selectEpisode(id) {
  selectedEpisode = id;
  const signal = beginRequest();
  renderList();
  try { await loadEpisode(signal); } catch (error) { showError(error); }
}

function profileFields(profile) {
  if (!profile) return message("실행 당시 프로필을 확인할 수 없습니다.");
  const list = el("dl", "profile-fields");
  for (const [key, value] of Object.entries(profile)) {
    const text = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value ?? "—");
    list.append(el("dt", "", fieldNames[key] || key), el("dd", "", text || "—"));
  }
  return list;
}

function disclosure(title, content, className = "") {
  const details = el("details", "disclosure " + className);
  const summary = el("summary", "", title);
  details.append(summary, content);
  return details;
}

function renderProfiles(data) {
  const section = el("section", "profiles");
  const agents = el("div", "profile-pair");
  data.agent_names.forEach((name, index) => {
    const card = el("section", "profile-agent agent-" + index);
    card.append(el("h3", "", "Agent " + (index + 1) + " · " + name));
    card.append(profileFields(data.profiles.agents[index]));
    if (data.perspectives[index]) {
      card.append(disclosure("에피소드 시작 시 전달된 맥락", el("pre", "context", data.perspectives[index].text)));
    }
    agents.append(card);
  });
  section.append(
    disclosure("Agent profile", agents),
    disclosure("Environment profile", profileFields(data.profiles.environment)),
    disclosure(
      "Relationship profile" + (data.profiles.relationship_label ? " · " + data.profiles.relationship_label : ""),
      profileFields(data.profiles.relationship),
    ),
  );
  if (data.profiles.warnings.length) {
    section.append(disclosure("프로필 확인 사항 (" + data.profiles.warnings.length + ")", warnings(data.profiles.warnings)));
  }
  return section;
}

function renderReport(base, reportId, title, enabled, signal) {
  if (!enabled) return el("div", "missing-report", title + " · 파일 없음");
  const content = el("div", "report-content");
  const details = disclosure(title, content, "report");
  let loaded = false;
  details.addEventListener("toggle", async () => {
    if (!details.open || loaded) return;
    loaded = true;
    content.replaceChildren(message("보고서를 불러오는 중입니다."));
    try {
      const data = await getJSON(base + "/reports/" + encodeURIComponent(reportId), signal);
      const toggle = el("button", "quiet-button source-toggle", "Markdown 원문 보기");
      toggle.type = "button";
      toggle.setAttribute("aria-pressed", "false");
      const rendered = el("div", "markdown");
      // This is server-rendered Markdown with raw HTML and image rules disabled.
      rendered.innerHTML = data.html;
      for (const link of rendered.querySelectorAll("a")) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      const original = el("pre", "markdown-source", data.markdown);
      original.hidden = true;
      toggle.addEventListener("click", () => {
        original.hidden = !original.hidden;
        rendered.hidden = !original.hidden;
        toggle.textContent = original.hidden ? "Markdown 원문 보기" : "문서 보기";
        toggle.setAttribute("aria-pressed", String(!original.hidden));
      });
      content.replaceChildren(toggle, rendered, original);
    } catch (error) {
      if (error.name !== "AbortError") {
        content.replaceChildren(message(error.message));
        loaded = false;
      }
    }
  });
  return details;
}

function renderConversation(data, base, signal) {
  const column = el("section", "conversation-column");
  const heading = el("div", "section-heading");
  heading.append(el("h2", "", "대화"), el("span", "muted", "수신 텍스트 · ASR"));
  column.append(heading);
  const playback = el("div", "playback");
  const caption = el("div", "playback-caption");
  caption.append(el("span", "", "전체 대화"), el("span", "muted", timeLabel(data.playback.duration)));
  playback.append(caption);
  if (data.playback.audio_available) {
    player = el("audio", "audio-player");
    player.controls = true;
    player.preload = "metadata";
    player.src = base + "/audio";
    player.setAttribute("aria-label", "두 에이전트의 전체 대화");
    playback.append(player);
  } else {
    playback.append(message("이 에피소드에는 재생 가능한 전체 음성이 없습니다."));
  }
  column.append(playback, warnings(data.playback.warnings));
  const transcript = el("div", "transcript");
  const entries = [];
  for (const [index, entry] of data.playback.entries.entries()) {
    const agentIndex = data.agent_names.indexOf(entry.speaker);
    const timed = Number.isFinite(entry.start);
    const row = el(timed ? "button" : "article", "utterance agent-" + agentIndex);
    row.dataset.index = index;
    if (timed) {
      row.type = "button";
      row.title = timeLabel(entry.start) + " 위치로 이동";
      row.addEventListener("click", () => {
        if (player) { player.currentTime = entry.start; highlight(); }
      });
    }
    const meta = el("div", "utterance-meta");
    meta.append(
      el("span", "speaker", entry.speaker || "화자 정보 없음"),
      el("span", "agent-label", agentIndex >= 0 ? "Agent " + (agentIndex + 1) : ""),
      el("span", "utterance-time", timeLabel(entry.start)),
    );
    if (entry.action !== "speak") meta.append(el("span", "action-label", entry.action));
    row.append(meta, el("p", "utterance-text", entry.text));
    if (!entry.committed) row.append(el("span", "muted", "미확정 발화"));
    transcript.append(row);
    entries.push([row, entry]);
  }
  function highlight() {
    const current = player?.currentTime ?? -1;
    for (const [node, entry] of entries) {
      const active = Number.isFinite(entry.start) && Number.isFinite(entry.end)
        && entry.start <= current && current < entry.end;
      node.classList.toggle("is-playing", active);
      if (active) node.setAttribute("aria-current", "true");
      else node.removeAttribute("aria-current");
    }
  }
  if (player) {
    for (const event of ["timeupdate", "seeking", "seeked", "loadedmetadata", "ended"]) {
      player.addEventListener(event, highlight);
    }
  }
  if (!entries.length) transcript.append(message("표시할 대화 기록이 없습니다.", "empty"));
  column.append(
    transcript,
    renderReport(base, "simulation", "시뮬레이션 readable 문서", data.report_available, signal),
  );
  return column;
}

function renderEvaluations(data, base, signal) {
  const section = el("section", "evaluation-column");
  const heading = el("div", "section-heading");
  heading.append(el("h2", "", "평가 비교"), el("span", "count", data.evaluations.length));
  section.append(heading);
  if (!data.evaluations.length) {
    section.append(message("설정의 evaluation_dirs에 평가 경로를 추가하면 여기에 표시됩니다."));
    return section;
  }
  const table = el("table", "scores");
  const head = el("thead");
  const modelRow = el("tr");
  const dimensionHead = el("th", "", "평가 차원");
  dimensionHead.rowSpan = 2;
  modelRow.append(dimensionHead);
  const agentsRow = el("tr");
  for (const evaluation of data.evaluations) {
    const model = el("th", "", shortModel(evaluation.model));
    model.colSpan = 2;
    model.title = evaluation.model + "\n" + evaluation.run_name;
    modelRow.append(model);
    data.agent_names.forEach((name, index) => {
      const cell = el("th", "agent-" + index, "Agent " + (index + 1));
      cell.title = name;
      agentsRow.append(cell);
    });
  }
  head.append(modelRow, agentsRow);
  const body = el("tbody");
  for (const dimension of data.dimensions) {
    const row = el("tr", dimension === "overall_score" ? "overall-row" : "");
    const label = el("th", "", dimensionNames[dimension] || dimension);
    label.scope = "row";
    row.append(label);
    for (const evaluation of data.evaluations) {
      for (let index = 0; index < 2; index++) {
        const score = evaluation.scores?.[index]?.[dimension];
        const cell = el("td", "", Number.isFinite(score) ? Number(score.toFixed(2)) : "—");
        if (!Number.isFinite(score)) cell.title = evaluation.message || statusNames[evaluation.status] || evaluation.status;
        row.append(cell);
      }
    }
    body.append(row);
  }
  table.append(head, body);
  const scroll = el("div", "table-scroll");
  scroll.tabIndex = 0;
  scroll.setAttribute("aria-label", "평가기별 점수 비교표");
  scroll.append(table);
  section.append(scroll, el("p", "score-note", "Overall score는 7개 원점수의 단순 평균입니다. 미평가·제외·원본 불일치는 —로 표시합니다."));
  for (const evaluation of data.evaluations) {
    const card = el("section", "evaluation-report");
    const meta = el("div", "evaluation-meta");
    meta.append(el("h3", "", shortModel(evaluation.model)), statusBadge(evaluation.status));
    card.append(meta, el("p", "run-name", evaluation.run_name));
    if (evaluation.message) card.append(message(evaluation.message));
    card.append(
      warnings(evaluation.warnings),
      renderReport(base, evaluation.id, "평가 readable 문서", evaluation.report_available, signal),
    );
    section.append(card);
  }
  return section;
}

async function loadEpisode(signal) {
  const base = episodeBase();
  const data = await getJSON(base, signal);
  const header = el("header", "episode-header");
  const run = runs.find((item) => item.id === data.run_id);
  if (run) header.append(el("p", "run-context", run.experiment_tag + " / " + run.pair_label));
  const titleRow = el("div", "title-row");
  titleRow.append(el("h1", "", data.id.replace("episode_", "Episode ")), statusBadge(data.status));
  header.append(el("p", "eyebrow", data.playback.mode === "round-robin" ? "ROUND ROBIN" : "FULL DUPLEX"));
  header.append(titleRow, el("p", "scenario-title", data.codename || "시나리오 정보 없음"));
  const participants = el("div", "participants");
  data.agent_names.forEach((name, index) => {
    const agent = el("div", "participant agent-" + index);
    agent.append(
      el("span", "agent-number", String(index + 1)),
      el("span", "participant-name", name),
      el("span", "participant-model", shortModel(data.models[index]) || "모델 정보 없음"),
    );
    participants.append(agent);
  });
  header.append(participants);
  const columns = el("div", "result-columns");
  columns.append(renderConversation(data, base, signal), renderEvaluations(data, base, signal));
  byId("detail").replaceChildren(
    header, warnings([...data.warnings, data.error]), renderProfiles(data), columns,
  );
  byId("detail").removeAttribute("aria-busy");
}

async function changeRun() {
  selectedEpisode = "";
  episodes = [];
  renderList();
  byId("catalog-warning").replaceChildren();
  const signal = beginRequest();
  try { await loadRun(signal); } catch (error) { showError(error); }
}

byId("experiment-select").addEventListener("change", () => {
  selectedExperiment = byId("experiment-select").value;
  selectedRun = "";
  renderSelectors();
  changeRun();
});
byId("run-select").addEventListener("change", () => {
  selectedRun = byId("run-select").value;
  renderSelectors();
  changeRun();
});
byId("refresh").addEventListener("click", refresh);
refresh();
