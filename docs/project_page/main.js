const data = window.projectData;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function renderHeroStats() {
  const root = document.querySelector("#hero-stats");
  if (!root) return;
  data.heroStats.forEach((item) => {
    const wrap = el("div");
    const dt = el("dt", null, item.label);
    const dd = el("dd", null, item.value);
    wrap.append(dt, dd);
    root.append(wrap);
  });
}

function renderPipeline() {
  const root = document.querySelector("#pipeline");
  if (!root) return;
  data.pipeline.forEach((item, index) => {
    const card = el("article", "pipeline-card");
    card.append(
      el("span", "pipeline-index", String(index + 1)),
      el("strong", null, item.title),
      el("p", null, item.body),
    );
    root.append(card);
  });
}

function renderTaskSettings() {
  const root = document.querySelector("#task-settings");
  if (!root || !data.taskSettings) return;

  if (data.taskSettings.intro) {
    root.append(el("p", "task-suite-intro", data.taskSettings.intro));
  }

  if (data.taskSettings.groups) {
    const groupGrid = el("div", "task-family-grid");
    data.taskSettings.groups.forEach((item) => {
      const card = el("article", "task-family-card");
      card.append(
        el("span", "label", item.label),
        el("strong", null, item.value),
        el("p", null, item.body),
      );
      groupGrid.append(card);
    });
    root.append(groupGrid);
  }

  const wrap = el("div", "task-table-wrap");
  const table = document.createElement("table");
  table.className = "task-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  data.taskSettings.columns.forEach((column) => headRow.append(el("th", null, column)));
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  data.taskSettings.rows.forEach((item) => {
    const row = document.createElement("tr");
    row.append(
      (() => {
        const cell = el("td", "setting-cell");
        cell.append(el("span", `pill ${item.settingTone}`, item.setting));
        return cell;
      })(),
      el("th", "task-name-cell", item.task),
      el("td", null, item.objects),
      el("td", null, item.dynamics),
      el("td", null, item.affordance),
      el("td", null, item.objective),
    );
    tbody.append(row);
  });
  table.append(tbody);
  wrap.append(table);
  root.append(wrap);

  if (data.taskSettings.references) {
    const referenceGrid = el("div", "task-reference-grid");
    data.taskSettings.references.forEach((item) => {
      const figure = document.createElement("figure");
      figure.className = "task-reference-card";

      const image = document.createElement("img");
      image.src = item.src;
      image.alt = item.title;

      const caption = document.createElement("figcaption");
      caption.append(el("strong", null, item.title), el("span", null, item.caption));
      if (item.href) {
        const link = document.createElement("a");
        link.href = item.href;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = "Reference paper";
        caption.append(link);
      }

      figure.append(image, caption);
      referenceGrid.append(figure);
    });
    root.append(referenceGrid);
  }
}

function renderModules() {
  const root = document.querySelector("#module-stack");
  if (!root || !data.modules) return;

  data.modules.forEach((item) => {
    const card = el("article", "module-card");
    const list = document.createElement("ul");
    item.items.forEach((line) => list.append(el("li", null, line)));
    card.append(
      el("span", "label", item.label),
      el("h3", null, item.title),
      el("p", null, item.body),
      list,
    );
    root.append(card);
  });
}

function renderTimeline() {
  const root = document.querySelector("#timeline");
  if (!root) return;
  data.experiments.forEach((item) => {
    const card = el("article", "timeline-item");
    const version = el("div", "timeline-version", item.version);
    const copy = el("div", "timeline-copy");
    copy.append(el("h3", null, item.title), el("p", null, item.body));
    const meta = el("div", "meta-row");
    meta.append(el("span", `pill ${item.status}`, item.result));
    card.append(version, copy, meta);
    root.append(card);
  });
}

function renderVideos() {
  const root = document.querySelector("#video-grid");
  if (!root) return;
  data.videos.forEach((item) => {
    const card = el("article", "video-card");
    const video = document.createElement("video");
    video.controls = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.poster = item.poster;

    const source = document.createElement("source");
    source.src = item.src;
    source.type = "video/mp4";
    video.append(source);

    const body = el("div", "video-card-body");
    const meta = el("div", "meta-row");
    meta.append(el("span", `pill ${item.tone}`, item.tag), el("span", "pill", item.result));
    body.append(el("h3", null, item.title), el("p", null, item.caption), meta);
    card.append(video, body);
    root.append(card);
  });
}

function renderAffordance() {
  const root = document.querySelector("#affordance-root");
  if (!root || !data.affordance) return;
  const affordance = data.affordance;

  root.append(el("p", "affordance-intro", affordance.intro));

  const metrics = el("div", "affordance-metrics");
  affordance.metrics.forEach((item) => {
    const card = el("article", "affordance-metric");
    card.append(el("span", "label", item.label), el("strong", null, item.value), el("p", null, item.detail));
    metrics.append(card);
  });
  root.append(metrics);

  const rawStats = el("div", "affordance-raw-stats");
  affordance.rawStats.forEach((item) => {
    const row = el("article", "affordance-raw-card");
    row.append(
      el("h3", null, item.label),
      el("p", null, item.vertices),
      el("p", null, item.labeled),
      el("p", null, item.overlap),
    );
    rawStats.append(row);
  });
  root.append(rawStats);

  const visualGrid = el("div", "affordance-visual-grid");
  affordance.visuals.forEach((item) => {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = item.src;
    image.alt = item.title;
    figure.append(image, el("figcaption", null, `${item.title}. ${item.caption}`));
    visualGrid.append(figure);
  });
  root.append(visualGrid);

  const details = el("div", "affordance-detail-grid");

  const method = el("article", "affordance-panel");
  const methodList = document.createElement("ol");
  affordance.method.forEach((item) => methodList.append(el("li", null, item)));
  method.append(el("h3", null, "Labeling pipeline"), methodList);
  details.append(method);

  const usage = el("article", "affordance-panel");
  const usageList = document.createElement("ul");
  affordance.labelUse.forEach((item) => usageList.append(el("li", null, item)));
  usage.append(el("h3", null, "Training use"), usageList);
  details.append(usage);

  const quality = el("article", "affordance-panel");
  quality.append(el("h3", null, "Quality notes"));
  affordance.quality.forEach((item) => {
    const row = el("div", "quality-row");
    row.append(el("strong", null, item.label), el("p", null, item.items));
    quality.append(row);
  });
  details.append(quality);

  const files = el("article", "affordance-panel file-panel");
  const fileList = document.createElement("ul");
  affordance.files.forEach((item) => {
    const li = document.createElement("li");
    li.append(el("code", null, item));
    fileList.append(li);
  });
  files.append(el("h3", null, "Where the artifacts live"), fileList);
  details.append(files);

  root.append(details);
}

function renderDiagnosis() {
  const root = document.querySelector("#diagnosis-list");
  if (!root) return;
  data.diagnosis.forEach((item) => {
    const card = el("article", "diagnosis-item");
    card.append(el("h3", null, item.title), el("p", null, item.body));
    root.append(card);
  });
}

renderHeroStats();
renderTaskSettings();
renderPipeline();
renderModules();
renderTimeline();
renderVideos();
renderAffordance();
renderDiagnosis();
