let DATA = null;
let currentTab = "known";
let sortKey = null;
let sortAsc = false;
let currentRows = [];

const COLS = {
  known: ["player_name", "season", "position", "overall_pick", "predicted_prob", "combine_grade", "games_played"],
  prospects: ["player_name", "season", "position", "overall_pick", "predicted_prob", "combine_grade", "games_played"],
  diamonds: ["player_name", "season", "position", "overall_pick", "predicted_prob", "combine_grade", "games_played"],
  busts: ["player_name", "season", "position", "overall_pick", "predicted_prob", "combine_grade", "games_played"],
};

fetch("assets/site_data.json")
  .then((r) => r.json())
  .then((data) => {
    DATA = data;
    render();
  });

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentTab = btn.dataset.tab;
    sortKey = null;
    document.getElementById("search").value = "";
    render();
  });
});

document.getElementById("search").addEventListener("input", render);

document.querySelectorAll("#player-table thead th").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (sortKey === key) {
      sortAsc = !sortAsc;
    } else {
      sortKey = key;
      sortAsc = true;
    }
    render();
  });
});

function gradeClass(grade) {
  if (grade >= 80) return "grade-a";
  if (grade >= 60) return "grade-b";
  if (grade >= 40) return "grade-c";
  return "grade-d";
}

function render() {
  if (!DATA) return;
  const tbody = document.getElementById("player-tbody");
  let rows = [...DATA[currentTab]];

  const q = document.getElementById("search").value.trim().toLowerCase();
  if (q) rows = rows.filter((r) => r.player_name.toLowerCase().includes(q));

  if (sortKey) {
    rows.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "string") {
        return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sortAsc ? av - bv : bv - av;
    });
  }

  rows = rows.slice(0, 120);
  currentRows = rows;

  tbody.innerHTML = rows
    .map((r, i) => {
      const pickText = r.overall_pick === -1 ? "Undrafted" : `#${r.overall_pick}`;
      const prob = r.predicted_prob;
      const grade = r.combine_grade;
      let hitBadge = "";
      if ("actual_hit" in r) {
        hitBadge =
          r.actual_hit === 1
            ? '<span class="hit-badge hit-yes">Rotation+</span>'
            : '<span class="hit-badge hit-no">Bench/out</span>';
      }
      return `<tr class="player-row" data-idx="${i}" tabindex="0">
        <td><strong>${r.player_name}</strong></td>
        <td>${r.season}</td>
        <td>${r.position || "—"}</td>
        <td>${pickText}</td>
        <td>
          <div class="prob-bar-wrap">
            <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${prob}%"></div></div>
            <span>${prob}%</span>
          </div>
        </td>
        <td><span class="grade-pill ${gradeClass(grade)}">${grade}</span></td>
        <td>${r.games_played} ${hitBadge}</td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll(".player-row").forEach((tr) => {
    const open = () => openReportCard(currentRows[Number(tr.dataset.idx)]);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open();
      }
    });
  });
}

function openReportCard(r) {
  if (!r) return;
  const overlay = document.getElementById("report-card-overlay");
  const pickText = r.overall_pick === -1 ? "Undrafted" : `Pick #${r.overall_pick}`;
  document.getElementById("rc-name").textContent = r.player_name;
  document.getElementById("rc-meta").textContent =
    `${r.season} combine \u00b7 ${r.position || "\u2014"} \u00b7 ${pickText}`;
  document.getElementById("rc-grade-num").textContent = r.combine_grade;
  const circle = document.getElementById("rc-grade-circle");
  circle.className = `rc-grade-circle ${gradeClass(r.combine_grade)}`;
  document.getElementById("rc-model-prob").textContent =
    `Model-estimated probability of becoming a rotation player: ${r.predicted_prob}%`;

  const confidence = document.getElementById("rc-confidence");
  const missing = r.missing_count || 0;
  if (missing > 0) {
    confidence.textContent =
      `Low-confidence grade — ${missing} of 10 combine measurements missing ` +
      `(filled in at the median), so this leans mostly on draft position.`;
    confidence.classList.add("show");
  } else {
    confidence.textContent = "";
    confidence.classList.remove("show");
  }

  const list = document.getElementById("rc-factors");
  list.innerHTML = (r.factors || [])
    .map((f) => {
      const sign = f.effect === "positive" ? "+" : "\u2212";
      const cls = f.effect === "positive" ? "factor-positive" : "factor-negative";
      return `<li class="${cls}"><span class="factor-sign">${sign}</span>
        <span class="factor-text"><strong>${f.feature}:</strong> ${f.detail}</span></li>`;
    })
    .join("");

  overlay.classList.remove("hidden");
}

function closeReportCard() {
  document.getElementById("report-card-overlay").classList.add("hidden");
}

document.getElementById("report-card-close").addEventListener("click", closeReportCard);
document.getElementById("report-card-overlay").addEventListener("click", (e) => {
  if (e.target.id === "report-card-overlay") closeReportCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeReportCard();
});
