let DATA = null;
let currentTab = "known";
let sortKey = null;
let sortAsc = false;

const COLS = {
  known: ["player_name", "season", "position", "overall_pick", "predicted_prob", "games_played"],
  prospects: ["player_name", "season", "position", "overall_pick", "predicted_prob", "games_played"],
  diamonds: ["player_name", "season", "position", "overall_pick", "predicted_prob", "games_played"],
  busts: ["player_name", "season", "position", "overall_pick", "predicted_prob", "games_played"],
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

  tbody.innerHTML = rows
    .map((r) => {
      const pickText = r.overall_pick === -1 ? "Undrafted" : `#${r.overall_pick}`;
      const prob = r.predicted_prob;
      let hitBadge = "";
      if ("actual_hit" in r) {
        hitBadge =
          r.actual_hit === 1
            ? '<span class="hit-badge hit-yes">Rotation+</span>'
            : '<span class="hit-badge hit-no">Bench/out</span>';
      }
      return `<tr>
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
        <td>${r.games_played} ${hitBadge}</td>
      </tr>`;
    })
    .join("");
}
