const consoleLog = document.getElementById("consoleLog");
const analyzeForm = document.getElementById("analyzeForm");
const runDemo = document.getElementById("runDemo");
const startAnalyze = document.getElementById("startAnalyze");
const viewPipeline = document.getElementById("viewPipeline");
const themeToggle = document.getElementById("themeToggle");
const connectApi = document.getElementById("connectApi");
const downloadReport = document.getElementById("downloadReport");
const errorBanner = document.getElementById("errorBanner");
const page1 = document.getElementById("page1");
const page2 = document.getElementById("page2");
const backToInput = document.getElementById("backToInput");

const phaseChart = document.getElementById("phaseChart");
const comparisonChart = document.getElementById("comparisonChart");
const summaryText = document.getElementById("summaryText");
const patternList = document.getElementById("patternList");
const moveList = document.getElementById("moveList");
const chessBoard = document.getElementById("chessBoard");
const moveMeta = document.getElementById("moveMeta");
const prevMove = document.getElementById("prevMove");
const nextMove = document.getElementById("nextMove");

const defaultApiBase = (() => {
  const { protocol, hostname, port, origin } = window.location;
  const isLocalFrontend =
    (hostname === "localhost" || hostname === "127.0.0.1") && port === "8080";
  if (isLocalFrontend) {
    return `${protocol}//${hostname}:8000`;
  }
  return origin;
})();
const API_BASE = localStorage.getItem("apiBase") || defaultApiBase;
let currentMoves = [];
let currentIndex = 0;

const pieceMap = {
  p: "\u265F",
  r: "\u265C",
  n: "\u265E",
  b: "\u265D",
  q: "\u265B",
  k: "\u265A",
  P: "\u2659",
  R: "\u2656",
  N: "\u2658",
  B: "\u2657",
  Q: "\u2655",
  K: "\u2654",
};

const logLine = (text) => {
  const line = document.createElement("div");
  line.className = "line";
  line.textContent = text;
  consoleLog.prepend(line);
};

const showError = (text) => {
  errorBanner.textContent = text;
  errorBanner.style.display = "block";
};

const clearError = () => {
  errorBanner.textContent = "";
  errorBanner.style.display = "none";
};

const formatLichessAccessError = (rawMessage) => {
  const message = rawMessage || "Failed to fetch games from Lichess.";
  const lower = message.toLowerCase();
  const isLichessAccessIssue =
    lower.includes("lichess api error") ||
    lower.includes("privacy") ||
    lower.includes("visibility") ||
    lower.includes("download") ||
    lower.includes("no games found");

  if (!isLichessAccessIssue) return message;

  return [
    "Could not fetch games from Lichess.",
    "Step 1: Log in to Lichess and open Settings.",
    "Step 2: Go to Privacy.",
    "Step 3: Enable game visibility for your account.",
    "Step 4: Enable game download/export access.",
    "Step 5: Confirm your username is correct and profile is public.",
    "Step 6: Retry analysis.",
    "Lichess settings: https://lichess.org/account/preferences/privacy",
  ].join(" ");
};

const drawBarChart = (data) => {
  if (!phaseChart) return;
  const ctx = phaseChart.getContext("2d");
  ctx.clearRect(0, 0, phaseChart.width, phaseChart.height);

  const padding = 40;
  const barWidth = 90;
  const maxVal = 1;
  const colors = ["#30f2cf", "#6d7bff", "#f6c453"];

  data.forEach((item, idx) => {
    const x = padding + idx * (barWidth + 40);
    const barHeight = (phaseChart.height - padding * 2) * (item.accuracy / maxVal);
    const y = phaseChart.height - padding - barHeight;
    ctx.fillStyle = colors[idx % colors.length];
    ctx.fillRect(x, y, barWidth, barHeight);
    ctx.fillStyle = "#f5f5f7";
    ctx.font = "14px 'IBM Plex Mono'";
    ctx.fillText(item.phase, x, phaseChart.height - padding + 20);
    ctx.fillText(`${Math.round(item.accuracy * 100)}%`, x, y - 10);
  });
};

const drawComparisonChart = (moves) => {
  if (!comparisonChart) return;
  const ctx = comparisonChart.getContext("2d");
  ctx.clearRect(0, 0, comparisonChart.width, comparisonChart.height);

  const padding = 30;
  const width = comparisonChart.width - padding * 2;
  const height = comparisonChart.height - padding * 2;

  const slice = moves.slice(0, 140);
  const cplValues = slice.map((m) => m.cpl);
  const diffValues = slice.map((m) => m.difficulty * 200);
  const maxVal = Math.max(...cplValues, ...diffValues, 200);

  const toY = (val) =>
    comparisonChart.height - padding - (val / maxVal) * height;

  ctx.strokeStyle = "#30f2cf";
  ctx.beginPath();
  slice.forEach((m, idx) => {
    const x = padding + (idx / (slice.length - 1)) * width;
    const y = toY(m.cpl);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = "#6d7bff";
  ctx.beginPath();
  slice.forEach((m, idx) => {
    const x = padding + (idx / (slice.length - 1)) * width;
    const y = toY(m.difficulty * 200);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#f5f5f7";
  ctx.font = "12px 'IBM Plex Mono'";
  ctx.fillText("CPL", padding, padding - 8);
  ctx.fillStyle = "#6d7bff";
  ctx.fillText("CNN", padding + 50, padding - 8);
};

const renderBoard = (fen) => {
  if (!chessBoard) return;
  chessBoard.innerHTML = "";
  const boardPart = fen.split(" ")[0];
  const rows = boardPart.split("/");

  rows.forEach((row, rowIdx) => {
    let col = 0;
    for (const char of row) {
      if (Number.isInteger(parseInt(char, 10))) {
        const emptyCount = parseInt(char, 10);
        for (let i = 0; i < emptyCount; i++) {
          const square = document.createElement("div");
          square.className = `square ${(rowIdx + col) % 2 === 0 ? "light" : "dark"}`;
          chessBoard.appendChild(square);
          col += 1;
        }
      } else {
        const square = document.createElement("div");
        square.className = `square ${(rowIdx + col) % 2 === 0 ? "light" : "dark"}`;
        square.classList.add(char === char.toUpperCase() ? "piece-white" : "piece-black");
        square.textContent = pieceMap[char] || "";
        chessBoard.appendChild(square);
        col += 1;
      }
    }
  });
};

const setMoveDetails = (index) => {
  if (!currentMoves.length) return;
  currentIndex = Math.max(0, Math.min(index, currentMoves.length - 1));
  const move = currentMoves[currentIndex];
  renderBoard(move.fen_after);
  moveMeta.textContent = `Move ${move.ply} - ${move.san} - CPL ${move.cpl.toFixed(
    1
  )} - Difficulty ${(move.difficulty * 100).toFixed(1)}% - ${move.phase} - ${move.suggestion}`;
};

const renderMoveList = () => {
  moveList.innerHTML = "";
  currentMoves.slice(0, 240).forEach((move, idx) => {
    const li = document.createElement("li");
    li.className = "move-item";
    li.innerHTML = `<strong>${move.ply}. ${move.san}</strong>
      <small>CPL ${move.cpl.toFixed(1)} - Difficulty ${(move.difficulty * 100).toFixed(
      1
    )}% - ${move.phase}</small>
      <small>${move.suggestion}</small>`;
    li.addEventListener("click", () => setMoveDetails(idx));
    moveList.appendChild(li);
  });
};

const setSummary = (payload) => {
  summaryText.textContent = payload.summary;
  patternList.innerHTML = "";
  payload.top_patterns.forEach((pattern) => {
    const li = document.createElement("li");
    li.textContent = pattern;
    patternList.appendChild(li);
  });
  drawBarChart(payload.phase_breakdown);
  currentMoves = payload.moves || [];
  if (currentMoves.length) {
    renderMoveList();
    setMoveDetails(0);
    drawComparisonChart(currentMoves);
  }
};

const runAnalysis = async (formData, pgnFile, allowDemoFallback = false) => {
  logLine("Connecting to analysis engine...");
  clearError();
  try {
    let response;
    if (pgnFile) {
      const upload = new FormData();
      upload.append("file", pgnFile);
      response = await fetch(`${API_BASE}/api/analyze_pgn`, {
        method: "POST",
        body: upload,
      });
    } else {
      response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
    }

    if (!response.ok) {
      let errText = "API request failed.";
      const raw = await response.text();
      if (raw) {
        try {
          const errJson = JSON.parse(raw);
          errText = errJson.detail || raw;
        } catch (e) {
          errText = raw;
        }
      }
      throw new Error(errText);
    }
    const payload = await response.json();
    logLine(`Analysis complete for ${payload.username}.`);
    setSummary(payload);
    if (payload.report_url) {
      downloadReport.dataset.report = `${API_BASE}${payload.report_url}`;
    }
    if (page1 && page2) {
      page1.style.display = "none";
      page2.style.display = "block";
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "API request failed.";
    const friendlyMessage = formatLichessAccessError(message);
    showError(friendlyMessage);
    if (!allowDemoFallback) {
      logLine(`Analysis failed: ${friendlyMessage}`);
      return;
    }
    logLine("API unavailable. Running demo mode.");
    const payload = {
      summary:
        "Demo data loaded. Plug in the trained model + Stockfish to enable live analysis.",
      phase_breakdown: [
        { phase: "Opening", accuracy: 0.82 },
        { phase: "Middlegame", accuracy: 0.73 },
        { phase: "Endgame", accuracy: 0.64 },
      ],
      top_patterns: [
        "Inaccurate pawn breaks under time pressure",
        "Loose pieces on open files",
        "Overlooking tactical rebounds",
      ],
      moves: [],
    };
    setSummary(payload);
    if (page1 && page2) {
      page1.style.display = "none";
      page2.style.display = "block";
    }
  }
};

analyzeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const data = new FormData(analyzeForm);
  const payload = {
    username: data.get("username"),
    games: Number(data.get("games")),
    mode: data.get("mode"),
    include_pgn: Boolean(data.get("include_pgn")),
  };
  const pgnFile = data.get("pgn_file");
  runAnalysis(payload, pgnFile && pgnFile.size ? pgnFile : null, false);
});

runDemo.addEventListener("click", () => {
  runAnalysis({ username: "demo_player", games: 12, mode: "blitz" }, null, true);
});

startAnalyze.addEventListener("click", () => {
  analyzeForm.scrollIntoView({ behavior: "smooth" });
  analyzeForm.querySelector("input[name='username']").focus();
});

viewPipeline.addEventListener("click", () => {
  analyzeForm.scrollIntoView({ behavior: "smooth" });
});

themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("nebula");
});

if (connectApi) {
  connectApi.addEventListener("click", () => {
    const url = prompt("Paste API base URL", API_BASE);
    if (url) {
      localStorage.setItem("apiBase", url);
      logLine(`API set to ${url}`);
    }
  });
}

downloadReport.addEventListener("click", () => {
  const path = downloadReport.dataset.report;
  if (path) {
    window.open(path, "_blank");
  } else {
    logLine("PDF will be available after analysis finishes.");
  }
});

prevMove.addEventListener("click", () => setMoveDetails(currentIndex - 1));
nextMove.addEventListener("click", () => setMoveDetails(currentIndex + 1));

if (backToInput) {
  backToInput.addEventListener("click", () => {
    if (page1 && page2) {
      page2.style.display = "none";
      page1.style.display = "block";
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });
}

window.addEventListener("load", () => {
  logLine(`Frontend ready. API: ${API_BASE}`);
  drawBarChart([
    { phase: "Opening", accuracy: 0.78 },
    { phase: "Middlegame", accuracy: 0.71 },
    { phase: "Endgame", accuracy: 0.66 },
  ]);
  initHeroScene();
});

const initHeroScene = () => {
  const canvas = document.getElementById("hero3d");
  if (!canvas || !window.THREE) return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(window.innerWidth, window.innerHeight);

  const geometry = new THREE.BufferGeometry();
  const count = 900;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 60;
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({ color: 0x5ad1ff, size: 0.08 });
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  camera.position.z = 20;

  const animate = () => {
    points.rotation.y += 0.0008;
    points.rotation.x += 0.0004;
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };
  animate();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
};
