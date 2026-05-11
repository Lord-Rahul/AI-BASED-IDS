const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const predictButton = document.getElementById('predictButton');
const healthStatus = document.getElementById('healthStatus');
const healthMeta = document.getElementById('healthMeta');
const rowCount = document.getElementById('rowCount');
const attackCount = document.getElementById('attackCount');
const benignCount = document.getElementById('benignCount');
const attackScore = document.getElementById('attackScore');
const predictionBody = document.getElementById('predictionBody');
const attackBody = document.getElementById('attackBody');
const resultState = document.getElementById('resultState');
const attackSearchInput = document.getElementById('attackSearchInput');
const riskFilterSelect = document.getElementById('riskFilterSelect');
const attackSummary = document.getElementById('attackSummary');
const scoreChart = document.getElementById('scoreChart');
const metricsChart = document.getElementById('metricsChart');
const downloadPredictionsButton = document.getElementById('downloadPredictionsButton');
const downloadReportButton = document.getElementById('downloadReportButton');
const attackDetailEmpty = document.getElementById('attackDetailEmpty');
const attackDetailList = document.getElementById('attackDetailList');

let latestPredictions = [];
let latestAttackScores = [];
let latestAttackDetails = [];
let filteredAttackDetails = [];
let latestAttackThreshold = 0.2;
let sortKey = 'row_number';
let sortDirection = 'asc';

async function checkHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) {
      throw new Error('Health check failed');
    }

    const data = await response.json();
    healthStatus.textContent = data.status.toUpperCase();
    latestAttackThreshold = Number(data.attack_threshold || 0.2);
    healthMeta.textContent = `Threshold ${latestAttackThreshold} | ${data.classes.join(', ')}`;
  } catch (error) {
    healthStatus.textContent = 'OFFLINE';
    healthMeta.textContent = 'Start ml/server.py to enable predictions';
  }
}

function renderPredictions(predictions, attackScores) {
  predictionBody.innerHTML = '';
  const preview = predictions.slice(0, 20);

  preview.forEach((label, index) => {
    const row = document.createElement('tr');
    const indexCell = document.createElement('td');
    const labelCell = document.createElement('td');

    indexCell.textContent = String(index + 1);
    labelCell.textContent = `${label} (${Number(attackScores[index] || 0).toFixed(3)})`;
    labelCell.className = label === 'ATTACK' ? 'label-danger' : 'label-safe';

    row.appendChild(indexCell);
    row.appendChild(labelCell);
    predictionBody.appendChild(row);
  });
}

function normalizeFilterText(value) {
  return String(value || '').trim().toLowerCase();
}

function matchesAttackFilter(detail) {
  const searchText = normalizeFilterText(attackSearchInput.value);
  const selectedRisk = riskFilterSelect.value;
  const detailText = [
    detail.row_number,
    detail.predicted_label,
    detail.attack_score,
    detail.risk,
    ...Object.entries(detail.feature_snapshot).flatMap(([name, value]) => [name, value]),
  ]
    .join(' ')
    .toLowerCase();

  const matchesSearch = !searchText || detailText.includes(searchText);
  const matchesRisk = selectedRisk === 'all' || detail.risk === selectedRisk;
  return matchesSearch && matchesRisk;
}

function sortAttackTable(key) {
  if (sortKey === key) {
    sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
  } else {
    sortKey = key;
    sortDirection = 'asc';
  }
  updateSortIndicators();
  renderAttackDetails(latestAttackDetails);
}

function updateSortIndicators() {
  const headers = document.querySelectorAll('th[onclick]');
  headers.forEach((th) => {
    const indicator = th.querySelector('.sort-indicator');
    if (!indicator) return;
    
    const key = th.getAttribute('onclick').match(/'([^']+)'/)[1];
    if (key === sortKey) {
      indicator.style.opacity = '1';
      indicator.style.color = sortDirection === 'asc' ? 'var(--accent)' : 'var(--danger)';
      indicator.innerHTML = sortDirection === 'asc' ? '▲' : '▼';
    } else {
      indicator.style.opacity = '0.3';
      indicator.innerHTML = '·';
    }
  });
}

function drawMetricsChart(benignCnt, attackCnt) {
  const context = metricsChart.getContext('2d');
  const width = metricsChart.width;
  const height = metricsChart.height;
  const padding = 48;
  const chartHeight = height - padding * 2;
  const chartWidth = width - padding * 2;
  const barWidth = chartWidth / 4;
  const total = benignCnt + attackCnt;

  context.clearRect(0, 0, width, height);
  context.fillStyle = '#07101d';
  context.fillRect(0, 0, width, height);

  context.strokeStyle = 'rgba(159, 177, 204, 0.2)';
  context.beginPath();
  context.moveTo(padding, padding);
  context.lineTo(padding, height - padding);
  context.lineTo(width - padding, height - padding);
  context.stroke();

  const benignHeight = (benignCnt / total) * chartHeight || 0;
  const attackHeight = (attackCnt / total) * chartHeight || 0;

  context.fillStyle = 'rgba(105, 240, 174, 0.75)';
  context.fillRect(padding + 20, height - padding - benignHeight, barWidth, benignHeight);
  context.fillStyle = '#69f0ae';
  context.font = 'bold 14px sans-serif';
  context.fillText('BENIGN', padding + 20, padding - 12);
  context.font = '12px sans-serif';
  context.fillText(benignCnt.toString(), padding + 20 + barWidth / 3, height - padding + 20);

  context.fillStyle = 'rgba(255, 111, 145, 0.75)';
  context.fillRect(padding + barWidth + 60, height - padding - attackHeight, barWidth, attackHeight);
  context.fillStyle = '#ff6f91';
  context.font = 'bold 14px sans-serif';
  context.fillText('ATTACK', padding + barWidth + 60, padding - 12);
  context.font = '12px sans-serif';
  context.fillText(attackCnt.toString(), padding + barWidth + 60 + barWidth / 3, height - padding + 20);
}

function renderAttackDetails(details) {
  latestAttackDetails = details;
  filteredAttackDetails = details.filter(matchesAttackFilter);

  let sortedDetails = [...filteredAttackDetails];
  sortedDetails.sort((a, b) => {
    let aVal = a[sortKey];
    let bVal = b[sortKey];
    if (sortKey === 'attack_score') {
      aVal = Number(aVal);
      bVal = Number(bVal);
    } else if (sortKey === 'row_number') {
      aVal = Number(aVal);
      bVal = Number(bVal);
    }
    const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    return sortDirection === 'asc' ? cmp : -cmp;
  });

  attackBody.innerHTML = '';

  if (!details.length) {
    attackBody.innerHTML = '<tr><td colspan="4">No attacks detected in this file.</td></tr>';
    attackDetailEmpty.textContent = 'No attack rows to inspect yet.';
    attackDetailList.classList.add('hidden');
    downloadReportButton.disabled = true;
    return;
  }

  if (!sortedDetails.length) {
    attackBody.innerHTML = '<tr><td colspan="4">No attacks match the current filters.</td></tr>';
    attackSummary.textContent = `Showing 0 of ${details.length} attack rows.`;
    attackDetailEmpty.textContent = 'No rows match the current filters.';
    attackDetailList.classList.add('hidden');
    downloadReportButton.disabled = false;
    return;
  }

  sortedDetails.forEach((detail) => {
    const row = document.createElement('tr');
    row.className = 'clickable-row';
    row.dataset.rowNumber = String(detail.row_number);

    const featureText = Object.entries(detail.feature_snapshot)
      .map(([name, value]) => `${name}: ${value}`)
      .join(' | ');

    row.innerHTML = `
      <td>${detail.row_number}</td>
      <td>${detail.attack_score.toFixed(6)}</td>
      <td class="label-danger">${detail.risk}</td>
      <td title="${featureText}">${featureText}</td>
    `;

    row.addEventListener('click', () => showAttackDetail(detail));
    attackBody.appendChild(row);
  });

  attackSummary.textContent = `Showing ${sortedDetails.length} of ${details.length} attack rows.`;
  downloadReportButton.disabled = false;
  showAttackDetail(sortedDetails[0]);
}

function showAttackDetail(detail) {
  attackDetailEmpty.classList.add('hidden');
  attackDetailList.classList.remove('hidden');
  attackDetailList.innerHTML = '';

  const topDetails = [
    ['Row number', detail.row_number],
    ['Predicted label', detail.predicted_label],
    ['Attack score', Number(detail.attack_score).toFixed(6)],
    ['Risk level', detail.risk.toUpperCase()],
  ];

  const detailsDiv = document.createElement('div');
  topDetails.forEach(([label, value]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = label;
    dd.textContent = String(value);
    attackDetailList.appendChild(dt);
    attackDetailList.appendChild(dd);
  });

  const featureHeader = document.createElement('dt');
  featureHeader.style.marginTop = '12px';
  featureHeader.style.fontWeight = 'bold';
  featureHeader.textContent = 'Features';
  attackDetailList.appendChild(featureHeader);

  const features = detail.feature_snapshot;
  const featureNames = Object.keys(features).sort();
  
  featureNames.forEach((name) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = name;
    dd.textContent = String(features[name]);
    dt.style.fontSize = '0.9em';
    dd.style.fontSize = '0.9em';
    attackDetailList.appendChild(dt);
    attackDetailList.appendChild(dd);
  });
}

function downloadCsv(filename, rows) {
  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function drawScoreChart(scores) {
  const context = scoreChart.getContext('2d');
  const width = scoreChart.width;
  const height = scoreChart.height;
  const padding = 36;
  const bins = 10;
  const histogram = Array.from({ length: bins }, () => 0);

  scores.forEach((score) => {
    const normalized = Math.min(Math.max(score, 0), 0.999999);
    const index = Math.min(bins - 1, Math.floor(normalized * bins));
    histogram[index] += 1;
  });

  context.clearRect(0, 0, width, height);
  context.fillStyle = '#07101d';
  context.fillRect(0, 0, width, height);

  const maxCount = Math.max(...histogram, 1);
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;
  const barWidth = chartWidth / bins - 10;

  context.strokeStyle = 'rgba(159, 177, 204, 0.2)';
  context.beginPath();
  context.moveTo(padding, padding);
  context.lineTo(padding, height - padding);
  context.lineTo(width - padding, height - padding);
  context.stroke();

  histogram.forEach((count, index) => {
    const barHeight = (count / maxCount) * chartHeight;
    const x = padding + index * (barWidth + 10) + 5;
    const y = height - padding - barHeight;

    context.fillStyle = 'rgba(83, 167, 255, 0.75)';
    context.fillRect(x, y, barWidth, barHeight);

    context.fillStyle = '#9fb1cc';
    context.font = '12px sans-serif';
    context.fillText((index / bins).toFixed(1), x + 2, height - 14);
  });

  const thresholdX = padding + chartWidth * latestAttackThreshold;
  context.strokeStyle = '#ff6f91';
  context.setLineDash([6, 6]);
  context.beginPath();
  context.moveTo(thresholdX, padding / 2);
  context.lineTo(thresholdX, height - padding);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = '#ff6f91';
  context.fillText(`threshold ${latestAttackThreshold.toFixed(2)}`, Math.min(thresholdX + 6, width - 120), padding / 2 + 12);
}

function downloadAttackReport() {
  if (!latestAttackDetails.length) {
    return;
  }

  const featureNames = Array.from(
    new Set(latestAttackDetails.flatMap((detail) => Object.keys(detail.feature_snapshot)))
  );

  const header = ['row_number', 'predicted_label', 'attack_score', 'risk', ...featureNames];
  const escapeCsv = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const rows = [header.map(escapeCsv).join(',')];

  latestAttackDetails.forEach((detail) => {
    const row = [
      detail.row_number,
      detail.predicted_label,
      detail.attack_score,
      detail.risk,
      ...featureNames.map((name) => detail.feature_snapshot[name] ?? ''),
    ];
    rows.push(row.map(escapeCsv).join(','));
  });

  downloadCsv('attack_report.csv', rows);
}

function downloadPredictionsCsv() {
  if (!latestPredictions.length) {
    return;
  }

  const rows = ['row_number,predicted_label,attack_score'];
  latestPredictions.forEach((label, index) => {
    rows.push([index + 1, label, Number(latestAttackScores[index] || 0).toFixed(6)].join(','));
  });

  downloadCsv('predictions_summary.csv', rows);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    resultState.textContent = 'Choose a CSV file first.';
    return;
  }

  const payload = new FormData();
  payload.append('file', file);

  predictButton.disabled = true;
  resultState.textContent = 'Running prediction...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: payload,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction request failed');
    }

    latestPredictions = data.predictions || [];
    latestAttackScores = data.attack_scores || [];

    rowCount.textContent = String(data.rows);
    attackCount.textContent = String(data.attack_rows);
    benignCount.textContent = String(data.benign_rows);
    attackScore.textContent = Number(data.attack_average_score || 0).toFixed(3);
    resultState.textContent = `Processed ${data.rows} rows.`;
    renderPredictions(latestPredictions, latestAttackScores);
    renderAttackDetails(data.attack_details || []);
    drawScoreChart(latestAttackScores);
    drawMetricsChart(data.benign_rows, data.attack_rows);
    downloadPredictionsButton.disabled = false;
    attackSearchInput.value = '';
    riskFilterSelect.value = 'all';
    sortKey = 'row_number';
    sortDirection = 'asc';
    updateSortIndicators();
  } catch (error) {
    resultState.textContent = error.message;
  } finally {
    predictButton.disabled = false;
  }
});

downloadReportButton.addEventListener('click', downloadAttackReport);
downloadPredictionsButton.addEventListener('click', downloadPredictionsCsv);
attackSearchInput.addEventListener('input', () => renderAttackDetails(latestAttackDetails));
riskFilterSelect.addEventListener('change', () => renderAttackDetails(latestAttackDetails));

checkHealth();