// static/js/app.js
let currentSection = 'dashboard';
let chartInstances = {};
let currentDatasetInfo = null;
let predictionResults = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupSidebarToggle();
    loadSection('dashboard');
    setupEventListeners();
});

// Sidebar Toggle
function setupSidebarToggle() {
    const toggleBtn = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    
    if (toggleBtn && sidebar && overlay) {
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('active');
            overlay.classList.toggle('active');
        });
        
        overlay.addEventListener('click', function() {
            sidebar.classList.remove('active');
            overlay.classList.remove('active');
        });
    }
}

// Setup event listeners
function setupEventListeners() {
    document.querySelectorAll('.nav-link[data-section]').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.dataset.section;
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            loadSection(section);
            
            // Close sidebar on mobile
            if (window.innerWidth <= 767) {
                document.getElementById('sidebar').classList.remove('active');
                document.getElementById('sidebarOverlay').classList.remove('active');
            }
        });
    });

    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            loadSection(currentSection);
            showToast('Refreshed', 'Content refreshed successfully', 'success');
        });
    }
}

// Load section
function loadSection(section) {
    currentSection = section;
    const contentArea = document.getElementById('content-area');
    const pageTitle = document.getElementById('page-title');

    const sectionTitles = {
        'dashboard': 'Dashboard',
        'dataset': 'Dataset Management',
        'preprocessing': 'Data Preprocessing',
        'training': 'Model Training',
        'evaluation': 'Model Evaluation',
        'features': 'Feature Analysis',
        'predict': 'Single Student Prediction',
        'batch': 'Batch Prediction'
    };

    if (pageTitle) pageTitle.textContent = sectionTitles[section] || 'Dashboard';

    switch(section) {
        case 'dashboard': renderDashboard(contentArea); break;
        case 'dataset': renderDataset(contentArea); break;
        case 'preprocessing': renderPreprocessing(contentArea); break;
        case 'training': renderTraining(contentArea); break;
        case 'evaluation': renderEvaluation(contentArea); break;
        case 'features': renderFeatures(contentArea); break;
        case 'predict': renderPrediction(contentArea); break;
        case 'batch': renderBatchPrediction(contentArea); break;
        default: contentArea.innerHTML = '<div class="alert alert-info">Section under construction</div>';
    }
}

// ==================== DASHBOARD ====================
function renderDashboard(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="row g-3" id="dashboardStats">
                <div class="col-md-3 col-6">
                    <div class="card card-stat">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <div class="stat-label">Total Students</div>
                                <div class="stat-number" id="totalStudents">-</div>
                            </div>
                            <i class="bi bi-people stat-icon"></i>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="card card-stat">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <div class="stat-label">Models Trained</div>
                                <div class="stat-number" id="modelsTrained">-</div>
                            </div>
                            <i class="bi bi-cpu stat-icon"></i>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="card card-stat">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <div class="stat-label">Best Model</div>
                                <div class="stat-number" id="bestModel" style="font-size: 1rem;">-</div>
                            </div>
                            <i class="bi bi-trophy stat-icon"></i>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="card card-stat">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <div class="stat-label">At-Risk Students</div>
                                <div class="stat-number" id="atRiskStudents">-</div>
                            </div>
                            <i class="bi bi-exclamation-triangle stat-icon"></i>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row g-3 mt-2">
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6>Performance Distribution</h6>
                        <canvas id="performanceChart" height="200"></canvas>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6>Risk Level Distribution</h6>
                        <canvas id="riskChart" height="200"></canvas>
                    </div>
                </div>
            </div>

            <div class="row g-3 mt-2">
                <div class="col-12">
                    <div class="card p-3">
                        <h6>Model Comparison</h6>
                        <canvas id="modelComparisonChart" height="250"></canvas>
                    </div>
                </div>
            </div>

            <div class="row g-3 mt-2">
                <div class="col-12 text-center">
                    <button class="btn btn-primary" onclick="loadDashboardData()">
                        <i class="bi bi-arrow-repeat"></i> Refresh
                    </button>
                    <button class="btn btn-success" onclick="generateDataset()">
                        <i class="bi bi-file-earmark-plus"></i> Generate Dataset
                    </button>
                </div>
            </div>
        </div>
    `;

    setTimeout(loadDashboardData, 200);
}

function loadDashboardData() {
    // Fetch dataset summary
    fetch('/api/dataset/summary')
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const stats = data.data;
                const totalEl = document.getElementById('totalStudents');
                if (totalEl) totalEl.textContent = stats.total_rows || '-';
                if (stats.target_distribution) {
                    updatePerformanceChart(stats.target_distribution);
                }
            }
        })
        .catch(e => console.error('Error loading dataset summary:', e));

    // Fetch model results
    fetch('/api/models/results')
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const results = data.data;
                const models = Object.keys(results.comparison_table || {});
                const modelsEl = document.getElementById('modelsTrained');
                const bestEl = document.getElementById('bestModel');
                if (modelsEl) modelsEl.textContent = models.length || '-';
                if (bestEl) bestEl.textContent = results.best_model || '-';
                if (results.comparison_table && Object.keys(results.comparison_table).length > 0) {
                    updateModelComparisonChart(results.comparison_table);
                }
            }
        })
        .catch(e => console.error('Error loading model results:', e));

    // Estimate at-risk students
    fetch('/api/features/importance')
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const atRiskEl = document.getElementById('atRiskStudents');
                if (atRiskEl) atRiskEl.textContent = '?';
            }
        })
        .catch(e => console.error('Error:', e));
}

function updatePerformanceChart(distribution) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    if (chartInstances.performance) chartInstances.performance.destroy();

    const labels = Object.keys(distribution);
    const values = Object.values(distribution);
    const colors = labels.map(l => l === 'PASS' ? '#10b981' : '#ef4444');

    chartInstances.performance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { position: 'bottom' } }
        }
    });

    const total = values.reduce((a, b) => a + b, 0);
    const passCount = distribution['PASS'] || 0;
    const failCount = distribution['FAIL'] || 0;
    updateRiskChart({ 
        'SAFE': Math.round(passCount * 0.85), 
        'AT_RISK': Math.round(failCount * 0.6), 
        'FAIL': Math.round(failCount * 0.4) 
    });
}

function updateRiskChart(distribution) {
    const ctx = document.getElementById('riskChart');
    if (!ctx) return;
    if (chartInstances.risk) chartInstances.risk.destroy();

    const labels = Object.keys(distribution);
    const values = Object.values(distribution);
    const colors = {
        'SAFE': '#10b981', 'LOW': '#10b981',
        'MEDIUM': '#f59e0b', 'HIGH': '#f97316',
        'CRITICAL': '#ef4444', 'AT_RISK': '#f97316',
        'FAIL': '#ef4444'
    };

    chartInstances.risk = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Students',
                data: values,
                backgroundColor: labels.map(l => colors[l] || '#6366f1'),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

function updateModelComparisonChart(comparisonTable) {
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx) return;
    if (chartInstances.modelComparison) chartInstances.modelComparison.destroy();

    const models = Object.keys(comparisonTable);
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444'];

    const datasets = metrics.map((metric, index) => ({
        label: metric.toUpperCase(),
        data: models.map(m => comparisonTable[m][metric] || 0),
        backgroundColor: colors[index],
        borderRadius: 4
    }));

    chartInstances.modelComparison = new Chart(ctx, {
        type: 'bar',
        data: { labels: models, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: true, max: 1 } }
        }
    });
}

function generateDataset() {
    showLoading(true);
    fetch('/api/dataset/generate', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                showToast('Success', 'Generated ' + data.data.rows + ' records', 'success');
                loadDashboardData();
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => {
            showLoading(false);
            showToast('Error', 'Failed to generate dataset', 'error');
            console.error(e);
        });
}

// ==================== DATASET ====================
function renderDataset(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-upload"></i> Upload Dataset</h5></div>
                <div class="card-body">
                    <div class="upload-area" id="dropArea">
                        <i class="bi bi-cloud-upload upload-icon"></i>
                        <h5>Drag & drop CSV file here</h5>
                        <p class="text-muted">or click to browse</p>
                        <input type="file" id="fileInput" accept=".csv" style="display:none;">
                        <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                            <i class="bi bi-folder-open"></i> Browse
                        </button>
                    </div>
                    <div class="mt-3">
                        <div class="progress d-none" id="uploadProgress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%">Uploading...</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h5><i class="bi bi-table"></i> Dataset Info</h5></div>
                <div class="card-body">
                    <div id="datasetInfo"><p class="text-muted">No dataset loaded.</p></div>
                    <div id="datasetPreview" style="display:none;">
                        <h6 class="mt-3">Preview</h6>
                        <div class="table-container">
                            <table class="table table-striped table-hover mb-0" id="previewTable">
                                <thead></thead><tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    setupUploadHandlers();
    loadDatasetSummary();
}

function setupUploadHandlers() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    if (!dropArea || !fileInput) return;

    const newDropArea = dropArea.cloneNode(true);
    dropArea.parentNode.replaceChild(newDropArea, dropArea);
    const newFileInput = fileInput.cloneNode(true);
    fileInput.parentNode.replaceChild(newFileInput, fileInput);

    newDropArea.addEventListener('dragover', e => { e.preventDefault(); e.currentTarget.classList.add('dragover'); });
    newDropArea.addEventListener('dragleave', e => { e.preventDefault(); e.currentTarget.classList.remove('dragover'); });
    newDropArea.addEventListener('drop', e => {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    newDropArea.addEventListener('click', () => newFileInput.click());
    newFileInput.addEventListener('change', function() {
        if (this.files.length) uploadFile(this.files[0]);
    });
}

function uploadFile(file) {
    const progress = document.getElementById('uploadProgress');
    if (progress) progress.classList.remove('d-none');
    
    const formData = new FormData();
    formData.append('file', file);

    showLoading(true);
    fetch('/api/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (progress) progress.classList.add('d-none');
            if (data.success) {
                showToast('Success', data.message, 'success');
                loadDatasetSummary();
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => {
            showLoading(false);
            if (progress) progress.classList.add('d-none');
            showToast('Error', 'Upload failed', 'error');
            console.error(e);
        });
}

function loadDatasetSummary() {
    fetch('/api/dataset/summary')
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                currentDatasetInfo = data.data;
                updateDatasetInfo(data.data);
                updateDatasetPreview(data.data.preview);
            }
        })
        .catch(e => console.error('Error:', e));
}

function updateDatasetInfo(info) {
    const container = document.getElementById('datasetInfo');
    if (!container) return;
    container.innerHTML = `
        <div class="row g-2">
            <div class="col-md-3 col-6"><strong>Rows:</strong> ${info.total_rows || 0}</div>
            <div class="col-md-3 col-6"><strong>Columns:</strong> ${info.total_columns || 0}</div>
            <div class="col-md-6 col-12">
                <strong>Missing:</strong>
                <pre class="bg-light p-2 mt-1" style="max-height:80px;overflow-y:auto;font-size:0.8rem;">${JSON.stringify(info.missing_values || {}, null, 2)}</pre>
            </div>
        </div>
        <div class="mt-2">
            <strong>Target Distribution:</strong>
            <pre class="bg-light p-2" style="font-size:0.85rem;">${JSON.stringify(info.target_distribution || {}, null, 2)}</pre>
        </div>
    `;
}

function updateDatasetPreview(preview) {
    const container = document.getElementById('datasetPreview');
    if (!preview || !preview.length) { if(container) container.style.display = 'none'; return; }
    if (container) container.style.display = 'block';
    
    const table = document.getElementById('previewTable');
    if (!table) return;
    const headers = Object.keys(preview[0]);
    table.querySelector('thead').innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
    table.querySelector('tbody').innerHTML = preview.map(row => 
        `<tr>${headers.map(h => `<td>${row[h] !== undefined ? row[h] : '-'}</td>`).join('')}</tr>`
    ).join('');
}

// ==================== PREPROCESSING ====================
function renderPreprocessing(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-gear"></i> Data Preprocessing</h5></div>
                <div class="card-body">
                    <div class="row g-2">
                        <div class="col-md-4">
                            <button class="btn btn-primary w-100" onclick="runPreprocessing()">
                                <i class="bi bi-play"></i> Preprocess
                            </button>
                        </div>
                        <div class="col-md-4">
                            <button class="btn btn-success w-100" onclick="runFeatureEngineering()">
                                <i class="bi bi-plus-circle"></i> Feature Engineering
                            </button>
                        </div>
                        <div class="col-md-4">
                            <button class="btn btn-warning w-100" onclick="runFeatureSelection()">
                                <i class="bi bi-funnel"></i> Feature Selection
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-info-circle"></i> Results</h5></div>
                <div class="card-body" id="preprocessingResults">
                    <p class="text-muted">Run preprocessing to see results.</p>
                </div>
            </div>
        </div>
    `;
}

function runPreprocessing() {
    showLoading(true);
    fetch('/api/preprocess', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                showToast('Success', data.message, 'success');
                const el = document.getElementById('preprocessingResults');
                el.innerHTML = `
                    <div class="row g-2">
                        <div class="col-md-4"><strong>Features:</strong> ${data.data.features_shape[0]} × ${data.data.features_shape[1]}</div>
                        <div class="col-md-8">
                            <strong>Feature Names:</strong>
                            <div class="bg-light p-2 mt-1" style="max-height:120px;overflow-y:auto;font-size:0.8rem;">${data.data.feature_names.join(', ')}</div>
                        </div>
                        <div class="col-12">
                            <strong>Target:</strong>
                            <pre class="bg-light p-2" style="font-size:0.85rem;">${JSON.stringify(data.data.target_distribution, null, 2)}</pre>
                        </div>
                    </div>
                `;
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Preprocessing failed', 'error'); console.error(e); });
}

function runFeatureEngineering() {
    showLoading(true);
    fetch('/api/features/engineer', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                showToast('Success', data.message, 'success');
                const el = document.getElementById('preprocessingResults');
                el.innerHTML += `
                    <div class="mt-3">
                        <strong>Engineered Features:</strong>
                        <div class="bg-light p-2 mt-1" style="font-size:0.85rem;">${data.data.engineered_features.join(', ')}</div>
                        <div class="mt-2"><strong>Total Columns:</strong> ${data.data.total_columns}</div>
                    </div>
                `;
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Feature engineering failed', 'error'); console.error(e); });
}

function runFeatureSelection() {
    showLoading(true);
    fetch('/api/features/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_features: 10 })
    })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                showToast('Success', data.message, 'success');
                const el = document.getElementById('preprocessingResults');
                el.innerHTML += `
                    <div class="mt-3">
                        <strong>Feature Selection</strong>
                        <div class="row g-2 mt-1">
                            <div class="col-md-4"><strong>Original:</strong> ${data.data.original_count}</div>
                            <div class="col-md-4"><strong>Selected:</strong> ${data.data.selected_count}</div>
                            <div class="col-md-12">
                                <strong>Selected Features:</strong>
                                <div class="bg-light p-2 mt-1" style="font-size:0.85rem;">${data.data.selected_features.join(', ')}</div>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Feature selection failed', 'error'); console.error(e); });
}

// ==================== TRAINING ====================
function renderTraining(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-cpu"></i> Model Training</h5></div>
                <div class="card-body">
                    <button class="btn btn-primary btn-lg w-100" onclick="trainModels()">
                        <i class="bi bi-play"></i> Train All Models
                    </button>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-table"></i> Model Comparison</h5></div>
                <div class="card-body" id="trainingResults">
                    <p class="text-muted">Train models to see results.</p>
                </div>
            </div>
        </div>
    `;
}

function trainModels() {
    showLoading(true);
    fetch('/api/train', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                showToast('Success', data.message, 'success');
                displayTrainingResults(data.data);
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Training failed', 'error'); console.error(e); });
}

function displayTrainingResults(data) {
    const container = document.getElementById('trainingResults');
    if (!container) return;
    
    const table = data.comparison_table;
    const models = Object.keys(table);

    let html = `
        <div class="alert alert-success"><strong>Best Model:</strong> ${data.best_model}</div>
        <div class="table-container">
            <table class="table table-striped table-hover mb-0">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>ROC-AUC</th>
                    </tr>
                </thead>
                <tbody>
    `;

    models.forEach(model => {
        const m = table[model];
        const isBest = model === data.best_model;
        html += `
            <tr ${isBest ? 'class="table-success"' : ''}>
                <td><strong>${model}</strong> ${isBest ? '<span class="badge bg-success">Best</span>' : ''}</td>
                <td>${(m.accuracy || 0).toFixed(4)}</td>
                <td>${(m.precision || 0).toFixed(4)}</td>
                <td>${(m.recall || 0).toFixed(4)}</td>
                <td>${(m.f1_score || 0).toFixed(4)}</td>
                <td>${m.roc_auc ? (m.roc_auc || 0).toFixed(4) : 'N/A'}</td>
            </tr>
        `;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

// ==================== EVALUATION ====================
function renderEvaluation(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-bar-chart"></i> Model Evaluation</h5></div>
                <div class="card-body" id="evaluationResults">
                    <p class="text-muted">Load model results to see evaluation metrics.</p>
                    <button class="btn btn-primary" onclick="loadEvaluationResults()">
                        <i class="bi bi-arrow-repeat"></i> Load Results
                    </button>
                </div>
            </div>
        </div>
    `;
}

function loadEvaluationResults() {
    showLoading(true);
    fetch('/api/models/results')
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                displayEvaluationResults(data.data);
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Failed to load results', 'error'); console.error(e); });
}

function displayEvaluationResults(data) {
    const container = document.getElementById('evaluationResults');
    if (!container) return;
    
    const table = data.comparison_table;
    const models = Object.keys(table);

    let html = `
        <div class="alert alert-info"><strong>Best Model:</strong> ${data.best_model}</div>
        <div class="table-container">
            <table class="table table-striped table-hover mb-0">
                <thead>
                    <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>ROC-AUC</th></tr>
                </thead>
                <tbody>
    `;

    models.forEach(model => {
        const m = table[model];
        html += `<tr>
            <td>${model}</td>
            <td>${(m.accuracy || 0).toFixed(4)}</td>
            <td>${(m.precision || 0).toFixed(4)}</td>
            <td>${(m.recall || 0).toFixed(4)}</td>
            <td>${(m.f1_score || 0).toFixed(4)}</td>
            <td>${m.roc_auc ? (m.roc_auc || 0).toFixed(4) : 'N/A'}</td>
        </tr>`;
    });

    html += `</tbody></table></div>`;

    if (data.confusion_matrices) {
        html += `<div class="mt-4"><h6>Confusion Matrices</h6>`;
        Object.keys(data.confusion_matrices).forEach(model => {
            const cm = data.confusion_matrices[model];
            html += `
                <div class="mt-2">
                    <strong>${model}</strong>
                    <pre class="bg-light p-2" style="font-size:0.85rem;">${cm.map(row => row.join('  ')).join('\n')}</pre>
                </div>
            `;
        });
        html += `</div>`;
    }

    container.innerHTML = html;
}

// ==================== FEATURE ANALYSIS ====================
function renderFeatures(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-list-ul"></i> Feature Importance</h5></div>
                <div class="card-body" id="featureAnalysisResults">
                    <p class="text-muted">Load feature importance analysis.</p>
                    <button class="btn btn-primary" onclick="loadFeatureImportance()">
                        <i class="bi bi-arrow-repeat"></i> Load
                    </button>
                </div>
            </div>
        </div>
    `;
}

function loadFeatureImportance() {
    showLoading(true);
    fetch('/api/features/importance')
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                displayFeatureImportance(data.data);
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Failed to load', 'error'); console.error(e); });
}

// function displayFeatureImportance(data) {
//     const container = document.getElementById('featureAnalysisResults');
//     if (!container) return;
    
//     const importances = data.feature_importance || [];

//     let html = `
//         <div class="alert alert-info"><strong>Total Features:</strong> ${data.total_features}</div>
//         <h6>Top 10 Features</h6>
//         <div class="mt-2">
//     `;

//     importances.forEach((item, index) => {
//         const name = item[0] || item.feature || `Feature ${index + 1}`;
//         const score = item[1] || item.importance || 0;
//         const maxScore = importances[0]?.[1] || 1;
//         const percentage = maxScore > 0 ? (score / maxScore) * 100 : 0;
        
//         html += `
//             <div class="mb-2">
//                 <div class="d-flex justify-content-between">
//                     <span>${index + 1}. ${name}</span>
//                     <span>${(score * 100).toFixed(2)}%</span>
//                 </div>
//                 <div class="progress" style="height:8px;">
//                     <div class="progress-bar bg-primary" style="width:${Math.min(percentage, 100)}%"></div>
//                 </div>
//             </div>
//         `;
//     });

//     html += `</div>`;
//     container.innerHTML = html;
// }

// ==================== SINGLE PREDICTION ====================
function renderPrediction(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-magic"></i> Single Prediction</h5></div>
                <div class="card-body">
                    <form id="predictionForm" class="row g-2">
                        <div class="col-md-4">
                            <label class="form-label">Student ID</label>
                            <input type="text" class="form-control" id="studentId" value="S0001">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Gender</label>
                            <select class="form-select" id="gender">
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Age</label>
                            <input type="number" class="form-control" id="age" value="20">
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Department</label>
                            <select class="form-select" id="department">
                                <option value="Computer Science">Computer Science</option>
                                <option value="Engineering">Engineering</option>
                                <option value="Business">Business</option>
                                <option value="Mathematics">Mathematics</option>
                                <option value="Physics">Physics</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Course</label>
                            <select class="form-select" id="course">
                                <option value="CS101">CS101</option>
                                <option value="CS201">CS201</option>
                                <option value="ENG101">ENG101</option>
                                <option value="BUS101">BUS101</option>
                                <option value="MATH101">MATH101</option>
                                <option value="PHY101">PHY101</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Historical Grade</label>
                            <input type="number" class="form-control" id="historicalGrade" value="70" min="0" max="100">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Previous Grade</label>
                            <input type="number" class="form-control" id="previousGrade" value="65" min="0" max="100">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Assignment Score</label>
                            <input type="number" class="form-control" id="assignmentScore" value="68" min="0" max="100">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Quiz Score</label>
                            <input type="number" class="form-control" id="quizScore" value="62" min="0" max="100">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Midterm Score</label>
                            <input type="number" class="form-control" id="midtermScore" value="65" min="0" max="100">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Attendance %</label>
                            <input type="number" class="form-control" id="attendance" value="75" min="0" max="100">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">LMS Login Count</label>
                            <input type="number" class="form-control" id="lmsLogin" value="25" min="0">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">LMS Activity Count</label>
                            <input type="number" class="form-control" id="lmsActivity" value="60" min="0">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Study Hours</label>
                            <input type="number" class="form-control" id="studyHours" value="10" min="0" max="30">
                        </div>
                        <div class="col-12 mt-3">
                            <button type="submit" class="btn btn-primary btn-lg w-100">
                                <i class="bi bi-magic"></i> Predict Performance
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            <div id="predictionResult" style="display:none;">
                <div class="card">
                    <div class="card-header"><h5><i class="bi bi-check-circle"></i> Prediction Result</h5></div>
                    <div class="card-body prediction-result" id="resultContent"></div>
                </div>
            </div>
        </div>
    `;

    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            makePrediction();
        });
    }
}

function makePrediction() {
    const data = {
        student_id: document.getElementById('studentId')?.value || 'S0001',
        gender: document.getElementById('gender')?.value || 'Male',
        age: parseInt(document.getElementById('age')?.value) || 20,
        department: document.getElementById('department')?.value || 'Computer Science',
        course: document.getElementById('course')?.value || 'CS101',
        historical_grade: parseFloat(document.getElementById('historicalGrade')?.value) || 70,
        previous_grade: parseFloat(document.getElementById('previousGrade')?.value) || 65,
        assignment_score: parseFloat(document.getElementById('assignmentScore')?.value) || 68,
        quiz_score: parseFloat(document.getElementById('quizScore')?.value) || 62,
        midterm_score: parseFloat(document.getElementById('midtermScore')?.value) || 65,
        attendance_percentage: parseFloat(document.getElementById('attendance')?.value) || 75,
        lms_login_count: parseInt(document.getElementById('lmsLogin')?.value) || 25,
        lms_activity_count: parseInt(document.getElementById('lmsActivity')?.value) || 60,
        study_hours: parseFloat(document.getElementById('studyHours')?.value) || 10
    };

    showLoading(true);
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        showLoading(false);
        if (result.success) {
            displayPredictionResult(result.data);
            document.getElementById('predictionResult').style.display = 'block';
        } else {
            showToast('Error', result.message || 'Prediction failed', 'error');
        }
    })
    .catch(e => {
        showLoading(false);
        showToast('Error', 'Prediction failed: ' + e.message, 'error');
        console.error(e);
    });
}

// function displayPredictionResult(data) {
//     const content = document.getElementById('resultContent');
//     if (!content) return;

//     const statusClass = data.prediction === 'PASS' ? 'success' : 'danger';
//     const riskClass = `risk-${data.risk_level?.toLowerCase() || 'low'}`;
//     const confidence = ((data.confidence || 0) * 100).toFixed(1);
//     const failureProb = ((data.failure_probability || 0) * 100).toFixed(1);

//     let html = `
//         <div class="row g-3">
//             <div class="col-md-6 text-center">
//                 <h6>Prediction</h6>
//                 <div class="status-badge badge bg-${statusClass} fs-2">${data.prediction || 'UNKNOWN'}</div>
//             </div>
//             <div class="col-md-6 text-center">
//                 <h6>Risk Level</h6>
//                 <div class="${riskClass} fs-4" style="display:inline-block;padding:0.5rem 1.5rem;">${data.risk_level || 'UNKNOWN'}</div>
//             </div>
//         </div>
//         <div class="row g-3 mt-2">
//             <div class="col-md-6">
//                 <h6>Confidence</h6>
//                 <div class="confidence-meter"><div class="fill bg-primary" style="width:${confidence}%"></div></div>
//                 <div class="text-center mt-1">${confidence}%</div>
//             </div>
//             <div class="col-md-6">
//                 <h6>Failure Probability</h6>
//                 <div class="confidence-meter"><div class="fill bg-danger" style="width:${failureProb}%"></div></div>
//                 <div class="text-center mt-1">${failureProb}%</div>
//             </div>
//         </div>
//     `;

//     if (data.important_factors && data.important_factors.length) {
//         html += `<div class="mt-3"><h6>Key Factors</h6><div class="row g-1">`;
//         data.important_factors.forEach(f => {
//             const color = f.impact === 'HIGH' ? 'danger' : f.impact === 'MEDIUM' ? 'warning' : 'info';
//             html += `<div class="col-md-6"><div class="border rounded p-2"><strong>${f.feature}:</strong> ${f.value} <span class="badge bg-${color}">${f.impact}</span></div></div>`;
//         });
//         html += `</div></div>`;
//     }

//     if (data.recommendation) {
//         html += `<div class="mt-3"><div class="alert alert-info"><i class="bi bi-lightbulb"></i> <strong>Recommendation:</strong> ${data.recommendation}</div></div>`;
//     }

//     content.innerHTML = html;
// }

// ==================== BATCH PREDICTION ====================
function renderBatchPrediction(container) {
    container.innerHTML = `
        <div class="fade-in">
            <div class="card">
                <div class="card-header"><h5><i class="bi bi-upload"></i> Batch Prediction</h5></div>
                <div class="card-body">
                    <div class="upload-area" id="batchDropArea">
                        <i class="bi bi-cloud-upload upload-icon"></i>
                        <h5>Upload CSV for batch prediction</h5>
                        <p class="text-muted">File should contain student features</p>
                        <input type="file" id="batchFileInput" accept=".csv" style="display:none;">
                        <button class="btn btn-primary" onclick="document.getElementById('batchFileInput').click()">
                            <i class="bi bi-folder-open"></i> Browse CSV
                        </button>
                    </div>
                </div>
            </div>
            <div id="batchResults" style="display:none;">
                <div class="card">
                    <div class="card-header"><h5><i class="bi bi-table"></i> Results</h5></div>
                    <div class="card-body" id="batchResultsContent"></div>
                </div>
            </div>
        </div>
    `;
    setupBatchUploadHandlers();
}

function setupBatchUploadHandlers() {
    const dropArea = document.getElementById('batchDropArea');
    const fileInput = document.getElementById('batchFileInput');
    if (!dropArea || !fileInput) return;

    const newDropArea = dropArea.cloneNode(true);
    dropArea.parentNode.replaceChild(newDropArea, dropArea);
    const newFileInput = fileInput.cloneNode(true);
    fileInput.parentNode.replaceChild(newFileInput, fileInput);

    newDropArea.addEventListener('dragover', e => { e.preventDefault(); e.currentTarget.classList.add('dragover'); });
    newDropArea.addEventListener('dragleave', e => { e.preventDefault(); e.currentTarget.classList.remove('dragover'); });
    newDropArea.addEventListener('drop', e => {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        if (e.dataTransfer.files.length) batchPredict(e.dataTransfer.files[0]);
    });
    newDropArea.addEventListener('click', () => newFileInput.click());
    newFileInput.addEventListener('change', function() {
        if (this.files.length) batchPredict(this.files[0]);
    });
}

function batchPredict(file) {
    const formData = new FormData();
    formData.append('file', file);

    showLoading(true);
    fetch('/api/predict/batch', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                displayBatchResults(data.data);
                showToast('Success', data.message, 'success');
            } else {
                showToast('Error', data.message, 'error');
            }
        })
        .catch(e => { showLoading(false); showToast('Error', 'Batch prediction failed', 'error'); console.error(e); });
}

function displayBatchResults(data) {
    const container = document.getElementById('batchResults');
    const content = document.getElementById('batchResultsContent');
    if (!container || !content) return;
    container.style.display = 'block';

    const results = data.results || [];
    const summary = data.predictions_summary || {};
    const riskSummary = data.risk_summary || {};

    let html = `
        <div class="row g-2 mb-3">
            <div class="col-md-3 col-6"><div class="card bg-light p-2 text-center"><strong>Total</strong><span class="fs-4">${data.total}</span></div></div>
    `;
    Object.keys(summary).forEach(key => {
        const color = key === 'PASS' ? 'success' : 'danger';
        html += `<div class="col-md-3 col-6"><div class="card bg-${color} text-white p-2 text-center"><strong>${key}</strong><span class="fs-4">${summary[key]}</span></div></div>`;
    });
    html += `</div><div class="row g-2 mb-3">`;
    Object.keys(riskSummary).forEach(key => {
        const color = key === 'LOW' ? 'success' : key === 'MEDIUM' ? 'warning' : 'danger';
        html += `<div class="col-md-3 col-6"><div class="card bg-${color} text-white p-2 text-center"><strong>${key}</strong><span class="fs-4">${riskSummary[key]}</span></div></div>`;
    });
    html += `</div>`;

    html += `<div class="table-container"><table class="table table-striped table-hover mb-0">
        <thead><tr><th>ID</th><th>Prediction</th><th>Confidence</th><th>Risk</th><th>Recommendation</th></tr></thead><tbody>`;
    
    results.slice(0, 20).forEach(row => {
        const riskClass = `risk-${row.risk_level?.toLowerCase() || 'low'}`;
        html += `<tr>
            <td>${row.student_id || 'N/A'}</td>
            <td><span class="badge bg-${row.prediction === 'PASS' ? 'success' : 'danger'}">${row.prediction || 'UNKNOWN'}</span></td>
            <td>${((row.confidence || 0) * 100).toFixed(1)}%</td>
            <td><span class="${riskClass}">${row.risk_level || 'UNKNOWN'}</span></td>
            <td><small>${row.recommendation || 'N/A'}</small></td>
        </tr>`;
    });
    html += `</tbody></table></div>`;
    if (results.length > 20) html += `<p class="text-muted mt-2">Showing first 20 of ${results.length}</p>`;
    html += `<div class="mt-3 text-center"><button class="btn btn-success" onclick="downloadPredictions()"><i class="bi bi-download"></i> Download CSV</button></div>`;

    content.innerHTML = html;
    predictionResults = results;
}

function downloadPredictions() {
    if (!predictionResults.length) {
        showToast('Warning', 'No results to download', 'warning');
        return;
    }
    fetch('/api/predict/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results: predictionResults })
    })
    .then(r => r.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Success', 'Download started', 'success');
    })
    .catch(e => { showToast('Error', 'Download failed', 'error'); console.error(e); });
}

// ==================== UTILITY ====================
function showLoading(show) {
    let overlay = document.querySelector('.spinner-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'spinner-overlay';
        overlay.innerHTML = `<div class="spinner-border text-light" role="status"><span class="visually-hidden">Loading...</span></div>`;
        document.body.appendChild(overlay);
    }
    overlay.classList.toggle('active', show);
}

function showToast(title, message, type = 'info') {
    const toast = document.getElementById('liveToast');
    const toastTitle = document.getElementById('toastTitle');
    const toastMessage = document.getElementById('toastMessage');

    if (!toast || !toastTitle || !toastMessage) {
        alert(`${title}: ${message}`);
        return;
    }

    const colors = { success: 'text-success', error: 'text-danger', warning: 'text-warning', info: 'text-info' };
    toastTitle.className = `me-auto ${colors[type] || colors.info}`;
    toastTitle.textContent = title;
    toastMessage.textContent = message;

    new bootstrap.Toast(toast).show();
}

// static/js/app.js - Updated displayFeatureImportance function

function displayFeatureImportance(data) {
    const container = document.getElementById('featureAnalysisResults');
    if (!container) return;
    
    const importances = data.feature_importance || [];
    const totalFeatures = data.total_features || 0;
    const featureNames = data.feature_names || [];

    // Debug log
    console.log('Feature Importance Data:', data);
    console.log('Feature Names:', featureNames);

    let html = `
        <div class="alert alert-info">
            <strong>Total Features:</strong> ${totalFeatures}
        </div>
        <h6>Top 10 Most Important Features</h6>
        <div class="mt-3">
    `;

    if (importances.length === 0) {
        html += `<p class="text-muted">No feature importance data available.</p>`;
    } else {
        // Find the maximum importance for scaling
        const maxImportance = importances[0]?.importance || 1;
        
        importances.forEach((item, index) => {
            let name, score;
            
            // Handle different data formats
            if (Array.isArray(item)) {
                name = item[0];
                score = item[1];
            } else {
                name = item.feature || item.name || `Feature ${index + 1}`;
                score = item.importance || item.score || 0;
            }
            
            // If name is a number (like "0", "1", "2"), try to get actual name from feature_names
            if (name !== null && name !== undefined) {
                // Check if name is a number string
                const numName = parseInt(name);
                if (!isNaN(numName) && featureNames && featureNames.length > 0) {
                    if (numName < featureNames.length) {
                        name = featureNames[numName];
                        console.log(`Replaced index ${numName} with name: ${name}`);
                    }
                }
                
                // Also check if the name contains "feature" and we have feature names
                if (String(name).toLowerCase().includes('feature') && featureNames && featureNames.length > index) {
                    name = featureNames[index];
                    console.log(`Replaced generic name with: ${name}`);
                }
            }
            
            // If name is still generic or empty, use feature_names array
            if (!name || name === '' || name === '0' || name === '1' || name === '2' || name === '3' || name === '4' || 
                name === '5' || name === '6' || name === '7' || name === '8' || name === '9' || 
                String(name).toLowerCase().includes('feature')) {
                if (featureNames && featureNames.length > 0 && index < featureNames.length) {
                    name = featureNames[index];
                } else {
                    name = `Feature ${index + 1}`;
                }
            }
            
            // Clean up common patterns
            const nameMap = {
                'historical_grade': 'Historical Grade',
                'previous_grade': 'Previous Grade',
                'assignment_score': 'Assignment Score',
                'quiz_score': 'Quiz Score',
                'midterm_score': 'Midterm Score',
                'attendance_percentage': 'Attendance Percentage',
                'lms_login_count': 'LMS Login Count',
                'lms_activity_count': 'LMS Activity Count',
                'study_hours': 'Study Hours',
                'age': 'Age',
                'gender_male': 'Gender: Male',
                'gender_female': 'Gender: Female',
                'department_computer science': 'Department: Computer Science',
                'department_engineering': 'Department: Engineering',
                'department_business': 'Department: Business',
                'department_mathematics': 'Department: Mathematics',
                'department_physics': 'Department: Physics'
            };
            
            // Check if name matches any key in nameMap (case insensitive)
            const lowerName = String(name).toLowerCase();
            for (const [key, value] of Object.entries(nameMap)) {
                if (lowerName === key.toLowerCase() || lowerName.includes(key.toLowerCase())) {
                    name = value;
                    break;
                }
            }
            
            const percentage = maxImportance > 0 ? (score / maxImportance) * 100 : 0;
            
            html += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span><strong>${index + 1}.</strong> ${name}</span>
                        <span>${(score * 100).toFixed(2)}%</span>
                    </div>
                    <div class="progress" style="height: 8px; background: #e5e7eb;">
                        <div class="progress-bar bg-primary" style="width: ${Math.min(percentage, 100)}%; transition: width 1s ease;"></div>
                    </div>
                </div>
            `;
        });
    }

    html += `
        </div>
        <div class="mt-3">
            <small class="text-muted">
                <i class="bi bi-info-circle"></i> 
                Features with higher percentages have greater impact on the model's predictions.
            </small>
        </div>
    `;

    container.innerHTML = html;
}

// static/js/app.js - Updated displayPredictionResult function

function displayPredictionResult(data) {
    const content = document.getElementById('resultContent');
    if (!content) return;

    const statusClass = data.prediction === 'PASS' ? 'success' : 'danger';
    const riskClass = `risk-${data.risk_level?.toLowerCase() || 'low'}`;
    
    // Use the correct probabilities
    const confidence = ((data.confidence || 0) * 100).toFixed(1);
    const passProb = ((data.pass_probability || 0) * 100).toFixed(1);
    const failureProb = ((data.failure_probability || 0) * 100).toFixed(1);

    let html = `
        <div class="row g-3">
            <div class="col-md-6 text-center">
                <h6>Prediction</h6>
                <div class="status-badge badge bg-${statusClass} fs-2">${data.prediction || 'UNKNOWN'}</div>
            </div>
            <div class="col-md-6 text-center">
                <h6>Risk Level</h6>
                <div class="${riskClass} fs-4" style="display:inline-block;padding:0.5rem 1.5rem;">${data.risk_level || 'UNKNOWN'}</div>
            </div>
        </div>
        <div class="row g-3 mt-2">
            <div class="col-md-6">
                <h6>Confidence</h6>
                <div class="confidence-meter"><div class="fill bg-primary" style="width:${confidence}%"></div></div>
                <div class="text-center mt-1">${confidence}%</div>
            </div>
            <div class="col-md-6">
                <h6>${data.prediction === 'PASS' ? 'Pass' : 'Fail'} Probability</h6>
                <div class="confidence-meter"><div class="fill bg-${data.prediction === 'PASS' ? 'success' : 'danger'}" style="width:${data.prediction === 'PASS' ? passProb : failureProb}%"></div></div>
                <div class="text-center mt-1">${data.prediction === 'PASS' ? passProb : failureProb}%</div>
            </div>
        </div>
        <div class="row g-3 mt-2">
            <div class="col-md-6">
                <h6>Failure Probability</h6>
                <div class="confidence-meter"><div class="fill bg-danger" style="width:${failureProb}%"></div></div>
                <div class="text-center mt-1">${failureProb}%</div>
            </div>
            <div class="col-md-6">
                <h6>Pass Probability</h6>
                <div class="confidence-meter"><div class="fill bg-success" style="width:${passProb}%"></div></div>
                <div class="text-center mt-1">${passProb}%</div>
            </div>
        </div>
    `;

    if (data.important_factors && data.important_factors.length) {
        html += `<div class="mt-3"><h6>Risk Factors</h6><div class="row g-1">`;
        data.important_factors.forEach(f => {
            const color = f.impact === 'HIGH' ? 'danger' : f.impact === 'MEDIUM' ? 'warning' : 'info';
            html += `<div class="col-md-6"><div class="border rounded p-2"><strong>${f.feature}:</strong> ${f.value} <span class="badge bg-${color}">${f.impact}</span></div></div>`;
        });
        html += `</div></div>`;
    }

    if (data.recommendation) {
        const alertClass = data.risk_level === 'LOW' ? 'success' : 
                          data.risk_level === 'MEDIUM' ? 'warning' : 'danger';
        html += `<div class="mt-3"><div class="alert alert-${alertClass}"><i class="bi bi-lightbulb"></i> <strong>Recommendation:</strong> ${data.recommendation}</div></div>`;
    }

    content.innerHTML = html;
}