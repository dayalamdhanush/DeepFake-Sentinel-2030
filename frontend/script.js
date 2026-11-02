// frontend/script.js - DeepFake Sentinel 2030 Interactivity
// Production-Ready Frontend Controller

const API_BASE = (window.location.protocol.startsWith('http') && window.location.origin) ? window.location.origin : 'http://localhost:8000';

let selectedImageFile = null;
let selectedVideoFile = null;
let selectedAudioFile = null;
let latestAnalysisData = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    setupDragAndDrop();
    checkBackendStatus();
    loadModelStatus();
    loadHistory();
});

// ==========================================
// 1. NAVIGATION & TABS
// ==========================================
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('active', pane.id === `tab-${tabId}`);
    });

    const titles = {
        dashboard: ['Dashboard Overview', 'Explainable AI multi-modal forensic analysis suite'],
        image: ['Image DeepFake Detection', 'EfficientNet-B0 transfer learning with Grad-CAM heatmap explainability'],
        video: ['Video DeepFake Detection', 'ResNet-18 CNN + LSTM temporal sequence classification across 16 frames'],
        audio: ['Audio Synthetic Voice Detection', '16kHz Log-Mel spectrogram transformation + 3-Block CNN classifier'],
        multimodal: ['Multi-Modal Fusion Pipeline', 'Weighted probability score fusion across image, video, and audio'],
        investigation: ['Forensic Investigation Mode', 'Evidence breakdown separating model confidence from definitive proof'],
        history: ['Forensic Log History', 'SQLite database audit trail of past predictions'],
        models: ['Neural Network Model Status', 'Verifies presence and loadability of trained PyTorch checkpoints']
    };

    if (titles[tabId]) {
        document.getElementById('current-page-title').innerText = titles[tabId][0];
        document.getElementById('current-page-desc').innerText = titles[tabId][1];
    }

    if (tabId === 'history') loadHistory();
    if (tabId === 'models') loadModelStatus();
}

// ==========================================
// 2. BACKEND HEALTH CHECK
// ==========================================
async function checkBackendStatus() {
    const dot = document.getElementById('sys-status-dot');
    const text = document.getElementById('sys-status-text');
    const deviceBadge = document.getElementById('sys-device-badge');

    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            dot.className = 'pulse-dot green';
            text.innerText = 'Connected & Operational';
            deviceBadge.innerText = `Device: ${(data.device || 'CPU').toUpperCase()}`;
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch (err) {
        dot.className = 'pulse-dot red';
        text.innerText = 'Backend Offline';
        deviceBadge.innerText = 'Server Disconnected';
    }
}

// ==========================================
// 3. DRAG AND DROP SETUP
// ==========================================
function setupDragAndDrop() {
    setupZone('img-upload-zone', handleImageSelected);
    setupZone('vid-upload-zone', handleVideoSelected);
    setupZone('aud-upload-zone', handleAudioSelected);
}

function setupZone(zoneId, fileHandler) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.remove('drag-active');
        }, false);
    });

    zone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileHandler(files[0]);
        }
    }, false);
}

// ==========================================
// 4. IMAGE DETECTION PIPELINE
// ==========================================
function handleImageSelected(file) {
    if (!file) return;
    const validExts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
    const hasValidExt = validExts.some(ext => file.name.toLowerCase().endsWith(ext));
    if (!hasValidExt && !file.type.startsWith('image/')) {
        showCardError('img-result-box', 'Invalid Image Format: Please select a JPG, PNG, WEBP, or BMP file.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showCardError('img-result-box', 'File Size Limit Exceeded: Image must be 10 MB or less.');
        return;
    }

    selectedImageFile = file;
    document.getElementById('img-filename').innerText = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('img-preview').src = e.target.result;
        document.getElementById('img-preview-container').classList.remove('hidden');
        document.getElementById('img-result-box').classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function resetImageScan() {
    selectedImageFile = null;
    const input = document.getElementById('img-file-input');
    if (input) input.value = '';
    document.getElementById('img-preview-container').classList.add('hidden');
    document.getElementById('img-result-box').classList.add('hidden');
}

async function runImageAnalysis() {
    if (!selectedImageFile) return;
    const btn = document.getElementById('btn-run-img');
    const box = document.getElementById('img-result-box');
    
    btn.disabled = true;
    btn.innerText = 'Analyzing Image (EfficientNet-B0)...';
    showLoadingState(box, 'Running EfficientNet-B0 inference and generating Grad-CAM heatmap...');

    const formData = new FormData();
    formData.append('file', selectedImageFile);

    try {
        const res = await fetch(`${API_BASE}/predict/image`, { method: 'POST', body: formData });
        const data = await res.json();
        
        if (!res.ok) {
            showCardError('img-result-box', data.detail || `Server error: ${res.status}`);
            return;
        }

        renderModalityResult(box, data, 'Image', selectedImageFile.name);
        latestAnalysisData = { type: 'Image', filename: selectedImageFile.name, data };
        updateInvestigationView();
        loadHistory();
    } catch (err) {
        showCardError('img-result-box', 'Failed to connect to image endpoint: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerText = 'Run EfficientNet-B0 Analysis';
    }
}

// ==========================================
// 5. VIDEO DETECTION PIPELINE
// ==========================================
function handleVideoSelected(file) {
    if (!file) return;
    const validExts = ['.mp4', '.avi', '.mov', '.mkv'];
    const hasValidExt = validExts.some(ext => file.name.toLowerCase().endsWith(ext));
    if (!hasValidExt && !file.type.startsWith('video/')) {
        showCardError('vid-result-box', 'Invalid Video Format: Please select an MP4, AVI, MOV, or MKV file.');
        return;
    }
    if (file.size > 100 * 1024 * 1024) {
        showCardError('vid-result-box', 'File Size Limit Exceeded: Video must be 100 MB or less.');
        return;
    }

    selectedVideoFile = file;
    document.getElementById('vid-filename').innerText = file.name;
    document.getElementById('vid-preview-container').classList.remove('hidden');
    document.getElementById('vid-result-box').classList.add('hidden');
}

function resetVideoScan() {
    selectedVideoFile = null;
    const input = document.getElementById('vid-file-input');
    if (input) input.value = '';
    document.getElementById('vid-preview-container').classList.add('hidden');
    document.getElementById('vid-result-box').classList.add('hidden');
}

async function runVideoAnalysis() {
    if (!selectedVideoFile) return;
    const btn = document.getElementById('btn-run-vid');
    const box = document.getElementById('vid-result-box');
    
    btn.disabled = true;
    btn.innerText = 'Sampling 16 Frames & Running ResNet-18 + LSTM...';
    showLoadingState(box, 'Extracting 16 uniform frames and executing temporal LSTM classification...');

    const formData = new FormData();
    formData.append('file', selectedVideoFile);

    try {
        const res = await fetch(`${API_BASE}/predict/video`, { method: 'POST', body: formData });
        const data = await res.json();
        
        if (!res.ok) {
            showCardError('vid-result-box', data.detail || `Server error: ${res.status}`);
            return;
        }

        renderModalityResult(box, data, 'Video', selectedVideoFile.name);
        latestAnalysisData = { type: 'Video', filename: selectedVideoFile.name, data };
        updateInvestigationView();
        loadHistory();
    } catch (err) {
        showCardError('vid-result-box', 'Failed to connect to video endpoint: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerText = 'Run ResNet-18 + LSTM Analysis';
    }
}

// ==========================================
// 6. AUDIO DETECTION PIPELINE
// ==========================================
function handleAudioSelected(file) {
    if (!file) return;
    const validExts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a'];
    const hasValidExt = validExts.some(ext => file.name.toLowerCase().endsWith(ext));
    if (!hasValidExt && !file.type.startsWith('audio/')) {
        showCardError('aud-result-box', 'Invalid Audio Format: Please select a WAV, MP3, OGG, FLAC, or M4A file.');
        return;
    }
    if (file.size > 20 * 1024 * 1024) {
        showCardError('aud-result-box', 'File Size Limit Exceeded: Audio must be 20 MB or less.');
        return;
    }

    selectedAudioFile = file;
    document.getElementById('aud-filename').innerText = file.name;
    document.getElementById('aud-preview-container').classList.remove('hidden');
    document.getElementById('aud-result-box').classList.add('hidden');
}

function resetAudioScan() {
    selectedAudioFile = null;
    const input = document.getElementById('aud-file-input');
    if (input) input.value = '';
    document.getElementById('aud-preview-container').classList.add('hidden');
    document.getElementById('aud-result-box').classList.add('hidden');
}

async function runAudioAnalysis() {
    if (!selectedAudioFile) return;
    const btn = document.getElementById('btn-run-aud');
    const box = document.getElementById('aud-result-box');
    
    btn.disabled = true;
    btn.innerText = 'Generating Log-Mel Spectrogram & Running AudioCNN...';
    showLoadingState(box, 'Resampling to 16kHz, computing 128 log-Mel bins, and evaluating 3-Block CNN...');

    const formData = new FormData();
    formData.append('file', selectedAudioFile);

    try {
        const res = await fetch(`${API_BASE}/predict/audio`, { method: 'POST', body: formData });
        const data = await res.json();
        
        if (!res.ok) {
            showCardError('aud-result-box', data.detail || `Server error: ${res.status}`);
            return;
        }

        renderModalityResult(box, data, 'Audio', selectedAudioFile.name);
        latestAnalysisData = { type: 'Audio', filename: selectedAudioFile.name, data };
        updateInvestigationView();
        loadHistory();
    } catch (err) {
        showCardError('aud-result-box', 'Failed to connect to audio endpoint: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerText = 'Run Log-Mel CNN Analysis';
    }
}

// ==========================================
// 7. RESULT RENDERING
// ==========================================
function showLoadingState(container, message) {
    container.classList.remove('hidden');
    container.innerHTML = `
        <div style="padding: 24px; text-align: center;">
            <div style="display: inline-block; width: 36px; height: 36px; border: 3px solid rgba(0,242,254,0.2); border-top: 3px solid var(--primary-cyan); border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 12px;"></div>
            <p style="font-size: 14px; color: var(--text-main); font-weight: 500;">${message}</p>
            <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Running real PyTorch model forward pass...</p>
        </div>
    `;
}

function showCardError(containerId, message) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.classList.remove('hidden');
    container.innerHTML = `
        <div style="padding: 16px; border-left: 4px solid var(--danger-red); background: rgba(239, 68, 68, 0.08); border-radius: 6px;">
            <h4 style="color: var(--danger-red); margin-bottom: 4px;">⚠️ Error</h4>
            <p style="font-size: 13px; color: var(--text-main);">${message}</p>
        </div>
    `;
}

function renderModalityResult(container, data, mediaType, filename) {
    container.classList.remove('hidden');
    
    if (data.error === 'MODEL_NOT_TRAINED') {
        container.innerHTML = `
            <div class="flex-between">
                <h4>${mediaType} Forensic Result (${filename})</h4>
                <span class="badge-result badge-untrained">MODEL NOT TRAINED</span>
            </div>
            <p style="margin-top: 12px; color: var(--warning-amber);">
                ⚠️ The trained checkpoint <code>models/${mediaType.toLowerCase()}_...pth</code> is missing. 
                Per system guidelines, predictions are NOT hardcoded or fabricated. 
                Please train the model using <code>python scripts/train_${mediaType.toLowerCase()}.py</code>.
            </p>
        `;
        return;
    }

    if (data.error) {
        container.innerHTML = `<h4 style="color: var(--danger-red);">Analysis Error: ${data.error}</h4>`;
        return;
    }

    const isFake = data.label === 'FAKE';
    const badgeClass = isFake ? 'badge-fake' : 'badge-real';
    const confPercent = (data.confidence * 100).toFixed(1);
    const fakeProbPercent = (data.fake_probability * 100).toFixed(1);

    let extraHtml = '';
    if (data.gradcam) {
        extraHtml = `
            <div class="gradcam-container" style="margin-top: 20px;">
                <div class="gradcam-box">
                    <h5 style="margin-bottom: 8px; color: var(--text-muted);">Input Image</h5>
                    <img src="${document.getElementById('img-preview').src}" alt="Input" style="max-height: 240px; border-radius: 8px; border: 1px solid var(--border-color);">
                </div>
                <div class="gradcam-box">
                    <h5 style="margin-bottom: 8px; color: var(--primary-cyan);">Grad-CAM Visual Heatmap (Layer: features.7)</h5>
                    <img src="${data.gradcam}" alt="Grad-CAM Heatmap" style="max-height: 240px; border-radius: 8px; border: 1px solid var(--border-accent);">
                </div>
            </div>
            <p style="font-size: 11px; color: var(--text-muted); margin-top: 8px;">
                ${isFake ? '🔴 Red/Yellow highlights indicate anomalous spatial regions corresponding to suspected deepfake manipulation patterns.' : '🟢 / 🔴 Highlights indicate primary facial landmark regions evaluated by EfficientNet-B0 confirming natural biometric consistency.'}
            </p>
        `;
    } else if (data.sampled_frames) {
        extraHtml = `
            <div style="margin-top: 16px;">
                <h5 style="margin-bottom: 8px; color: var(--text-muted);">Sampled Video Frames (16 Uniform Steps)</h5>
                <div style="display: flex; gap: 6px; overflow-x: auto; padding: 8px 0;">
                    ${data.sampled_frames.map(f => `<span style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px; font-size: 11px; white-space: nowrap;">#${f.frame_index} (${f.timestamp_sec}s)</span>`).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = `
        <div class="flex-between">
            <h4>${mediaType} Forensic Analysis Result</h4>
            <span class="badge-result ${badgeClass}">${data.label}</span>
        </div>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-label">Classification</div>
                <div class="metric-val" style="color: ${isFake ? 'var(--danger-red)' : 'var(--success-green)'}">${data.label}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-val">${confPercent}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Fake Probability</div>
                <div class="metric-val">${fakeProbPercent}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Level</div>
                <div class="metric-val" style="color: var(--primary-cyan)">${data.risk}</div>
            </div>
        </div>

        <p style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">Inference completed in ${data.processing_time_sec}s and recorded to SQLite forensic history.</p>
        ${extraHtml}
    `;
}

// ==========================================
// 8. MULTI-MODAL FUSION PIPELINE
// ==========================================
async function runMultiModalFusion() {
    const imgFile = document.getElementById('fusion-img-input').files[0];
    const vidFile = document.getElementById('fusion-vid-input').files[0];
    const audFile = document.getElementById('fusion-aud-input').files[0];

    if (!imgFile && !vidFile && !audFile) {
        showCardError('fusion-result-box', 'Please select at least one media file for multi-modal analysis.');
        return;
    }

    const fusionBox = document.getElementById('fusion-result-box');
    showLoadingState(fusionBox, 'Executing multi-modal pipelines across active media inputs...');

    const results = {};

    if (imgFile) {
        const formData = new FormData();
        formData.append('file', imgFile);
        try {
            const res = await fetch(`${API_BASE}/predict/image`, { method: 'POST', body: formData });
            if (res.ok) results.image = await res.json();
        } catch(e) {}
    }

    if (vidFile) {
        const formData = new FormData();
        formData.append('file', vidFile);
        try {
            const res = await fetch(`${API_BASE}/predict/video`, { method: 'POST', body: formData });
            if (res.ok) results.video = await res.json();
        } catch(e) {}
    }

    if (audFile) {
        const formData = new FormData();
        formData.append('file', audFile);
        try {
            const res = await fetch(`${API_BASE}/predict/audio`, { method: 'POST', body: formData });
            if (res.ok) results.audio = await res.json();
        } catch(e) {}
    }

    // Call fusion endpoint
    try {
        const fuseRes = await fetch(`${API_BASE}/predict/fusion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(results)
        });
        const fuseData = await fuseRes.json();
        renderFusionResult(fusionBox, fuseData, results);
        latestAnalysisData = { type: 'Fusion', data: fuseData, modalResults: results };
        updateInvestigationView();
        loadHistory();
    } catch(err) {
        showCardError('fusion-result-box', 'Fusion endpoint error: ' + err.message);
    }
}

function renderFusionResult(container, fuseData, rawResults) {
    if (fuseData.error) {
        container.innerHTML = `
            <div style="padding: 16px; border-left: 4px solid var(--warning-amber); background: rgba(245, 158, 11, 0.08); border-radius: 6px;">
                <h4 style="color: var(--warning-amber);">Multi-Modal Fusion Notice</h4>
                <p style="margin-top: 8px;">${fuseData.message || fuseData.error}</p>
            </div>
        `;
        return;
    }

    const isFake = fuseData.final_label === 'FAKE';
    const badgeClass = isFake ? 'badge-fake' : 'badge-real';

    let modalitiesListHtml = '';
    for (const [mod, res] of Object.entries(rawResults)) {
        const status = res.error ? `<span style="color:var(--warning-amber)">NOT TRAINED</span>` : `<span style="color:${res.label==='FAKE'?'var(--danger-red)':'var(--success-green)'}">${res.label} (${(res.fake_probability*100).toFixed(1)}%)</span>`;
        modalitiesListHtml += `<div style="padding: 10px; background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 6px; font-size: 13px;"><strong>${mod.toUpperCase()}:</strong> ${status}</div>`;
    }

    container.innerHTML = `
        <div class="flex-between">
            <h4>Combined Multi-Modal Forensic Assessment</h4>
            <span class="badge-result ${badgeClass}">${fuseData.final_label}</span>
        </div>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-label">Final Label</div>
                <div class="metric-val" style="color: ${isFake ? 'var(--danger-red)' : 'var(--success-green)'}">${fuseData.final_label}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Fused Confidence</div>
                <div class="metric-val">${(fuseData.final_confidence * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Score</div>
                <div class="metric-val">${fuseData.risk_score} / 100</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Level</div>
                <div class="metric-val" style="color: var(--primary-cyan)">${fuseData.risk_level}</div>
            </div>
        </div>

        <h5 style="margin-bottom: 8px; color: var(--text-muted);">Active Modalities Contributed:</h5>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;">
            ${modalitiesListHtml}
        </div>

        <p style="font-size: 11px; color: var(--text-muted);">${fuseData.threshold_notice}</p>
    `;
}

// ==========================================
// 9. INVESTIGATION VIEW
// ==========================================
function updateInvestigationView() {
    const container = document.getElementById('investigation-content');
    if (!latestAnalysisData) {
        container.innerHTML = '<p class="placeholder-text">No active scan loaded. Run an image, video, audio, or multi-modal scan first.</p>';
        return;
    }

    const { type, data } = latestAnalysisData;
    const label = data.label || data.final_label || 'COMPLETED';
    const conf = data.confidence !== undefined ? `${(data.confidence * 100).toFixed(1)}%` : (data.final_confidence !== undefined ? `${(data.final_confidence * 100).toFixed(1)}%` : 'N/A');
    const risk = data.risk || data.risk_level || 'N/A';

    container.innerHTML = `
        <div style="border-left: 3px solid var(--primary-cyan); padding-left: 16px; margin-bottom: 20px;">
            <h4 style="font-size: 18px; color: var(--primary-cyan);">FORENSIC INVESTIGATION REPORT (${type.toUpperCase()})</h4>
            <p style="font-size: 13px; color: var(--text-muted);">Detailed breakdown of neural network evidence for forensic review.</p>
        </div>

        <div style="background: rgba(0, 242, 254, 0.04); border: 1px solid var(--border-accent); border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>Latest Evaluated Scan:</strong> ${type} Forensic Scan
                </div>
                <div>
                    <strong>Verdict:</strong> <span style="font-weight: 700; color: ${label === 'FAKE' ? 'var(--danger-red)' : 'var(--success-green)'}">${label}</span> (Confidence: ${conf}, Risk: ${risk})
                </div>
            </div>
        </div>

        <div class="features-list grid-2">
            <div class="feature-item">
                <div class="feature-icon">🔍</div>
                <div>
                    <h4>1. Preprocessing & Tensor Normalization</h4>
                    <p>Standardized transformation pipeline (ImageNet normalization, 16-frame uniform decimation, or 16kHz Log-Mel spectrogram generation).</p>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <div>
                    <h4>2. Model Architecture & Forward Pass</h4>
                    <p>Live PyTorch inference on CPU/CUDA yielding raw logit outputs and calibrated softmax probabilities.</p>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div>
                    <h4>3. Empirical Risk Stratification</h4>
                    <p>Mapped to predefined forensic safety intervals: 0-40% (LOW), 40-70% (MEDIUM), 70-100% (HIGH).</p>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📜</div>
                <div>
                    <h4>4. Explainable Evidence (XAI)</h4>
                    <p>Grad-CAM spatial activation overlays provide visual artifact localization rather than black-box verdicts.</p>
                </div>
            </div>
        </div>
    `;
}

// ==========================================
// 10. HISTORY TABLE
// ==========================================
async function loadHistory() {
    const tbody = document.getElementById('history-tbody');
    try {
        const res = await fetch(`${API_BASE}/history`);
        const data = await res.json();
        if (!data.history || data.history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center">No forensic prediction records found in SQLite database.</td></tr>';
            return;
        }
        tbody.innerHTML = data.history.map(r => `
            <tr>
                <td>#${r.id}</td>
                <td><strong>${r.filename}</strong></td>
                <td><span class="stat-tag">${(r.media_type || '').toUpperCase()}</span></td>
                <td><span style="font-weight: 700; color: ${r.label === 'FAKE' ? 'var(--danger-red)' : 'var(--success-green)'}">${r.label}</span></td>
                <td>${(r.confidence * 100).toFixed(1)}%</td>
                <td><span class="stat-tag">${r.risk}</span></td>
                <td>${r.timestamp ? new Date(r.timestamp).toLocaleString() : 'N/A'}</td>
            </tr>
        `).join('');
    } catch(err) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color:var(--danger-red)">Failed to load history: ${err.message}</td></tr>`;
    }
}

// ==========================================
// 11. MODEL STATUS
// ==========================================
async function loadModelStatus() {
    const grid = document.getElementById('models-grid');
    try {
        const res = await fetch(`${API_BASE}/model-status`);
        const data = await res.json();

        // Update dashboard shortcuts
        const imgModel = data.models.find(m => m.modality === 'image');
        const vidModel = data.models.find(m => m.modality === 'video');
        const audModel = data.models.find(m => m.modality === 'audio');

        if (document.getElementById('dash-img-status') && imgModel) {
            document.getElementById('dash-img-status').innerText = imgModel.status === 'TRAINED' ? 'Status: Trained & Ready' : 'Status: Model Not Trained';
        }
        if (document.getElementById('dash-vid-status') && vidModel) {
            document.getElementById('dash-vid-status').innerText = vidModel.status === 'TRAINED' ? 'Status: Trained & Ready' : 'Status: Model Not Trained';
        }
        if (document.getElementById('dash-aud-status') && audModel) {
            document.getElementById('dash-aud-status').innerText = audModel.status === 'TRAINED' ? 'Status: Trained & Ready' : 'Status: Model Not Trained';
        }

        grid.innerHTML = data.models.map(m => {
            const isTrained = m.status === 'TRAINED';
            return `
                <div class="model-card ${isTrained ? 'trained' : 'untrained'}">
                    <div class="flex-between" style="margin-bottom: 12px;">
                        <h4>${m.name}</h4>
                        <span class="badge-result ${isTrained ? 'badge-real' : 'badge-untrained'}">${m.status.replace('_', ' ')}</span>
                    </div>
                    <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">${m.architecture}</p>
                    <div style="font-size: 11px; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px;">
                        <div><strong>Checkpoint:</strong> <code>${m.checkpoint_path}</code></div>
                        <div style="margin-top: 4px;"><strong>Target Device:</strong> ${data.device.toUpperCase()}</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch(err) {
        grid.innerHTML = `<p style="color: var(--danger-red)">Failed to retrieve model pipeline status: ${err.message}</p>`;
    }
}
