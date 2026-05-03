// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null;   // FIX: store file here — fi.files is read-only in most browsers
let currentJobId = null;
let eventSource   = null;

// ── DOM refs ────────────────────────────────────────────────────────────────
const dz        = document.getElementById('dropzone');
const fi        = document.getElementById('fileInput');
const fileCard  = document.getElementById('fileChosen');

// ── Drag & Drop ─────────────────────────────────────────────────────────────
dz.addEventListener('dragover', e => {
  e.preventDefault();
  dz.classList.add('drag-over');
});

dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));

dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fi.addEventListener('change', () => {
  if (fi.files[0]) setFile(fi.files[0]);
});

// ── File selection ───────────────────────────────────────────────────────────
function setFile(file) {
  selectedFile = file;   // FIX: save to our own variable

  // Hide dropzone, show file card
  dz.classList.add('hidden');
  fileCard.classList.add('visible');
  document.getElementById('fileChosenName').textContent =
    '▸ ' + file.name + '  (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';

  document.getElementById('btnStart').disabled = false;
}

function removeFile() {
  selectedFile = null;
  fi.value = '';

  // Show dropzone, hide file card
  dz.classList.remove('hidden');
  fileCard.classList.remove('visible');

  document.getElementById('btnStart').disabled = true;
}

// ── Toggle buttons ───────────────────────────────────────────────────────────
function toggle(btn) {
  btn.classList.toggle('active');
}

// ── Step tracker ─────────────────────────────────────────────────────────────
const STEP_MAP = {
  'STEP 1': 1, 'STEP 2': 2, 'STEP 3': 3, 'STEP 4': 4, 'STEP 5': 5,
};
let currentStep = 0;

function setStep(n) {
  ['s1', 's2', 's3', 's4', 's5'].forEach((id, i) => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
    if (i < n - 1)      el.classList.add('done');
    else if (i === n - 1) el.classList.add('active');
  });
  const pct = Math.round((n / 5) * 85);
  const bar = document.getElementById('progressBar');
  bar.classList.remove('indeterminate');
  bar.style.width = pct + '%';
  currentStep = n;
}

// ── Log terminal ──────────────────────────────────────────────────────────────
function appendLog(line) {
  const t   = document.getElementById('terminal');
  const div = document.createElement('div');

  let cls = 'log-info';
  if      (line.includes('[WARNING]') || line.toLowerCase().includes('warning')) cls = 'log-warn';
  else if (line.includes('[ERROR]')   || line.toLowerCase().includes('error'))   cls = 'log-error';
  else if (line.match(/\[STEP \d/))                                               cls = 'log-step';
  else if (/done|DONE|saved/i.test(line))                                         cls = 'log-done';

  div.className   = cls;
  div.textContent = line;
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;

  // Advance step indicator
  for (const [key, n] of Object.entries(STEP_MAP)) {
    if (line.includes(key)) { setStep(n); break; }
  }

  document.getElementById('progressStatus').textContent = line.slice(0, 90);
}

// ── Start job ─────────────────────────────────────────────────────────────────
async function startJob() {
  if (!selectedFile) return;   // FIX: use our stored file, not fi.files[0]

  // Reset UI
  document.getElementById('errorBanner').classList.remove('visible');
  document.getElementById('transcriptBox').classList.remove('visible');
  document.getElementById('downloads').classList.remove('visible');
  document.getElementById('terminal').innerHTML = '';
  document.getElementById('progressSection').classList.add('visible');
  document.getElementById('btnStart').disabled = true;

  const bar = document.getElementById('progressBar');
  bar.classList.add('indeterminate');
  bar.style.width = '0%';
  bar.style.background = 'var(--accent)';

  currentStep = 0;
  ['s1', 's2', 's3', 's4', 's5'].forEach(id =>
    document.getElementById(id).classList.remove('active', 'done')
  );
  document.getElementById('progressStatus').textContent = 'Uploading...';

  // Build FormData
  const form = new FormData();
  form.append('video',           selectedFile);   // FIX: use selectedFile
  form.append('language',        document.getElementById('optLang').value);
  form.append('model',           document.getElementById('optModel').value);
  form.append('llm_provider',    document.getElementById('optLLM').value);
  form.append('llm_postprocess', document.getElementById('togLLM').classList.contains('active')       ? '1' : '0');
  form.append('restore_audio',   document.getElementById('togRestore').classList.contains('active')   ? '1' : '0');
  form.append('no_enhance',      document.getElementById('togNoEnhance').classList.contains('active') ? '1' : '0');

  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');

    currentJobId = data.job_id;
    listenProgress(currentJobId);
  } catch (err) {
    showError(err.message);
    document.getElementById('btnStart').disabled = false;
  }
}

// ── SSE progress ──────────────────────────────────────────────────────────────
function listenProgress(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/progress/' + jobId);

  eventSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'log') {
      appendLog(msg.text);
    } else if (msg.type === 'done') {
      eventSource.close();
      onDone(msg);
    } else if (msg.type === 'error') {
      eventSource.close();
      showError(msg.text);
      document.getElementById('btnStart').disabled = false;
    }
    // 'ping' — silently ignored
  };

  eventSource.onerror = () => eventSource.close();
}

// ── Done ──────────────────────────────────────────────────────────────────────
function onDone(msg) {
  const bar = document.getElementById('progressBar');
  bar.classList.remove('indeterminate');
  bar.style.width = '100%';
  document.getElementById('progressStatus').textContent = '✓ Processing complete';
  document.getElementById('btnStart').disabled = false;
  ['s1', 's2', 's3', 's4', 's5'].forEach(id =>
    document.getElementById(id).classList.add('done')
  );

  if (msg.transcript) {
    document.getElementById('transcriptText').innerHTML = formatTranscript(msg.transcript);
    document.getElementById('transcriptBox').classList.add('visible');
  }

  buildDownloads(msg.files, msg.job_id);
}

// ── Format transcript ────────────────────────────────────────────────────────
function formatTranscript(raw) {
  return raw
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/(\[предположение\])/g,   '<span class="tag-assumption">$1</span>')
    .replace(/(\[неразборчиво\])/g,    '<span class="tag-inaudible">$1</span>')
    .replace(/(\[[^\]]+\/[^\]]+\])/g,  '<span class="tag-variant">$1</span>')
    .replace(/(\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\])/g,
             '<span class="ts">$1</span>');
}

// ── Download cards ───────────────────────────────────────────────────────────
function buildDownloads(files, jobId) {
  const container = document.getElementById('downloads');
  container.innerHTML = '';

  const FILE_META = {
    'audio_enhanced.wav':   { icon: '🔊', title: 'Enhanced Audio',    desc: 'Noise reduced · denoised',    badge: 'badge-audio' },
    'audio_restored.wav':   { icon: '🤖', title: 'AI Restored Audio', desc: 'TTS spliced · unclear fixed', badge: 'badge-audio' },
    'transcript.txt':       { icon: '📄', title: 'Transcript',        desc: 'With timestamps & tags',      badge: 'badge-txt'   },
    'transcript_plain.txt': { icon: '📝', title: 'Plain Text',        desc: 'Clean · ready to read',       badge: 'badge-txt'   },
    'transcript.json':      { icon: '{ }', title: 'JSON Data',        desc: 'Full metadata · confidence',  badge: 'badge-txt'   },
  };

  // Video files (dynamic name)
  files.forEach(f => {
    if (/_fixed\.\w+$/.test(f)) {
      FILE_META[f] = { icon: '🎬', title: 'Restored Video', desc: 'Enhanced audio · original video', badge: 'badge-video' };
    }
  });

  files.forEach(fname => {
    const meta = FILE_META[fname] || { icon: '📁', title: fname, desc: 'Output file', badge: 'badge-txt' };
    const card = document.createElement('a');
    card.className = 'dl-card';
    card.href      = `/download/${jobId}/${fname}`;
    card.download  = fname;
    card.innerHTML = `
      <span class="dl-icon">${meta.icon}</span>
      <span class="dl-title">${meta.title}</span>
      <span class="dl-desc">${meta.desc}</span>
      <span class="dl-badge ${meta.badge}">${fname.split('.').pop().toUpperCase()}</span>
    `;
    container.appendChild(card);
  });

  container.classList.add('visible');
}

// ── Error ─────────────────────────────────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('errorBanner');
  el.textContent = '✖ ' + msg;
  el.classList.add('visible');
  const bar = document.getElementById('progressBar');
  bar.classList.remove('indeterminate');
  bar.style.background = 'var(--danger)';
  bar.style.width = '100%';
}
