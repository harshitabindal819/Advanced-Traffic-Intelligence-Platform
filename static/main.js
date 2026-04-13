let selectedFile = null;

function handleDragOver(e) { e.preventDefault(); }
function handleDragLeave(e) { }
function handleDrop(e) {
  e.preventDefault();
  if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
}

function handleFile(input) { if (input.files[0]) processFile(input.files[0]); }

function processFile(file) {
  selectedFile = file;
  document.getElementById('selectedFile').style.display = 'inline-block';
  document.getElementById('selectedFile').textContent = `📎 ${file.name}`;
  document.getElementById('previewPanel').style.display = 'block';

  const url = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    document.getElementById('previewImg').src = url;
    document.getElementById('previewImg').style.display = 'block';
    document.getElementById('previewVideo').style.display = 'none';
  } else {
    document.getElementById('previewVideo').src = url;
    document.getElementById('previewVideo').style.display = 'block';
    document.getElementById('previewImg').style.display = 'none';
  }
  document.getElementById('previewPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function runDetection() {
  if (!selectedFile) return;
  document.getElementById('previewPanel').style.display = 'none';
  document.getElementById('loadingPanel').style.display = 'block';
  document.getElementById('resultsPanel').style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedFile);

  fetch('/detect', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      document.getElementById('loadingPanel').style.display = 'none';
      document.getElementById('resultsPanel').style.display = 'block';

      // Stats Update
      document.getElementById('statVehicles').textContent = data.total_vehicles;
      document.getElementById('statZone').textContent = data.zone_density;
      document.getElementById('statConf').textContent = Math.round(parseFloat(data.avg_confidence) * 100) + '%';
      document.getElementById('statFrames').textContent = data.frames_processed || '1';

      // Render Class Breakdown dynamically
      const breakdownEl = document.getElementById('classBreakdown');
      breakdownEl.innerHTML = '';
      const emojis = { car: '🚙', motorcycle: '🏍️', bus: '🚌', truck: '🚚', bicycle: '🚲' };
      
      for (const [vClass, count] of Object.entries(data.class_breakdown)) {
          if (count > 0) {
              const div = document.createElement('div');
              div.className = 'breakdown-item';
              div.innerHTML = `<span style="font-size:1.5em">${emojis[vClass]}</span> ${vClass.toUpperCase()}: <span style="color:var(--accent); font-weight:bold">${count}</span>`;
              breakdownEl.appendChild(div);
          }
      }

      // Output Video/Image setup
      const outputSrc = '/static/' + data.output + '?t=' + Date.now();
      const dlBtn = document.getElementById('downloadBtn');

      if (data.type === 'image') {
        const img = document.getElementById('outputImg');
        img.src = outputSrc; img.style.display = 'block';
        document.getElementById('outputVideo').style.display = 'none';
        dlBtn.href = outputSrc; dlBtn.download = 'traffic_density_zones.jpg';
      } else {
        const vid = document.getElementById('outputVideo');
        vid.src = outputSrc; vid.style.display = 'block';
        document.getElementById('outputImg').style.display = 'none';
        dlBtn.href = outputSrc; dlBtn.download = 'traffic_density_zones.mp4';
      }

      document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(err => {
      document.getElementById('loadingPanel').style.display = 'none';
      alert('Detection failed: ' + err.message);
    });
}

function resetPage() { location.reload(); }