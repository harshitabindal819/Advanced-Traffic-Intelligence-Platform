let selectedFile = null;
let currentVideoId = null;

const canvas = document.getElementById('calibCanvas');
const ctx = canvas.getContext('2d');
let bgImage = new Image();

let clickedPoints = [];
const laneColors = [
    'rgba(0, 204, 255, 0.5)', 'rgba(255, 204, 0, 0.5)', 'rgba(255, 102, 0, 0.5)',
    'rgba(255, 0, 255, 0.5)', 'rgba(0, 255, 0, 0.5)', 'rgba(0, 255, 255, 0.5)'
];

function handleDragOver(e) { e.preventDefault(); }
function handleDrop(e) { e.preventDefault(); if (e.dataTransfer.files[0]) uploadAndExtract(e.dataTransfer.files[0]); }
function handleFile(input) { if (input.files[0]) uploadAndExtract(input.files[0]); }

function uploadAndExtract(file) {
  selectedFile = file;
  document.getElementById('selectedFile').style.display = 'inline-block';
  document.getElementById('selectedFile').textContent = `📎 ${file.name} - Extracting frame...`;
  
  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload_video', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      if(data.error) throw new Error(data.error);
      currentVideoId = data.video_id;
      
      bgImage.src = '/static/' + data.frame_url + '?t=' + Date.now();
      bgImage.onload = () => {
          canvas.width = bgImage.naturalWidth;
          canvas.height = bgImage.naturalHeight;
          redrawCanvas();
          
          document.getElementById('uploadZone').style.display = 'none';
          document.getElementById('calibrationPanel').style.display = 'block';
      };
    })
    .catch(err => alert('Upload failed: ' + err.message));
}

// Unlimited Clicks Logic
canvas.addEventListener('mousedown', function(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    clickedPoints.push([Math.round(x), Math.round(y)]);
    updateInstructions();
    redrawCanvas();
});

function updateInstructions() {
    const instr = document.getElementById('calibrationInstruction');
    const btnDet = document.getElementById('btnRunDet');
    const btnSim = document.getElementById('btnSimulate');

    const totalLanesCompleted = Math.floor(clickedPoints.length / 4);
    const ptsNeeded = 4 - (clickedPoints.length % 4);

    // If at least one lane is fully completed, enable the buttons
    if (totalLanesCompleted > 0 && ptsNeeded === 4) {
        instr.textContent = `✓ ${totalLanesCompleted} Lane(s) defined. Draw another or Run Analysis.`;
        instr.style.color = "#00ff9d";
        btnDet.disabled = false; btnDet.style.opacity = 1;
        btnSim.disabled = false; btnSim.style.opacity = 1;
    } else {
        instr.textContent = `Click ${ptsNeeded} more point(s) to define Lane ${totalLanesCompleted + 1}`;
        instr.style.color = "var(--accent2)";
        // Disable buttons if halfway through drawing a lane
        btnDet.disabled = true; btnDet.style.opacity = 0.5;
        btnSim.disabled = true; btnSim.style.opacity = 0.5;
    }
}

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bgImage, 0, 0);

    const totalCompletedLanes = Math.floor(clickedPoints.length / 4);
    
    for (let i = 0; i < totalCompletedLanes; i++) {
        const color = laneColors[i % laneColors.length];
        drawPolygon(clickedPoints.slice(i*4, (i+1)*4), color);
    }

    const activePts = clickedPoints.slice(totalCompletedLanes * 4);
    if (activePts.length > 0) {
        ctx.beginPath();
        ctx.moveTo(activePts[0][0], activePts[0][1]);
        for (let i = 1; i < activePts.length; i++) {
            ctx.lineTo(activePts[i][0], activePts[i][1]);
        }
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.stroke();

        activePts.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#ff0000';
            ctx.fill();
        });
    }
}

function drawPolygon(pts, color) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) { ctx.lineTo(pts[i][0], pts[i][1]); }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function resetCalibration() {
    // Remove the last active or fully completed lane
    if (clickedPoints.length % 4 !== 0) {
        // Drop the partially completed points
        clickedPoints = clickedPoints.slice(0, Math.floor(clickedPoints.length / 4) * 4);
    } else if (clickedPoints.length >= 4) {
        // Drop the last full lane
        clickedPoints = clickedPoints.slice(0, clickedPoints.length - 4);
    }
    
    updateInstructions();
    redrawCanvas();
}

function submitCalibration(simulateEmergency) {
  if (clickedPoints.length < 4 || clickedPoints.length % 4 !== 0) return;
  
  document.getElementById('calibrationPanel').style.display = 'none';
  document.getElementById('loadingPanel').style.display = 'block';

  // Chunk the flat array into groups of 4 points
  const lanesPayload = [];
  for (let i = 0; i < clickedPoints.length; i += 4) {
      lanesPayload.push(clickedPoints.slice(i, i + 4));
  }

  fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
          video_id: currentVideoId,
          lanes: lanesPayload,
          simulate_emergency: simulateEmergency
      })
  })
  .then(res => res.json())
  .then(data => {
      if(data.error) throw new Error(data.error);
      
      document.getElementById('loadingPanel').style.display = 'none';
      document.getElementById('resultsPanel').style.display = 'block';

      // 1. DYNAMICALLY GENERATE LANE STATS
      const statsContainer = document.getElementById('dynamicLaneStats');
      statsContainer.innerHTML = ''; // Clear container
      
      const colorPalette = ['#00ccff', '#ffcc00', '#ff6600', '#ff00ff', '#00ff00', '#00ffff'];

      Object.keys(data.lane_densities).forEach((laneName, index) => {
          const val = data.lane_densities[laneName];
          const div = document.createElement('div');
          div.className = 'stat-card';
          
          // Apply a subtle border tint mapping to the lane colors
          const themeColor = colorPalette[index % colorPalette.length];
          div.style.borderColor = themeColor;
          div.style.background = `rgba(${hexToRgb(themeColor)}, 0.05)`;

          div.innerHTML = `
              <div class="stat-value" style="color: ${themeColor}">${val}</div>
              <div class="stat-label">${laneName.toUpperCase()}</div>
          `;
          statsContainer.appendChild(div);
      });

      // Update remaining generic stats
      document.getElementById('statConf').textContent = Math.round(parseFloat(data.avg_confidence) * 100) + '%';
      document.getElementById('statFrames').textContent = data.frames_processed;

      // Emergency Routing Logic
      const dashboard = document.getElementById('emergencyDashboard');
      const statusInd = document.getElementById('statusIndicator');
      document.getElementById('recommendedLaneTxt').textContent = data.recommended_lane.toUpperCase();

      if (data.emergency_detected) {
          dashboard.classList.add('alert');
          statusInd.textContent = "🚨 EMERGENCY VEHICLE DETECTED 🚨";
      } else {
          dashboard.classList.remove('alert');
          statusInd.textContent = "NORMAL TRAFFIC FLOW";
      }

      // Output Video
      const outputSrc = '/static/' + data.output + '?t=' + Date.now();
      const vid = document.getElementById('outputVideo');
      vid.src = outputSrc; 
      vid.style.display = 'block';

      document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth' });
  })
  .catch(err => {
      document.getElementById('loadingPanel').style.display = 'none';
      alert('Analysis failed: ' + err.message);
  });
}

// Helper to tint backgrounds dynamically
function hexToRgb(hex) {
    const bigint = parseInt(hex.replace('#',''), 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return `${r}, ${g}, ${b}`;
}

function resetPage() { location.reload(); }