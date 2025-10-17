// 等待DOM完全載入後再執行
document.addEventListener('DOMContentLoaded', () => {
    // --- 變數與常數定義 ---
    const canvas = document.getElementById('main-canvas');
    const ctx = canvas.getContext('2d');
    const xInput = document.getElementById('x-coord');
    const yInput = document.getElementById('y-coord');
    const pm25Input = document.getElementById('pm25-value');
    const addSensorBtn = document.getElementById('add-sensor-btn');
    const generateMapBtn = document.getElementById('generate-map-btn');
    const algoSelect = document.getElementById('algo-select');
    const sensorList = document.getElementById('sensor-list');
    const sensorCountSpan = document.getElementById('sensor-count');
    const statusMessage = document.getElementById('status-message');
    const colorbarContainer = document.getElementById('colorbar-container');

    let sensorsData = [];
    const CANVAS_PADDING = 20;
    // 藍(低) -> 青 -> 綠 -> 黃 -> 紅(高)
    const COLOR_RAMP = ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']; 
    let lastBounds = null;  // 記錄上次 generateMap() 使用的 x/y 邊界

    // --- 事件監聽 ---
    addSensorBtn.addEventListener('click', addSensor);
    generateMapBtn.addEventListener('click', generateMap);
    canvas.addEventListener('click', onCanvasClick); 
    // **確保切換算法時，UI 立即更新點數要求**
    algoSelect.addEventListener('change', updateSensorList); 
    // --- 核心功能函式 ---

// app.js (新增/修改部分)

// --- 核心功能函式 ---

// ... (onCanvasClick 之前的程式碼保持不變) ...

function onCanvasClick(e) {
  // 取得滑鼠相對於 canvas 的像素座標
  const rect = canvas.getBoundingClientRect();
  const px = Math.round(e.clientX - rect.left);
  const py = Math.round(e.clientY - rect.top);

  // 只允許在留白框內點擊（避免點到 padding 區或外框）
  const left = CANVAS_PADDING;
  const top  = CANVAS_PADDING;
  const right  = canvas.width  - CANVAS_PADDING;
  const bottom = canvas.height - CANVAS_PADDING;
  if (px < left || px > right || py < top || py > bottom) return;

  // 檢查是否滿足最低點數要求
  const selectedAlgo = algoSelect.value;
  const requiredPoints = (selectedAlgo === 'LINEAR') ? 2 : 3; 
  if (sensorsData.length < requiredPoints) {
      alert(`請先新增至少 ${requiredPoints} 個偵測點以進行內插推估。`);
      return;
  }
  
  // 以目前的座標尺度換算世界座標：
  //   - 若已產生過地圖，使用 lastBounds（與畫面完全一致）
  //   - 否則，用目前資料自動估計（含 20% 外推）
  // 注意：如果 lastBounds 是 null，代表是第一次點擊，getDataBounds(0.2) 會提供一個合理的邊界
  const b = lastBounds || getDataBounds(0.2); 
  const w = canvasToWorld(px, py, b.xMin, b.xMax, b.yMin, b.yMax); // {x, y}

  try {
      const estimatedValue = predictAtPoint(w.x, w.y, selectedAlgo);

      // 使用 alert 顯示推估結果，保留一位小數
      alert(`座標 (${w.x.toFixed(1)}, ${w.y.toFixed(1)}) 的 PM2.5 推估值為: ${estimatedValue.toFixed(1)}`);
      
      // *** 註解或移除原本的 form 填充和 addSensor 呼叫 ***
      /*
      // 要使用者輸入此點的 PM2.5（可直接按取消離開）
      const input = prompt(`在 (x=${w.x.toFixed(1)}, y=${w.y.toFixed(1)}) 輸入 PM2.5 數值：`);
      if (input === null) return;
      
      const val = parseFloat(input);
      if (!isFinite(val) || val < 0) {
        alert('請輸入有效的非負數值');
        return;
      }

      // 把座標與數值灌進原本的輸入框，再呼叫既有的 addSensor()
      xInput.value = w.x.toFixed(1);
      yInput.value = w.y.toFixed(1);
      pm25Input.value = val.toFixed(1);
      addSensor(); // 會自動驗證、更新清單、重畫
      */

  } catch(e) {
      alert(`內插推估失敗: ${e.message}`);
  }
}

// **新的輔助函式：預測特定點的數值**
function predictAtPoint(x0, y0, selectedAlgo) {
    const values = sensorsData.map(s => s.value);
    const xCoords = sensorsData.map(s => s.x);
    const yCoords = sensorsData.map(s => s.y);
    const currentPoints = sensorsData.length;

    // 檢查點是否與任一感測器重疊
    for (let i = 0; i < currentPoints; i++) {
        const dist = Math.hypot(x0 - xCoords[i], y0 - yCoords[i]);
        if (dist < 1e-9) {
            return values[i]; // 重疊點直接回傳感測器數值
        }
    }

    if (selectedAlgo === 'Kriging') {
        // --- Kriging 預測 ---
        // Kriging 需要即時訓練模型，因為 Kriging 內插圖的計算沒有將模型儲存
        const hmax   = maxPairwiseDistance(xCoords, yCoords);
        const range  = 0.7 * hmax;
        const nugget = 1e-6;
        const model  = window.kriging.train(values, xCoords, yCoords, 'gaussian', nugget, range);
        return window.kriging.predict(x0, y0, model);

    } else if (selectedAlgo === 'IDW') {
        // --- IDW 預測 (複製 runIDW 的核心邏輯) ---
        const power = 2;
        let weightedSum = 0;
        let weightTotal = 0;
        for (let k = 0; k < currentPoints; k++) {
            const dist = Math.hypot(x0 - xCoords[k], y0 - yCoords[k]);
            const weight = 1.0 / Math.pow(dist, power);
            weightedSum += values[k] * weight;
            weightTotal += weight;
        }
        if (weightTotal === 0) throw new Error("IDW 無法計算權重。");
        return weightedSum / weightTotal;
        
    } else if (selectedAlgo === 'LINEAR') {
        // --- 線性預測 (分 1D 和 TIN) ---
        if (currentPoints === 2) {
            // 2 點一維線性內插 (Linear 1D) 邏輯
            const p1x = xCoords[0], p1y = yCoords[0], v1 = values[0];
            const p2x = xCoords[1], p2y = yCoords[1], v2 = values[1];
            const dSqTotal = Math.pow(p2x - p1x, 2) + Math.pow(p2y - p1y, 2);
            if (dSqTotal < 1e-9) return (v1 + v2) / 2; // 兩點重疊

            const dotProduct = (x0 - p1x) * (p2x - p1x) + (y0 - p1y) * (p2y - p1y);
            const t = dotProduct / dSqTotal;
            return v1 + t * (v2 - v1);
            
        } else { // currentPoints >= 3, 採用 TIN (Linear TIN) 邏輯
            // 1) 先對散點做 Delaunay 三角化
            const tris = delaunayTriangulate(xCoords, yCoords);
            
            // 2) 找包含 (x0,y0) 的 Delaunay 三角形 (內插)
            for (const tri of tris) {
                const iIdx=tri.i, jIdx=tri.j, kIdx=tri.k;
                const w = baryWeightsABC(x0, y0, xCoords[iIdx], yCoords[iIdx], xCoords[jIdx], yCoords[jIdx], xCoords[kIdx], yCoords[kIdx]);
                if (w && w[0] >= -1e-9 && w[1] >= -1e-9 && w[2] >= -1e-9) { 
                    // 在三角形內（含邊界）
                    return w[0]*values[iIdx] + w[1]*values[jIdx] + w[2]*values[kIdx];
                }
            }
            
            // 3) 凸包外：用最近的 3 個不共線點擬合平面，線性外插
            const idx = Array.from({length:currentPoints}, (_,ii)=>ii).sort((a,b)=>{
                const da=_sqr(xCoords[a]-x0)+_sqr(yCoords[a]-y0), db=_sqr(xCoords[b]-x0)+_sqr(yCoords[b]-y0);
                return da-db;
            });
            for (let a=0; a<idx.length-2; a++){
                for (let b=a+1; b<idx.length-1; b++){
                    for (let c2=b+1; c2<idx.length; c2++){
                        const iIdx=idx[a], jIdx=idx[b], kIdx=idx[c2];
                        if (Math.abs(_orient(xCoords[iIdx],yCoords[iIdx], xCoords[jIdx],yCoords[jIdx], xCoords[kIdx],yCoords[kIdx])) < 1e-12) continue; // 共線跳過
                        const pl = fitPlane3(xCoords[iIdx],yCoords[iIdx],values[iIdx], xCoords[jIdx],yCoords[jIdx],values[jIdx], xCoords[kIdx],yCoords[kIdx],values[kIdx]);
                        return pl.a*x0 + pl.b*y0 + pl.c;
                    }
                }
            }
            // 外插失敗，回傳平均值作為備援
            return values.reduce((sum, v) => sum + v, 0) / currentPoints;
        }
    }
    
    // 如果執行到這裡，應該是邏輯錯誤或演算法無效
    throw new Error(`未知的內插演算法: ${selectedAlgo}`);
}


function addSensor() {
  const x = parseFloat(xInput.value);
  const y = parseFloat(yInput.value);
  const value = parseFloat(pm25Input.value);

  if (isNaN(x) || isNaN(y) || isNaN(value)) {
    alert('請輸入有效的數字座標與 PM 2.5 數值！');
    return;
  }
  if (value < 0) {
    alert('PM 2.5 數值不能為負！');
    return;
  }

  sensorsData.push({ x, y, value });
  updateSensorList(); 

  // 1. 先計算新邊界（含 20% 外推）
  const newB = getDataBounds(0.2);
  
  // 2. 判斷 scale 是否變動，或尚未畫圖
  const changed = !lastBounds ||
    Math.abs(newB.xMin - lastBounds.xMin) > 1e-9 ||
    Math.abs(newB.xMax - lastBounds.xMax) > 1e-9 ||
    Math.abs(newB.yMin - lastBounds.yMin) > 1e-9 ||
    Math.abs(newB.yMax - lastBounds.yMax) > 1e-9;

  // 3. 策略：
  //    - 點數 $\ge 2$ 且邊界變動，或上次畫過圖 (lastBounds != null)：重跑 generateMap
  
  // 檢查是否達到最低運行要求（2點）
  if (sensorsData.length >= 2 && (changed || lastBounds)) {
    // 點數 $\ge 2$：重跑 generateMap
    generateMap(); 
  } else {
    // 點數 $< 2$，或邊界不變：只重畫點與色條即可（更快）
    ctx.clearRect(0, 0, canvas.width, canvas.height); // 清空
    redrawSensorPoints(newB);
    drawColorbar(newB.valMin, newB.valMax);
    lastBounds = newB; // 即使沒畫熱圖，也要更新 lastBounds 以便下次點擊畫布
  }

  // 清空表單
  xInput.value = '';
  yInput.value = '';
  pm25Input.value = '';
  xInput.focus();
}

    
    function updateSensorList() {
        sensorList.innerHTML = '';
        sensorsData.forEach((sensor, index) => {
            const listItem = document.createElement('li');
            listItem.innerHTML = `(${sensor.x.toFixed(1)}, ${sensor.y.toFixed(1)}) -> PM2.5: ${sensor.value.toFixed(1)} <button class="remove-btn" data-index="${index}">移除</button>`;
            sensorList.appendChild(listItem);
        });
        sensorCountSpan.textContent = sensorsData.length;

        // 移除按鈕事件
        document.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const indexToRemove = parseInt(e.target.dataset.index);
                sensorsData.splice(indexToRemove, 1);
                // 重新更新列表並重畫地圖/點
                updateSensorList();
                // 重新評估邊界並重畫
                if (sensorsData.length >= 2) { 
                    generateMap();
                } else {
                    const newB = getDataBounds(0.2);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    redrawSensorPoints(newB);
                    drawColorbar(newB.valMin, newB.valMax);
                    lastBounds = newB;
                }
            });
        });

        // 最終修正 B：確保 UI 狀態與邏輯一致
        const selectedAlgo = algoSelect.value;
        const currentPoints = sensorsData.length;
        
        // Linear 只需要 2 點，IDW/Kriging 需要 3 點
        const requiredPoints = (selectedAlgo === 'LINEAR') ? 2 : 3; 
        
        if (currentPoints >= requiredPoints) {
            generateMapBtn.disabled = false;
            statusMessage.textContent = `準備就緒，可產生分佈圖 (${requiredPoints} 點要求已達成)。`;
        } else {
            generateMapBtn.disabled = true;
            statusMessage.textContent = `請再新增 ${requiredPoints - currentPoints} 個偵測點`;
            
            // 如果點數少於 2，則清空畫布，避免點位繪製錯誤
            if (currentPoints < 2) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                drawColorbar(0, 100);
                lastBounds = null; 
            }
        }
    }

    function redrawSensorPoints(bounds) {
        if (sensorsData.length === 0) return;
        
        // 確保使用當前的邊界
        const { xMin, xMax, yMin, yMax, valMin, valMax } = bounds || lastBounds || getDataBounds(0.2);

        sensorsData.forEach(sensor => {
            const canvasCoords = worldToCanvas(sensor.x, sensor.y, xMin, xMax, yMin, yMax);
            const color = getColorForValue(sensor.value, valMin, valMax);
            
            // 點的背景
            ctx.beginPath();
            ctx.arc(canvasCoords.x, canvasCoords.y, 10, 0, 2 * Math.PI);
            ctx.fillStyle = '#fff'; // 白色背景
            ctx.fill();

            // 點的外框
            ctx.beginPath();
            ctx.arc(canvasCoords.x, canvasCoords.y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.stroke();

            // 數值文字
            ctx.fillStyle = '#000';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(sensor.value.toFixed(1), canvasCoords.x, canvasCoords.y);
        });
    }

    async function generateMap() {
        // 檢查邏輯與 UI 邏輯保持一致
        const selectedAlgo = algoSelect.value;
        const currentPoints = sensorsData.length;
        
        const requiredPoints = (selectedAlgo === 'LINEAR') ? 2 : 3; 
        if (currentPoints < requiredPoints) {
            alert(`選擇 ${selectedAlgo} 演算法需要至少 ${requiredPoints} 個偵測點才能運算。`);
            return;
        }

        // 針對 Kriging 進行全同值檢查
        if (selectedAlgo === 'Kriging') {
            const values = sensorsData.map(s => s.value);
            const allValuesAreSame = values.every(val => val === values[0]);
            if (allValuesAreSame) {
                alert("克里金法計算錯誤：\n\n所有偵測點的數值都相同，無法計算空間變異。\n請至少提供一個不同數值的偵測點。");
                statusMessage.textContent = '計算失敗，請提供不同數值的偵測點。';
                generateMapBtn.disabled = false;
                return;
            }
        }

        statusMessage.textContent = '正在計算中，請稍候...';
        generateMapBtn.disabled = true;
        await new Promise(resolve => setTimeout(resolve, 50));

        const currentBounds = getDataBounds(0.2); 
        const { xMin, xMax, yMin, yMax, valMin, valMax } = currentBounds;

        const values = sensorsData.map(s => s.value);
        const xCoords = sensorsData.map(s => s.x);
        const yCoords = sensorsData.map(s => s.y);

        try {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (selectedAlgo === 'Kriging') {
              runKriging(values, xCoords, yCoords, xMin, xMax, yMin, yMax, valMin, valMax);
            } else if (selectedAlgo === 'IDW') {
              runIDW(values, xCoords, yCoords, xMin, xMax, yMin, yMax, valMin, valMax); 
            } else if (selectedAlgo === 'LINEAR') {
              // 根據點數分派 Linear 運算
              if (currentPoints === 2) {
                  runLinear1D(values, xCoords, yCoords, xMin, xMax, yMin, yMax, valMin, valMax);
              } else { // currentPoints >= 3
                  runLinearTIN(values, xCoords, yCoords, xMin, xMax, yMin, yMax, valMin, valMax);
              }
            }

            redrawSensorPoints(currentBounds); 
            drawColorbar(valMin, valMax);
            statusMessage.textContent = `分佈圖 (${selectedAlgo}) 已產生！`;
            lastBounds = currentBounds; // 記下本次 scale
        } catch(e) {
            statusMessage.textContent = `計算錯誤: ${e.message}`;
            alert(`計算時發生錯誤: ${e.message}\n請確認資料輸入是否正確或嘗試重新載入頁面。`);
        } finally {
            generateMapBtn.disabled = false;
        }
    }
    
// **新增函式：2 點一維線性內插 (Linear 1D)**
function runLinear1D(values, x, y, xMin, xMax, yMin, yMax, valMin, valMax) {
    const p1x = x[0], p1y = y[0], v1 = values[0];
    const p2x = x[1], p2y = y[1], v2 = values[1];
    
    // 兩點間的總距離平方
    const dSqTotal = Math.pow(p2x - p1x, 2) + Math.pow(p2y - p1y, 2);

    if (dSqTotal < 1e-9) { // 兩點重疊或極近
        // 畫布填滿單色（平均值）
        const avgVal = (v1 + v2) / 2;
        ctx.fillStyle = getColorForValue(avgVal, valMin, valMax);
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        return;
    }
    
    const usableW = canvas.width  - 2 * CANVAS_PADDING;
    const usableH = canvas.height - 2 * CANVAS_PADDING;

    for (let j = 0; j < usableH; j++) { 
        const canvasY = CANVAS_PADDING + j; 
        for (let i = 0; i < usableW; i++) { 
            const canvasX = CANVAS_PADDING + i; 
            
            const worldCoords = canvasToWorld(canvasX, canvasY, xMin, xMax, yMin, yMax);
            const px = worldCoords.x;
            const py = worldCoords.y;

            // 計算投影係數 t
            // t = [(P-P1) . (P2-P1)] / |P2-P1|^2
            const dotProduct = (px - p1x) * (p2x - p1x) + (py - p1y) * (p2y - p1y);
            const t = dotProduct / dSqTotal; // t in R. [0, 1] 是線段內部。

            // 線性內插/外插 Z 值
            let val = v1 + t * (v2 - v1);

            // Clamping：將結果限制在 [valMin, valMax] 之間
            val = Math.max(valMin, Math.min(valMax, val));

            // 上色
            ctx.fillStyle = getColorForValue(val, valMin, valMax);
            ctx.fillRect(canvasX, canvasY, 1, 1);
        }
    }
}


// **修改函式名稱：runLinear -> runLinearTIN (處理 $\ge 3$ 點)**
function runLinearTIN(values, x, y, xMin, xMax, yMin, yMax, valMin, valMax) {
    // 1) 先對散點做 Delaunay 三角化
    const tris = delaunayTriangulate(x, y);

    // 2) 為每個三角形建 AABB，加速點內測試
    const T = tris.map(tr => {
        const i=tr.i,j=tr.j,k=tr.k;
        const xmin = Math.min(x[i],x[j],x[k]), xmax=Math.max(x[i],x[j],x[k]);
        const ymin = Math.min(y[i],y[j],y[k]), ymax=Math.max(y[i],y[j],y[k]);
        return {i,j,k,xmin,xmax,ymin,ymax};
    });

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const usableW = canvas.width  - 2 * CANVAS_PADDING;
    const usableH = canvas.height - 2 * CANVAS_PADDING;
    
    // 計算平均值作為外插失敗時的備援值
    const valAvg = values.reduce((sum, v) => sum + v, 0) / values.length;

    // 3) 遍歷畫布內每個像素 (從左上角開始)
    for (let j = 0; j < usableH; j++) { 
        const canvasY = CANVAS_PADDING + j; 
        
        for (let i = 0; i < usableW; i++) { 
            const canvasX = CANVAS_PADDING + i; 
            
            // 3a) 像素轉世界座標
            const worldCoords = canvasToWorld(canvasX, canvasY, xMin, xMax, yMin, yMax);
            const px = worldCoords.x;
            const py = worldCoords.y;

            // 3b) 先找包含 (px,py) 的 Delaunay 三角形 (內插)
            let val = null;
            for (let t = 0; t < T.length; t++) {
                const tri = T[t];
                // AABB 篩選 (可選，但有助於效能)
                if (px < tri.xmin || px > tri.xmax || py < tri.ymin || py > tri.ymax) continue; 

                const iIdx=tri.i, jIdx=tri.j, kIdx=tri.k;
                const w = baryWeightsABC(px,py, x[iIdx],y[iIdx], x[jIdx],y[jIdx], x[kIdx],y[kIdx]);
                if (!w) continue;
                if (w[0] >= -1e-9 && w[1] >= -1e-9 && w[2] >= -1e-9) { // 在三角形內（含邊界）
                    val = w[0]*values[iIdx] + w[1]*values[jIdx] + w[2]*values[kIdx];
                    break;
                }
            }

            // 3c) 凸包外：用最近的 3 個不共線點擬合平面，線性外插
            if (val === null) {
                const idx = Array.from({length:x.length}, (_,ii)=>ii).sort((a,b)=>{
                    const da=_sqr(x[a]-px)+_sqr(y[a]-py), db=_sqr(x[b]-px)+_sqr(y[b]-py);
                    return da-db;
                });
                let got=false;
                for (let a=0; a<idx.length-2 && !got; a++){
                    for (let b=a+1; b<idx.length-1 && !got; b++){
                        for (let c2=b+1; c2<idx.length && !got; c2++){
                            const iIdx=idx[a], jIdx=idx[b], kIdx=idx[c2];
                            if (Math.abs(_orient(x[iIdx],y[iIdx], x[jIdx],y[jIdx], x[kIdx],y[kIdx])) < 1e-12) continue; // 共線跳過
                            const pl = fitPlane3(x[iIdx],y[iIdx],values[iIdx], x[jIdx],y[jIdx],values[jIdx], x[kIdx],y[kIdx],values[kIdx]);
                            val = pl.a*px + pl.b*py + pl.c;
                            got = true;
                        }
                    }
                }
                if (val === null) val = valAvg; 
            }

            // 外插結果必須被限制在色帶範圍內 (Clamping)
            val = Math.max(valMin, Math.min(valMax, val));

            // 3d) 上色
            ctx.fillStyle = getColorForValue(val, valMin, valMax);
            ctx.fillRect(canvasX, canvasY, 1, 1); // 畫一個 1x1 像素的點
        }
    }
}


// IDW (保持不變)
function runIDW(values, x, y, xMin, xMax, yMin, yMax, valMin, valMax) {
        // IDW 在數學上允許 2 點，但地學應用中通常要求 3 點以上來提供穩定的方向性，故保留 3 點限制
        if (values.length < 3) throw new Error("IDW 演算法需要至少 3 個偵測點才能提供穩定的空間內插。");
        
        const power = 2; // 預設權重次方

        const usableW = canvas.width  - 2 * CANVAS_PADDING;
        const usableH = canvas.height - 2 * CANVAS_PADDING;

        for (let i = 0; i < usableW; i++) {
            for (let j = 0; j < usableH; j++) {
                const canvasX = CANVAS_PADDING + i;
                const canvasY = CANVAS_PADDING + j;
                
                const worldCoords = canvasToWorld(canvasX, canvasY, xMin, xMax, yMin, yMax);
                
                let weightedSum = 0;
                let weightTotal = 0;
                let isAtSensor = false;

                for (let k = 0; k < sensorsData.length; k++) {
                    const dist = Math.hypot(worldCoords.x - x[k], worldCoords.y - y[k]);
                    
                    if (dist < 1e-9) { 
                        weightedSum = values[k];
                        weightTotal = 1;
                        isAtSensor = true;
                        break;
                    }
                    const weight = 1.0 / Math.pow(dist, power);
                    weightedSum += values[k] * weight;
                    weightTotal += weight;
                }

                const estimatedValue = isAtSensor ? weightedSum : (weightTotal === 0 ? valMin : weightedSum / weightTotal);
                
                ctx.fillStyle = getColorForValue(estimatedValue, valMin, valMax);
                ctx.fillRect(canvasX, canvasY, 1, 1);
            }
        }
    }


// Kriging (保持不變)
function runKriging(values, x, y, xMin, xMax, yMin, yMax, valMin, valMax) {
  if (typeof window.kriging === 'undefined') {
    throw new Error("kriging.js 函式庫未載入。");
  }

  // ——穩定參數——
  const hmax   = maxPairwiseDistance(x, y);
  const range  = 0.7 * hmax;    // 建議 0.6~0.8*hmax
  const nugget = 1e-6;          // 幾乎精確插值，避免 bullseye
  const model  = window.kriging.train(values, x, y, 'gaussian', nugget, range);

  // ——產格網 (至少 220x220)——
  const nx = 220, ny = 220;
  // 計算網格單元的世界座標大小，讓它至少能包含整個繪圖區
  const cellWidth  = (xMax - xMin) / (nx - 1);
  const cellHeight = (yMax - yMin) / (ny - 1);
  const cell = Math.max(cellWidth, cellHeight);
  // 定義克里金的插值範圍
  const polygon = [[xMin, yMin], [xMax, yMin], [xMax, yMax], [xMin, yMax]];
  const gridObj = window.kriging.grid([polygon], model, cell);
  const grid = gridObj.grid;
  const nxg = grid[0].length, nyg = grid.length;

  // ——用「自己的色帶＋固定量尺」畫到畫布——
  const usableW = canvas.width  - 2 * CANVAS_PADDING;
  const usableH = canvas.height - 2 * CANVAS_PADDING;
  const dx = usableW / nxg;
  const dy = usableH / nyg;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  for (let r = 0; r < nyg; r++) {
    for (let c = 0; c < nxg; c++) {
      const v = grid[r][c];
      
      // 注意：kriging.grid 的 r=0 是 yMin（底部），r=nyg-1 是 yMax（頂部）
      // 畫布的 y 座標 r=0 是頂部，r=H 是底部
      // 故：畫布 Y 座標應是 CANVAS_PADDING + (nyg - 1 - r) * dy
      const ypix = CANVAS_PADDING + (nyg - 1 - r) * dy; 

      ctx.fillStyle = getColorForValue(v, valMin, valMax);
      ctx.fillRect(CANVAS_PADDING + c * dx,
                   ypix,
                   dx, dy);
    }
  }
}


// -------------------------------------------------------------------
// 輔助工具函式
// -------------------------------------------------------------------

// 1) 補一個工具：資料點間的最大距離
function maxPairwiseDistance(xs, ys) {
  let hmax = 0;
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      const dx = xs[i] - xs[j], dy = ys[i] - ys[j];
      const h = Math.hypot(dx, dy);
      if (h > hmax) hmax = h;
    }
  }
  return hmax || 1;
}


// ==== 線性差補：正確版輔助函式（Delaunay + 重心座標） ====

// 幾何小工具
function _sqr(x){ return x*x; }
function _orient(ax,ay,bx,by,cx,cy){ return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax); } // >0: CCW
function _circumcircle(ax,ay,bx,by,cx,cy){
  const d = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by));
  if (Math.abs(d) < 1e-12) return {valid:false};
  const ax2=ax*ax+ay*ay, bx2=bx*bx+by*by, cx2=cx*cx+cy*cy;
  const ux = (ax2*(by-cy)+bx2*(cy-ay)+cx2*(ay-by))/d;
  const uy = (ax2*(cx-bx)+bx2*(ax-cx)+cx2*(bx-ax))/d;
  const r2 = _sqr(ux-ax)+_sqr(uy-ay);
  return {ux,uy,r2,valid:true};
}

// Bowyer–Watson Delaunay 三角化（回傳 [{i,j,k}]）
function delaunayTriangulate(xs, ys){
  const n = xs.length;
  const pts = []; for (let i=0;i<n;i++) pts.push({i, x:xs[i], y:ys[i]});

  // 超大三角形（包住所有點）
  let xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
  const span = Math.max(xmax-xmin, ymax-ymin) || 1;
  const cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
  const stA={i:n,   x:cx-2*span, y:cy-3*span};
  const stB={i:n+1, x:cx,        y:cy+4*span};
  const stC={i:n+2, x:cx+2*span, y:cy-3*span};
  let tris=[{i:stA.i,j:stB.i,k:stC.i}];
  const all=pts.concat([stA,stB,stC]);

  for (const p of pts){
    const badIdx=[]; const ccCache=[];
    for (let t=0;t<tris.length;t++){
      const tri=tris[t], A=all[tri.i], B=all[tri.j], C=all[tri.k];
      const cc=_circumcircle(A.x,A.y,B.x,B.y,C.x,C.y); ccCache[t]=cc;
      if (!cc.valid) continue;
      const d2=_sqr(p.x-cc.ux)+_sqr(p.y-cc.uy);
      if (d2 <= cc.r2*(1+1e-12)) badIdx.push(t);
    }
    const edgeCount=new Map();
    function addEdge(a,b){
      const key=a<b?`${a}_${b}`:`${b}_${a}`;
      edgeCount.set(key,(edgeCount.get(key)||0)+1);
    }
    for (const bi of badIdx){
      const tri=tris[bi]; addEdge(tri.i,tri.j); addEdge(tri.j,tri.k); addEdge(tri.k,tri.i);
    }
    tris=tris.filter((_,idx)=>!badIdx.includes(idx));
    for (const [key,cnt] of edgeCount.entries()){
      if (cnt===1){
        const [sa,sb]=key.split('_'); const a=parseInt(sa), b=parseInt(sb);
        tris.push({i:a,j:b,k:p.i});
      }
    }
  }
  // 移除含超大三角形頂點的三角形
  return tris.filter(tr=> tr.i< n && tr.j< n && tr.k< n);
}

// 重心座標（回傳 [wA,wB,wC]；任一 <0 代表在外面）
function baryWeightsABC(px,py, ax,ay,bx,by,cx,cy){
  const den = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
  if (Math.abs(den) < 1e-12) return null;
  const u = ((px-ax)*(cy-ay) - (py-ay)*(cx-ax)) / den; // 對應 B
  const v = ((bx-ax)*(py-ay) - (px-ax)*(by-ay)) / den; // 對應 C
  const w = 1 - u - v;                                 // 對應 A
  return [w,u,v];
}

// 三點擬合平面 z = a x + b y + c（Cramer's rule）
function fitPlane3(ax,ay,az, bx,by,bz, cx,cy,cz){
  function det3(M){
    return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
         - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
         + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
  }
  const A=[[ax,ay,1],[bx,by,1],[cx,cy,1]];
  const b=[az,bz,cz];
  const D = det3(A)||1e-12;
  const A1=[[b[0],ay,1],[b[1],by,1],[b[2],cy,1]];
  const A2=[[ax,b[0],1],[bx,b[1],1],[cx,b[2],1]];
  const A3=[[ax,ay,b[0]],[bx,by,b[1]],[cx,cy,b[2]]];
  return { a:det3(A1)/D, b:det3(A2)/D, c:det3(A3)/D };
}


    function getDataBounds(paddingFactor = 0) {
        if (sensorsData.length === 0) {
            return { xMin: 0, xMax: 100, yMin: 0, yMax: 100, valMin: 0, valMax: 100 };
        }
        const x = sensorsData.map(s => s.x);
        const y = sensorsData.map(s => s.y);
        const v = sensorsData.map(s => s.value);
        
        let xMin = Math.min(...x);
        let xMax = Math.max(...x);
        let yMin = Math.min(...y);
        let yMax = Math.max(...y);
        let valMin = Math.min(...v);
        let valMax = Math.max(...v);
        
        // 確保範圍不為零
        if (xMin === xMax) { xMin -= 5; xMax += 5; }
        if (yMin === yMax) { yMin -= 5; yMax += 5; }
        if (valMin === valMax) { valMin = valMin - 1 < 0 ? 0 : valMin - 1; valMax += 1; } // 確保有色帶

        const xRange = xMax - xMin;
        const yRange = yMax - yMin;

        xMin -= xRange * paddingFactor;
        xMax += xRange * paddingFactor;
        yMin -= yRange * paddingFactor;
        yMax += yRange * paddingFactor;

        return { xMin, xMax, yMin, yMax, valMin, valMax };
    }

function worldToCanvas(x, y, xMin, xMax, yMin, yMax) {
  const xRange = xMax - xMin;
  const yRange = yMax - yMin;
  const usableW = canvas.width  - 2 * CANVAS_PADDING;
  const usableH = canvas.height - 2 * CANVAS_PADDING;

  const cx = CANVAS_PADDING + (x - xMin) / xRange * usableW;
  // 畫布 Y 軸是「頂部」為 0，所以要用 usableH 減去向上為正的距離
  const cy = CANVAS_PADDING + usableH - (y - yMin) / yRange * usableH; 
  return { x: cx, y: cy };
}

    
    function canvasToWorld(canvasX, canvasY, xMin, xMax, yMin, yMax) {
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const usableWidth = canvas.width - CANVAS_PADDING * 2;
        const usableHeight = canvas.height - CANVAS_PADDING * 2;

        const x = ((canvasX - CANVAS_PADDING) / usableWidth) * xRange + xMin;
        
        // 畫布 Y 軸是「頂部」為 0，所以需要反轉
        const normalizedY = (canvasY - CANVAS_PADDING) / usableHeight; // [0, 1] 頂部到底部
        // 1 - normalizedY = [1, 0] 底部到頂部
        const y = (1 - normalizedY) * yRange + yMin; 
        
        return { x, y };
    }

    function getColorForValue(value, min, max) {
        // 如果 max === min，給定中間色
        if (min === max) {
            return COLOR_RAMP[Math.floor(COLOR_RAMP.length / 2)]; 
        }

        const ratio = (value - min) / (max - min);
        const clampedRatio = Math.max(0, Math.min(1, ratio)); 

        // 將 [0, 1] 的比例映射到 COLOR_RAMP 的索引
        const index = clampedRatio * (COLOR_RAMP.length - 1);
        const i0 = Math.floor(index);
        const i1 = Math.min(i0 + 1, COLOR_RAMP.length - 1);
        
        const t = index - i0; // 內插比例

        const c0 = hexToRgb(COLOR_RAMP[i0]);
        const c1 = hexToRgb(COLOR_RAMP[i1]);
        
        // 線性內插 RGB
        const r = Math.round(c0.r * (1 - t) + c1.r * t);
        const g = Math.round(c0.g * (1 - t) + c1.g * t);
        const b = Math.round(c0.b * (1 - t) + c1.b * t);
        return `rgb(${r},${g},${b})`;
    }
    
    function hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    function drawColorbar(min, max) {
        let barCanvas = document.getElementById('colorbar-canvas');
        if (!barCanvas) {
            barCanvas = document.createElement('canvas');
            barCanvas.id = 'colorbar-canvas';
            colorbarContainer.appendChild(barCanvas);
        }
        
        barCanvas.width = colorbarContainer.clientWidth;
        barCanvas.height = colorbarContainer.clientHeight;
        
        const barCtx = barCanvas.getContext('2d');
        barCtx.clearRect(0, 0, barCanvas.width, barCanvas.height);

        const gradient = barCtx.createLinearGradient(0, 0, barCanvas.width, 0);
        
        COLOR_RAMP.forEach((color, index) => {
            gradient.addColorStop(index / (COLOR_RAMP.length - 1), color);
        });

        const barHeight = barCanvas.height - 20; // 預留 20px 畫文字
        barCtx.fillStyle = gradient;
        barCtx.fillRect(0, 0, barCanvas.width, barHeight);
        
        // 畫邊界
        barCtx.strokeStyle = "#333";
        barCtx.lineWidth = 1;
        barCtx.strokeRect(0, 0, barCanvas.width, barHeight);

        // 畫數值
        barCtx.fillStyle = '#000';
        barCtx.font = '12px Arial';
        
        const textY = barCanvas.height - 5;
        
        barCtx.textAlign = 'start';
        barCtx.fillText(min.toFixed(1), 5, textY);
        
        barCtx.textAlign = 'center';
        barCtx.fillText(((min + max) / 2).toFixed(1), barCanvas.width / 2, textY);
        
        barCtx.textAlign = 'end';
        barCtx.fillText(max.toFixed(1), barCanvas.width - 5, textY);
    }
    
// -------------------------------------------------------------------
// Kriging, IDW, LinearTIN, Linear1D 函式內容 (保持不變)
// -------------------------------------------------------------------
// (為節省篇幅，此處省略了前幾次提供的 runKriging, runIDW, runLinearTIN, runLinear1D 及其輔助函式)
// -------------------------------------------------------------------

// **請確保你的 app.js 包含以下所有函式：**
// - maxPairwiseDistance
// - _sqr, _orient, _circumcircle
// - delaunayTriangulate, baryWeightsABC, fitPlane3
// - runKriging, runIDW, runLinearTIN (即原 runLinear), runLinear1D (2點線性)


    // 網頁初始化
    updateSensorList();
    ctx.clearRect(0, 0, canvas.width, canvas.height); 
    drawColorbar(0, 100);
});
