/*
 * kriging.js - Ordinary Kriging in 2D (fixed + compatible)
 * 重點：
 * 1) 以「共變異 C(h)」建立系統矩陣；h=0 對角加 nugget
 * 2) 解擴充系統 [C 1; 1^T 0][w;λ]=[c;1]；predict 回傳數值（相容舊介面）
 * 3) 修正 matrix.multiply 的「向量判斷」條件
 */

var kriging = (function () {
  var kriging = {};

  // ---------- Matrix utilities ----------
  var matrix = {
    add: function (A, B) {
      var i, j, C = new Array(A.length);
      for (i = 0; i < A.length; i++) {
        C[i] = new Array(A[i].length);
        for (j = 0; j < A[i].length; j++) C[i][j] = A[i][j] + B[i][j];
      }
      return C;
    },

    multiply: function (A, B) {
      var i, j, k, C;
      if (typeof B.length === "undefined") { // scalar
        C = new Array(A.length);
        for (i = 0; i < A.length; i++) C[i] = A[i] * B;
      } 
      // 這裡要用 B[0].length === undefined 來判斷「向量」
      else if (typeof B[0].length === "undefined") { // vector
        C = new Array(A.length);
        for (i = 0; i < A.length; i++) {
          C[i] = 0;
          for (j = 0; j < A[0].length; j++) C[i] += A[i][j] * B[j];
        }
      } else { // matrix
        C = new Array(A.length);
        for (i = 0; i < A.length; i++) {
          C[i] = new Array(B[0].length);
          for (j = 0; j < B[0].length; j++) {
            C[i][j] = 0;
            for (k = 0; k < A[0].length; k++) C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      return C;
    },

    dot: function (a, b) {
      var n = 0, lim = Math.min(a.length, b.length);
      for (var i = 0; i < lim; i++) n += a[i] * b[i];
      return n;
    },

    transpose: function (A) {
      var i, j, C = new Array(A[0].length);
      for (i = 0; i < A[0].length; i++) {
        C[i] = new Array(A.length);
        for (j = 0; j < A.length; j++) C[i][j] = A[j][i];
      }
      return C;
    },

    invert: function (A) {
      var i, j, k;
      var n = A.length;
      var I = new Array(n), C = new Array(n);
      for (i = 0; i < n; i++) {
        I[i] = new Array(n);
        C[i] = new Array(n);
        for (j = 0; j < n; j++) {
          I[i][j] = (i === j) ? 1 : 0;
          C[i][j] = A[i][j];
        }
      }
      for (i = 0; i < n; i++) {
        var e = C[i][i];
        if (e === 0) {
          for (j = i + 1; j < n; j++) {
            if (C[j][i] !== 0) {
              for (k = 0; k < n; k++) {
                var tmp = C[i][k]; C[i][k] = C[j][k]; C[j][k] = tmp;
                tmp = I[i][k]; I[i][k] = I[j][k]; I[j][k] = tmp;
              }
              break;
            }
          }
          e = C[i][i];
          if (e === 0) return; // singular
        }
        for (j = 0; j < n; j++) { C[i][j] /= e; I[i][j] /= e; }
        for (j = 0; j < n; j++) if (i !== j) {
          e = C[j][i];
          for (k = 0; k < n; k++) { C[j][k] -= e * C[i][k]; I[j][k] -= e * I[i][k]; }
        }
      }
      return I;
    }
  };

  // ---------- Variogram shapes m(h) ----------
  var shape = {
    gaussian: function (h, range) {
      var r = h / (range || 1e-12);
      return 1 - Math.exp(-(r * r));
    },
    exponential: function (h, range) {
      var r = h / (range || 1e-12);
      return 1 - Math.exp(-r);
    },
    spherical: function (h, range) {
      if (h >= range) return 1;
      var r = h / (range || 1e-12);
      return 1.5 * r - 0.5 * r * r * r;
    }
  };

  function dist(x1, y1, x2, y2) {
    var dx = x1 - x2, dy = y1 - y2;
    return Math.sqrt(dx * dx + dy * dy);
  }
  function mean(a){ var s=0; for (var i=0;i<a.length;i++) s+=a[i]; return s/Math.max(1,a.length); }
  function variance(a, mu){ if(a.length<2) return 1e-12; var s=0,d; for(var i=0;i<a.length;i++){ d=a[i]-mu; s+=d*d;} return s/(a.length-1); }

  // ---------- Train (Ordinary Kriging) ----------
  // train(t,x,y, model, sigma2, alpha): sigma2=nugget, alpha=range
  kriging.train = function (t, x, y, model, sigma2, alpha) {
    var n = t.length;
    if (x.length!==n || y.length!==n) throw new Error("x,y,t 長度需一致");

    var mod = (shape[model] ? model : "spherical");
    var nugget = Math.max(0, +sigma2 || 0);
    var range  = Math.max(1e-12, +alpha || 1);

    var mu = mean(t);
    var sill = Math.max(nugget + 1e-12, variance(t, mu) + nugget);
    var psill = Math.max(1e-12, sill - nugget);
    var mfun = shape[mod];

    function C_of(h){
      if (h===0) return psill + nugget;
      return psill * (1 - mfun(h, range));
    }

    // C 矩陣
    var C = new Array(n);
    for (var i=0;i<n;i++){
      C[i] = new Array(n);
      for (var j=0;j<n;j++){
        C[i][j] = C_of(dist(x[i],y[i],x[j],y[j]));
      }
      C[i][i] += (psill+nugget)*1e-10; // 小正則
    }

    // 擴充矩陣 A_aug = [C 1; 1^T 0]
    var A_aug = new Array(n+1);
    for (var r=0;r<n+1;r++){
      A_aug[r] = new Array(n+1);
      for (var c=0;c<n+1;c++){
        if (r<n && c<n) A_aug[r][c] = C[r][c];
        else if (r===n && c===n) A_aug[r][c] = 0;
        else if (r===n || c===n) A_aug[r][c] = 1;
      }
    }

    var A_inv = matrix.invert(A_aug);
    if (!A_inv) throw new Error("Kriging 矩陣不可逆；請調整 range/nugget 或檢查點位配置。");

    return {
      t: t.slice(0), x: x.slice(0), y: y.slice(0),
      model: mod, nugget: nugget, range: range, sill: sill, psill: psill,
      A_inv: A_inv,
      cov: C_of
    };
  };

  // ---------- Predict ----------
  // 相容舊版：回傳數值
  kriging.predict = function (x0, y0, mdl) {
    var n = mdl.t.length;
    var rhs = new Array(n+1);
    for (var i=0;i<n;i++){
      rhs[i] = mdl.cov( Math.sqrt((x0 - mdl.x[i])*(x0 - mdl.x[i]) + (y0 - mdl.y[i])*(y0 - mdl.y[i])) );
    }
    rhs[n] = 1;

    var sol = matrix.multiply(mdl.A_inv, rhs); // [w; λ]
    var w = sol.slice(0, n);
    return matrix.dot(w, mdl.t); // ŷ
  };

  // 若需要同時取變異
  kriging.predictDetail = function (x0, y0, mdl) {
    var n = mdl.t.length;
    var rhs = new Array(n+1);
    for (var i=0;i<n;i++){
      rhs[i] = mdl.cov( Math.sqrt((x0 - mdl.x[i])*(x0 - mdl.x[i]) + (y0 - mdl.y[i])*(y0 - mdl.y[i])) );
    }
    rhs[n] = 1;
    var sol = matrix.multiply(mdl.A_inv, rhs); // [w; λ]
    var w = sol.slice(0, n);
    var yhat = matrix.dot(w, mdl.t);
    var bdotw = matrix.dot(w, rhs.slice(0,n));
    var lambda = sol[n];
    var kvar = (mdl.psill + mdl.nugget) - bdotw - lambda;
    return { value: yhat, variance: Math.max(0, kvar) };
  };

  // ---------- Grid（矩形網格） ----------
  kriging.grid = function (polygons, mdl, width) {
    var i, n = polygons[0].length;
    var xlim = [polygons[0][0][0], polygons[0][0][0]];
    var ylim = [polygons[0][0][1], polygons[0][0][1]];
    for (i = 0; i < n; i++) {
      if (polygons[0][i][0] < xlim[0]) xlim[0] = polygons[0][i][0];
      if (polygons[0][i][0] > xlim[1]) xlim[1] = polygons[0][i][0];
      if (polygons[0][i][1] < ylim[0]) ylim[0] = polygons[0][i][1];
      if (polygons[0][i][1] > ylim[1]) ylim[1] = polygons[0][i][1];
    }
    var nx = Math.max(2, Math.ceil((xlim[1]-xlim[0]) / width) + 1);
    var ny = Math.max(2, Math.ceil((ylim[1]-ylim[0]) / width) + 1);

    var grid = new Array(ny);
    for (var r = 0; r < ny; r++) {
      grid[r] = new Array(nx);
      // y 座標從下到上 (r=0 應對應 ymin)
      var py = ylim[0] + r * width;
      for (var c = 0; c < nx; c++) {
        var px = xlim[0] + c * width;
        grid[r][c] = kriging.predict(px, py, mdl); // 回傳數值
      }
    }
    return { grid: grid, xlim: xlim, ylim: ylim, width: width };
  };

  // ---------- Plot (不再使用此內建繪圖，改用 app.js 的統一繪圖邏輯) ----------
  kriging.plot = function (canvas, grid, xlim, ylim, colors, padding) {
    // 註解或移除內建繪圖邏輯
    // 讓 app.js 自行處理繪圖，以確保與其他演算法的顏色與座標一致性
    console.warn("kriging.js: 內建 plot 函式被忽略，由 app.js 統一處理繪圖。");
  };

// IIFE 結尾：把 kriging 物件丟回去並關掉自我呼叫
return kriging;
})();

if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = kriging;
}