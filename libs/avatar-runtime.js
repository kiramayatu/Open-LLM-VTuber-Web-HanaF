/**
 * Open-LLM-VTuber Multi-Format Avatar Runtime & In-App Uploader (Parts 2, 3, 4, 5)
 *
 * Implements:
 * - [Part 2.3] In-App Avatar Uploader & Model Switcher UI Helpers
 * - [Part 3.1] Generic `IAvatarController` Lifecycle Interface & `AvatarRuntimeManager`
 * - [Part 3.2 & 3.3] 3D VRM (`.vrm`) & glTF (`.glb`/`.gltf`) Engine (Three.js + `@pixiv/three-vrm` + Offline WebGL2 GLB/VRM Fallback Renderer)
 * - [Part 4.1] 2D PNGTuber Reactive Multi-State Sprite Adapter
 * - [Part 4.2] Legacy Live2D Cubism 2.1 (`.model.json`) Adapter (`live2d.min.js`)
 * - [Part 5.1, 5.2, 5.3] Unified Cross-Format Emotion Engine, Procedural Idle/Breathing/Blinking/Gaze, and Real-Time Audio Lip-Sync + Vowel Viseme Estimator (`a`, `i`, `u`, `e`, `o`)
 */
(function () {
  "use strict";

  // ============================================================================
  // Part 5.1: Unified Cross-Format Emotion Mapping Table
  // ============================================================================
  const CANONICAL_EMOTIONS = {
    neutral: {
      vrm1: { neutral: 1.0 },
      vrm0: { Neutral: 1.0 },
      cubism2: "f01",
      pngtuber: "neutral",
      mouthCurve: 0.0,
      browOffset: 0.0,
    },
    joy: {
      vrm1: { happy: 1.0 },
      vrm0: { Joy: 1.0, Fun: 0.5 },
      cubism2: "f02",
      pngtuber: "joy",
      mouthCurve: 1.0,
      browOffset: 0.3,
    },
    anger: {
      vrm1: { angry: 1.0 },
      vrm0: { Angry: 1.0 },
      cubism2: "f03",
      pngtuber: "anger",
      mouthCurve: -0.6,
      browOffset: -0.8,
    },
    sadness: {
      vrm1: { sad: 1.0 },
      vrm0: { Sorrow: 1.0 },
      cubism2: "f04",
      pngtuber: "sadness",
      mouthCurve: -0.8,
      browOffset: -0.4,
    },
    surprise: {
      vrm1: { surprised: 1.0 },
      vrm0: { Surprised: 1.0, Fun: 0.4 },
      cubism2: "f05",
      pngtuber: "surprise",
      mouthCurve: 0.2,
      browOffset: 0.9,
    },
    fear: {
      vrm1: { surprised: 0.6, sad: 0.5 },
      vrm0: { Sorrow: 0.6, Surprised: 0.5 },
      cubism2: "f04",
      pngtuber: "fear",
      mouthCurve: -0.5,
      browOffset: 0.5,
    },
    disgust: {
      vrm1: { angry: 0.7, sad: 0.3 },
      vrm0: { Angry: 0.7, Sorrow: 0.3 },
      cubism2: "f03",
      pngtuber: "disgust",
      mouthCurve: -0.7,
      browOffset: -0.6,
    },
    smirk: {
      vrm1: { relaxed: 0.8, happy: 0.4 },
      vrm0: { Fun: 0.9 },
      cubism2: "f02",
      pngtuber: "smirk",
      mouthCurve: 0.6,
      browOffset: 0.2,
    },
  };

  const INDEX_TO_EMOTION = {
    0: "neutral",
    1: "sadness",
    2: "anger",
    3: "joy",
    4: "surprise",
  };

  function resolveCanonicalEmotion(emotionOrIndex, modelInfo) {
    if (typeof emotionOrIndex === "string") {
      const lower = emotionOrIndex.trim().toLowerCase();
      if (CANONICAL_EMOTIONS[lower]) return lower;
      if (lower.includes("happy") || lower.includes("smile") || lower.includes("joy")) return "joy";
      if (lower.includes("ang")) return "anger";
      if (lower.includes("sad") || lower.includes("sorrow") || lower.includes("cry")) return "sadness";
      if (lower.includes("surp") || lower.includes("shock")) return "surprise";
      if (lower.includes("fear") || lower.includes("scare")) return "fear";
      if (lower.includes("disgust")) return "disgust";
      if (lower.includes("smirk") || lower.includes("relax") || lower.includes("fun")) return "smirk";
    }
    if (typeof emotionOrIndex === "number" && modelInfo && modelInfo.emotionMap) {
      for (const [k, v] of Object.entries(modelInfo.emotionMap)) {
        if (v === emotionOrIndex && CANONICAL_EMOTIONS[k.toLowerCase()]) {
          return k.toLowerCase();
        }
      }
    }
    return INDEX_TO_EMOTION[emotionOrIndex] || "neutral";
  }

  // ============================================================================
  // Part 5.2: Real-Time Audio Lip-Sync & Formant Vowel Estimator (a, i, u, e, o)
  // ============================================================================
  class AudioLipSyncAnalyzer {
    constructor() {
      this.ctx = null;
      this.analyser = null;
      this.sourceMap = new WeakMap();
      this.timeData = new Uint8Array(512);
      this.freqData = new Uint8Array(256);
      this.activeAudio = null;
      this.rms = 0;
      this.visemes = { a: 0, i: 0, u: 0, e: 0, o: 0 };
      this._fallbackPhase = 0;
    }

    attachAudio(audioEl) {
      this.activeAudio = audioEl;
      if (!audioEl) {
        this.reset();
        return;
      }
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        if (!this.ctx) {
          this.ctx = new AudioCtx();
        }
        if (this.ctx.state === "suspended") {
          this.ctx.resume().catch(() => {});
        }
        if (!this.analyser) {
          this.analyser = this.ctx.createAnalyser();
          this.analyser.fftSize = 512;
          this.analyser.smoothingTimeConstant = 0.65;
          this.timeData = new Uint8Array(this.analyser.fftSize);
          this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
        }
        let source = this.sourceMap.get(audioEl);
        if (!source) {
          source = this.ctx.createMediaElementSource(audioEl);
          source.connect(this.analyser);
          this.analyser.connect(this.ctx.destination);
          this.sourceMap.set(audioEl, source);
        }
      } catch (err) {
        // Fallback if MediaElementSource already attached or CORS restricted
        console.debug("[AudioLipSyncAnalyzer] Using envelope fallback:", err);
      }
    }

    reset() {
      this.activeAudio = null;
      this.rms = 0;
      this.visemes = { a: 0, i: 0, u: 0, e: 0, o: 0 };
    }

    sample(dt) {
      if (!this.activeAudio || this.activeAudio.paused || this.activeAudio.ended) {
        this.rms = Math.max(0, this.rms - dt * 6.0);
        for (const k of Object.keys(this.visemes)) {
          this.visemes[k] = Math.max(0, this.visemes[k] - dt * 6.0);
        }
        return { rms: this.rms, visemes: this.visemes };
      }

      if (this.analyser) {
        this.analyser.getByteTimeDomainData(this.timeData);
        this.analyser.getByteFrequencyData(this.freqData);

        let sumSq = 0;
        for (let i = 0; i < this.timeData.length; i++) {
          const norm = (this.timeData[i] - 128) / 128;
          sumSq += norm * norm;
        }
        const rawRms = Math.sqrt(sumSq / this.timeData.length);
        const amp = Math.min(1.0, rawRms * 3.8);
        this.rms = this.rms * 0.35 + amp * 0.65;

        // Estimate vowel visemes from low/mid/high formant energy bands
        // Sample rate ~44.1kHz-48kHz, 256 bins => ~90Hz per bin
        const bandEnergy = (startBin, endBin) => {
          let total = 0;
          const count = Math.max(1, endBin - startBin);
          for (let b = startBin; b < endBin && b < this.freqData.length; b++) {
            total += this.freqData[b] / 255;
          }
          return total / count;
        };

        const f1Low = bandEnergy(2, 6);    // ~180 - 540 Hz (u, i, o)
        const f1High = bandEnergy(6, 12);  // ~540 - 1080 Hz (a, e)
        const f2Low = bandEnergy(10, 16);  // ~900 - 1440 Hz (o, u, a)
        const f2High = bandEnergy(18, 32); // ~1600 - 2880 Hz (i, e)

        const gate = this.rms > 0.03 ? Math.min(1.0, this.rms * 1.35) : 0;
        const targetA = Math.min(1.0, gate * (f1High * 1.15 + f2Low * 0.4));
        const targetI = Math.min(1.0, gate * (f1Low * 0.5 + f2High * 1.25));
        const targetU = Math.min(1.0, gate * (f1Low * 1.1 + f2Low * 0.5) * 0.85);
        const targetE = Math.min(1.0, gate * (f1High * 0.7 + f2High * 0.9));
        const targetO = Math.min(1.0, gate * (f1Low * 0.8 + f2Low * 1.1));

        const lerp = (a, b, t) => a + (b - a) * t;
        const s = Math.min(1.0, dt * 18);
        this.visemes.a = lerp(this.visemes.a, targetA, s);
        this.visemes.i = lerp(this.visemes.i, targetI, s);
        this.visemes.u = lerp(this.visemes.u, targetU, s);
        this.visemes.e = lerp(this.visemes.e, targetE, s);
        this.visemes.o = lerp(this.visemes.o, targetO, s);
      } else {
        // Fallback procedural syllabic modulation while audio element plays
        this._fallbackPhase += dt * 14.0;
        const env = 0.45 + 0.35 * Math.sin(this._fallbackPhase) + 0.2 * Math.cos(this._fallbackPhase * 2.3);
        this.rms = Math.max(0.05, Math.min(1.0, env));
        this.visemes.a = this.rms * (0.6 + 0.4 * Math.sin(this._fallbackPhase * 0.7));
        this.visemes.o = this.rms * (0.4 + 0.4 * Math.cos(this._fallbackPhase * 1.1));
        this.visemes.i = this.rms * 0.25;
        this.visemes.u = this.rms * 0.2;
        this.visemes.e = this.rms * 0.3;
      }

      return { rms: this.rms, visemes: this.visemes };
    }
  }

  // ============================================================================
  // Part 5.3: Procedural Idle Life Controller (Breathing, Blinking, Gaze Saccades)
  // ============================================================================
  class ProceduralLifeController {
    constructor() {
      this.time = 0;
      this.nextBlinkAt = 2.2 + Math.random() * 3.0;
      this.blinkProgress = -1;
      this.blinkDuration = 0.15;
      this.blinkValue = 0;

      this.gazeTargetX = 0;
      this.gazeTargetY = 0;
      this.gazeCurrentX = 0;
      this.gazeCurrentY = 0;
      this.pointerActiveUntil = 0;
      this.nextSaccadeAt = 1.5;
    }

    setPointerGaze(normX, normY) {
      this.gazeTargetX = Math.max(-1, Math.min(1, normX));
      this.gazeTargetY = Math.max(-1, Math.min(1, normY));
      this.pointerActiveUntil = this.time + 2.5;
    }

    update(dt) {
      this.time += dt;

      // 1. Harmonic breathing & body sway
      const breath = 0.5 + 0.5 * Math.sin(this.time * 1.9);
      const swayX = Math.sin(this.time * 0.85) * 0.35 + Math.sin(this.time * 0.37) * 0.15;
      const swayY = Math.cos(this.time * 1.1) * 0.25;
      const headTilt = Math.sin(this.time * 0.62) * 0.22;

      // 2. Stochastic natural eye blinking (with occasional double-blink)
      if (this.blinkProgress < 0 && this.time >= this.nextBlinkAt) {
        this.blinkProgress = 0;
        const isDouble = Math.random() < 0.18;
        this.nextBlinkAt = this.time + (isDouble ? 0.26 : 2.0 + Math.random() * 4.2);
      }
      if (this.blinkProgress >= 0) {
        this.blinkProgress += dt / this.blinkDuration;
        if (this.blinkProgress >= 1.0) {
          this.blinkProgress = -1;
          this.blinkValue = 0;
        } else {
          this.blinkValue = Math.sin(this.blinkProgress * Math.PI);
        }
      }

      // 3. Micro-saccades when pointer is idle
      if (this.time > this.pointerActiveUntil && this.time >= this.nextSaccadeAt) {
        this.gazeTargetX = (Math.random() - 0.5) * 0.55;
        this.gazeTargetY = (Math.random() - 0.5) * 0.35;
        this.nextSaccadeAt = this.time + 1.4 + Math.random() * 2.8;
      }

      const damp = Math.min(1.0, dt * 8.0);
      this.gazeCurrentX += (this.gazeTargetX - this.gazeCurrentX) * damp;
      this.gazeCurrentY += (this.gazeTargetY - this.gazeCurrentY) * damp;

      return {
        time: this.time,
        breath,
        swayX,
        swayY,
        headTilt,
        blink: this.blinkValue,
        gazeX: this.gazeCurrentX,
        gazeY: this.gazeCurrentY,
      };
    }
  }

  // ============================================================================
  // Part 4.1: 2D PNGTuber Reactive Multi-State Sprite Controller
  // ============================================================================
  class PNGTuberController {
    constructor(containerEl) {
      this.containerEl = containerEl;
      this.overlayCanvas = null;
      this.ctx = null;
      this.manifest = null;
      this.images = {};
      this.currentEmotion = "neutral";
      this.lipSyncRms = 0;
      this.visemes = { a: 0, i: 0, u: 0, e: 0, o: 0 };
      this.scale = 1.0;
      this.posX = 0;
      this.posY = 0;
      this.bounceBoost = 0;
    }

    async load(url, modelInfo) {
      this.unload();
      this.scale = Number(modelInfo?.kScale || 0.5) * 2.0;

      const canvas = document.createElement("canvas");
      canvas.id = "olv-avatar-overlay-canvas";
      canvas.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;pointer-events:auto;z-index:2;display:block;cursor:grab;";
      this.containerEl.appendChild(canvas);
      this.overlayCanvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.onResize();
      this._bindInteraction(canvas);

      let baseOrigin = "";
      try {
        const u = new URL(url, window.location.href);
        baseOrigin = u.origin;
      } catch (_) {}

      const resolveAssetUrl = (p) => {
        if (!p) return "";
        if (p.startsWith("http://") || p.startsWith("https://") || p.startsWith("data:")) return p;
        return baseOrigin + (p.startsWith("/") ? p : "/" + p);
      };

      if (url.toLowerCase().endsWith(".json")) {
        const resp = await fetch(url);
        this.manifest = await resp.json();
      } else {
        this.manifest = {
          name: modelInfo?.name || "pngtuber",
          states: { closed: url, open: url },
          emotions: {},
          lipSyncThreshold: 0.08,
          bounceIntensity: 8.0,
        };
      }

      const loadImg = (src) =>
        new Promise((resolve) => {
          if (!src) return resolve(null);
          const full = resolveAssetUrl(src);
          if (this.images[src]) return resolve(this.images[src]);
          const img = new Image();
          img.crossOrigin = "anonymous";
          img.onload = () => {
            this.images[src] = img;
            resolve(img);
          };
          img.onerror = () => resolve(null);
          img.src = full;
        });

      const states = this.manifest.states || {};
      for (const src of Object.values(states)) {
        await loadImg(src);
      }
      const emotions = this.manifest.emotions || {};
      for (const emoObj of Object.values(emotions)) {
        if (emoObj && typeof emoObj === "object") {
          for (const src of Object.values(emoObj)) {
            await loadImg(src);
          }
        }
      }
    }

    _bindInteraction(canvas) {
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        canvas.style.cursor = "grabbing";
        try {
          canvas.setPointerCapture(e.pointerId);
        } catch (_) {}
      });

      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const w = Math.max(1, canvas.clientWidth || window.innerWidth);
        const h = Math.max(1, canvas.clientHeight || window.innerHeight);
        this.posX += ((e.clientX - lastX) / w) * 2.0;
        this.posY += ((e.clientY - lastY) / h) * 2.0;
        lastX = e.clientX;
        lastY = e.clientY;
      });

      const stopDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        canvas.style.cursor = "grab";
        try {
          canvas.releasePointerCapture(e.pointerId);
        } catch (_) {}
      };
      canvas.addEventListener("pointerup", stopDrag);
      canvas.addEventListener("pointercancel", stopDrag);

      canvas.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.08 : 0.92;
          this.scale = Math.max(0.15, Math.min(6.0, this.scale * factor));
        },
        { passive: false }
      );

      canvas.addEventListener("dblclick", () => {
        this.posX = 0;
        this.posY = 0;
        this.scale = 1.0;
      });
    }

    onResize() {
      if (!this.overlayCanvas || !this.containerEl) return;
      const dpr = window.devicePixelRatio || 1;
      const w = this.containerEl.clientWidth || window.innerWidth;
      const h = this.containerEl.clientHeight || window.innerHeight;
      this.overlayCanvas.width = Math.max(1, Math.floor(w * dpr));
      this.overlayCanvas.height = Math.max(1, Math.floor(h * dpr));
    }

    setEmotion(emotionName) {
      this.currentEmotion = emotionName || "neutral";
      this.bounceBoost = 1.0;
    }

    playMotion() {
      this.bounceBoost = 1.25;
    }

    setLipSync(rms, visemes) {
      this.lipSyncRms = rms;
      if (visemes) this.visemes = visemes;
    }

    update(dt, life) {
      if (!this.ctx || !this.overlayCanvas || !this.manifest) return;
      const ctx = this.ctx;
      const cw = this.overlayCanvas.width;
      const ch = this.overlayCanvas.height;
      ctx.clearRect(0, 0, cw, ch);

      this.bounceBoost = Math.max(0, this.bounceBoost - dt * 3.0);

      const threshold = Number(this.manifest.lipSyncThreshold || 0.07);
      const isTalking = this.lipSyncRms > threshold;
      const isBlinking = life.blink > 0.55;

      const states = this.manifest.states || {};
      const emoStates = (this.manifest.emotions || {})[this.currentEmotion] || {};

      let chosenKey = isTalking ? "open" : "closed";
      if (isBlinking) {
        if (isTalking && (emoStates.blink_open || states.blink_open)) {
          chosenKey = "blink_open";
        } else if (emoStates.blink || states.blink) {
          chosenKey = "blink";
        }
      }

      const src =
        emoStates[chosenKey] ||
        emoStates.closed ||
        states[chosenKey] ||
        states.closed ||
        Object.values(states)[0];
      const img = this.images[src];
      if (!img) return;

      const bounceAmp = Number(this.manifest.bounceIntensity || 8.0);
      const talkBounce = isTalking ? Math.sin(life.time * 22) * bounceAmp * this.lipSyncRms : 0;
      const jumpOffset = Math.sin(this.bounceBoost * Math.PI) * 18;
      const breathScaleY = 1.0 + (life.breath - 0.5) * 0.018 + this.lipSyncRms * 0.025;
      const breathScaleX = 1.0 - (life.breath - 0.5) * 0.009;
      const tiltRad = (life.headTilt * 1.8 + life.gazeX * 2.2) * (Math.PI / 180);

      const baseFit = Math.min(cw / (img.width || 512), ch / (img.height || 512)) * 0.82;
      const drawW = (img.width || 512) * baseFit * Math.max(0.2, this.scale);
      const drawH = (img.height || 512) * baseFit * Math.max(0.2, this.scale);

      ctx.save();
      ctx.translate(
        cw * 0.5 + this.posX * cw * 0.5 + life.swayX * 8,
        ch * 0.55 + this.posY * ch * 0.5 - Math.abs(talkBounce) - jumpOffset
      );
      ctx.rotate(tiltRad);
      ctx.scale(breathScaleX, breathScaleY);
      ctx.drawImage(img, -drawW * 0.5, -drawH * 0.5, drawW, drawH);
      ctx.restore();
    }

    unload() {
      if (this.overlayCanvas && this.overlayCanvas.parentNode) {
        this.overlayCanvas.parentNode.removeChild(this.overlayCanvas);
      }
      this.overlayCanvas = null;
      this.ctx = null;
      this.images = {};
    }
  }

  // ============================================================================
  // Part 4.2: Legacy Live2D Cubism 2.1 (`.model.json`) Adapter (`live2d.min.js`)
  // ============================================================================
  let cubism2ScriptPromise = null;
  function ensureCubism2Runtime() {
    if (window.Live2D && window.Live2DModelWebGL) return Promise.resolve();
    if (!cubism2ScriptPromise) {
      cubism2ScriptPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "./libs/live2d.min.js";
        script.onload = () => resolve();
        script.onerror = (e) => reject(e);
        document.head.appendChild(script);
      });
    }
    return cubism2ScriptPromise;
  }

  class Live2DCubism2Controller {
    constructor(containerEl) {
      this.containerEl = containerEl;
      this.glCanvas = null;
      this.gl = null;
      this.live2DModel = null;
      this.modelJson = null;
      this.textures = [];
      this.lipSyncRms = 0;
      this.visemes = { a: 0, i: 0, u: 0, e: 0, o: 0 };
      this.currentEmotion = "neutral";
      this.scale = 1.0;
      this.posX = 0;
      this.posY = 0;
    }

    async load(url, modelInfo) {
      this.unload();
      await ensureCubism2Runtime();
      this.scale = Number(modelInfo?.kScale || 0.5) * 2.0;

      const canvas = document.createElement("canvas");
      canvas.id = "olv-avatar-overlay-canvas";
      canvas.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;pointer-events:auto;z-index:2;display:block;cursor:grab;";
      this.containerEl.appendChild(canvas);
      this.glCanvas = canvas;
      this.onResize();
      this._bindInteraction(canvas);

      this.gl =
        canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true }) ||
        canvas.getContext("experimental-webgl", { alpha: true, premultipliedAlpha: true });
      if (!this.gl) throw new Error("WebGL not available for Live2D Cubism 2.1");

      window.Live2D.init();
      window.Live2D.setGL(this.gl);

      const resp = await fetch(url);
      this.modelJson = await resp.json();
      const baseDir = url.substring(0, url.lastIndexOf("/") + 1);

      const mocUrl = baseDir + (this.modelJson.model || "");
      const mocBuf = await (await fetch(mocUrl)).arrayBuffer();
      const model = window.Live2DModelWebGL.loadModel(mocBuf);
      model.setGL(this.gl);

      const texList = this.modelJson.textures || [];
      for (let i = 0; i < texList.length; i++) {
        const texUrl = baseDir + texList[i];
        await new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = "anonymous";
          img.onload = () => {
            const gl = this.gl;
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
            gl.generateMipmap(gl.TEXTURE_2D);
            model.setTexture(i, tex);
            this.textures.push(tex);
            resolve();
          };
          img.onerror = () => resolve();
          img.src = texUrl;
        });
      }

      this.live2DModel = model;
    }

    _bindInteraction(canvas) {
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        canvas.style.cursor = "grabbing";
        try {
          canvas.setPointerCapture(e.pointerId);
        } catch (_) {}
      });

      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const w = Math.max(1, canvas.clientWidth || window.innerWidth);
        const h = Math.max(1, canvas.clientHeight || window.innerHeight);
        this.posX += ((e.clientX - lastX) / w) * 2.0;
        this.posY -= ((e.clientY - lastY) / h) * 2.0;
        lastX = e.clientX;
        lastY = e.clientY;
      });

      const stopDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        canvas.style.cursor = "grab";
        try {
          canvas.releasePointerCapture(e.pointerId);
        } catch (_) {}
      };
      canvas.addEventListener("pointerup", stopDrag);
      canvas.addEventListener("pointercancel", stopDrag);

      canvas.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.08 : 0.92;
          this.scale = Math.max(0.15, Math.min(6.0, this.scale * factor));
        },
        { passive: false }
      );

      canvas.addEventListener("dblclick", () => {
        this.posX = 0;
        this.posY = 0;
        this.scale = 1.0;
      });
    }

    onResize() {
      if (!this.glCanvas || !this.containerEl) return;
      const dpr = window.devicePixelRatio || 1;
      const w = this.containerEl.clientWidth || window.innerWidth;
      const h = this.containerEl.clientHeight || window.innerHeight;
      this.glCanvas.width = Math.max(1, Math.floor(w * dpr));
      this.glCanvas.height = Math.max(1, Math.floor(h * dpr));
      if (this.gl) {
        this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
      }
    }

    setEmotion(emotionName) {
      this.currentEmotion = emotionName || "neutral";
    }

    playMotion() {}

    setLipSync(rms, visemes) {
      this.lipSyncRms = rms;
      if (visemes) this.visemes = visemes;
    }

    update(dt, life) {
      if (!this.gl || !this.live2DModel || !this.glCanvas) return;
      const gl = this.gl;
      const model = this.live2DModel;
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      const cw = this.glCanvas.width;
      const ch = this.glCanvas.height;
      const modelW = model.getCanvasWidth() || 2048;
      const modelH = model.getCanvasHeight() || 2048;
      const aspect = cw / Math.max(1, ch);
      const s = (2.0 / modelW) * Math.max(0.2, this.scale);
      const sy = s * aspect;

      const matrix = new Float32Array([
        s, 0, 0, 0,
        0, -sy, 0, 0,
        0, 0, 1, 0,
        -1.0 + this.posX, 1.0 + this.posY, 0, 1,
      ]);
      model.setMatrix(matrix);

      const emoCfg = CANONICAL_EMOTIONS[this.currentEmotion] || CANONICAL_EMOTIONS.neutral;
      model.setParamFloat("PARAM_ANGLE_X", life.gazeX * 28 + life.swayX * 10);
      model.setParamFloat("PARAM_ANGLE_Y", life.gazeY * 22 + life.swayY * 8);
      model.setParamFloat("PARAM_ANGLE_Z", life.headTilt * 20);
      model.setParamFloat("PARAM_BODY_ANGLE_X", life.gazeX * 8 + life.swayX * 5);
      model.setParamFloat("PARAM_EYE_BALL_X", life.gazeX);
      model.setParamFloat("PARAM_EYE_BALL_Y", life.gazeY);
      model.setParamFloat("PARAM_EYE_L_OPEN", 1.0 - life.blink);
      model.setParamFloat("PARAM_EYE_R_OPEN", 1.0 - life.blink);
      model.setParamFloat("PARAM_BREATH", life.breath);
      model.setParamFloat("PARAM_MOUTH_OPEN_Y", Math.min(1.0, this.lipSyncRms * 1.4));
      model.setParamFloat("PARAM_MOUTH_FORM", emoCfg.mouthCurve);
      model.setParamFloat("PARAM_BROW_L_Y", emoCfg.browOffset);
      model.setParamFloat("PARAM_BROW_R_Y", emoCfg.browOffset);

      model.update();
      model.draw();
    }

    unload() {
      if (this.gl && this.textures.length) {
        for (const t of this.textures) {
          try {
            this.gl.deleteTexture(t);
          } catch (_) {}
        }
      }
      this.textures = [];
      this.live2DModel = null;
      if (this.glCanvas && this.glCanvas.parentNode) {
        this.glCanvas.parentNode.removeChild(this.glCanvas);
      }
      this.glCanvas = null;
      this.gl = null;
    }
  }

  // ============================================================================
  // Part 3.2 & 3.3: 3D VRM (`.vrm`) & glTF (`.glb`/`.gltf`) Controller
  // Uses Three.js + `@pixiv/three-vrm` when available, with a built-in native
  // WebGL2 GLB/VRM binary parser, morph-target / humanoid bone animator, and
  // MToon-style cel shader so `.vrm` and `.glb` models render 100% offline!
  // ============================================================================
  class VRMAndGLTF3DController {
    constructor(containerEl) {
      this.containerEl = containerEl;
      this.glCanvas = null;
      this.gl = null;
      this.program = null;
      this.drawCalls = [];
      this.nodes = [];
      this.vrmMeta = null;
      this.morphBindings = {}; // expressionName -> [{ meshIdx, targetIdx, weight }]
      this.boneNodes = {}; // humanoidBoneName -> nodeIndex
      this.currentEmotion = "neutral";
      this.lipSyncRms = 0;
      this.visemes = { a: 0, i: 0, u: 0, e: 0, o: 0 };
      this.scale = 1.0;
      this.posX = 0;
      this.posY = 0;
      this.yawOffset = 0;
      this.userYaw = 0;
      this.userPitch = 0;
      this.bboxCenter = [0, 0.9, 0];
      this.bboxRadius = 1.0;
    }

    async load(url, modelInfo) {
      this.unload();
      this.scale = Number(modelInfo?.kScale || 0.5) * 2.0;

      const canvas = document.createElement("canvas");
      canvas.id = "olv-avatar-overlay-canvas";
      canvas.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;pointer-events:auto;z-index:2;display:block;cursor:grab;";
      this.containerEl.appendChild(canvas);
      this.glCanvas = canvas;
      this.onResize();
      this._bindInteraction(canvas);

      const gl = canvas.getContext("webgl2", { alpha: true, antialias: true });
      if (!gl) throw new Error("WebGL2 is required for 3D VRM / glTF rendering");
      this.gl = gl;
      this._initShaderProgram(gl);

      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Failed to fetch 3D model (${resp.status})`);
      const buf = await resp.arrayBuffer();
      if (String(url).toLowerCase().endsWith(".mqo") || modelInfo?.format === "mqo") {
        await this._parseMQO(buf, url);
      } else {
        await this._parseGLBOrGLTF(buf, url);
      }
    }

    _bindInteraction(canvas) {
      let dragging = false;
      let rotateMode = false;
      let lastX = 0;
      let lastY = 0;

      canvas.addEventListener("contextmenu", (e) => e.preventDefault());

      canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 && e.button !== 2) return;
        dragging = true;
        rotateMode = e.button === 2 || e.shiftKey;
        lastX = e.clientX;
        lastY = e.clientY;
        canvas.style.cursor = "grabbing";
        try {
          canvas.setPointerCapture(e.pointerId);
        } catch (_) {}
      });

      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;

        if (rotateMode || e.shiftKey) {
          this.userYaw += dx * 0.008;
          this.userPitch = Math.max(-0.8, Math.min(0.8, this.userPitch + dy * 0.006));
        } else {
          const h = Math.max(1, canvas.clientHeight || window.innerHeight);
          const dist = (this.bboxRadius * 2.35) / Math.max(0.1, this.scale);
          const worldPerPixel = (2.0 * dist * Math.tan(15 * (Math.PI / 180))) / h;
          this.posX += dx * worldPerPixel;
          this.posY -= dy * worldPerPixel;
        }
      });

      const stopDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        canvas.style.cursor = "grab";
        try {
          canvas.releasePointerCapture(e.pointerId);
        } catch (_) {}
      };
      canvas.addEventListener("pointerup", stopDrag);
      canvas.addEventListener("pointercancel", stopDrag);

      canvas.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.1 : 0.9;
          this.scale = Math.max(0.1, Math.min(10.0, this.scale * factor));
        },
        { passive: false }
      );

      canvas.addEventListener("dblclick", () => {
        this.posX = 0;
        this.posY = 0;
        this.userYaw = 0;
        this.userPitch = 0;
        this.scale = 1.0;
      });
    }

    _initShaderProgram(gl) {
      const vsSource = `#version 300 es
        precision highp float;
        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec3 a_normal;
        layout(location = 2) in vec2 a_uv;
        layout(location = 3) in vec3 a_morphPos0;
        layout(location = 4) in vec3 a_morphPos1;

        uniform mat4 u_mvp;
        uniform mat4 u_model;
        uniform vec2 u_morphWeights;

        out vec3 v_normal;
        out vec2 v_uv;
        out vec3 v_worldPos;

        void main() {
          vec3 pos = a_position + a_morphPos0 * u_morphWeights.x + a_morphPos1 * u_morphWeights.y;
          vec4 world = u_model * vec4(pos, 1.0);
          v_worldPos = world.xyz;
          v_normal = normalize(mat3(u_model) * a_normal);
          v_uv = a_uv;
          gl_Position = u_mvp * vec4(pos, 1.0);
        }
      `;

      const fsSource = `#version 300 es
        precision highp float;
        in vec3 v_normal;
        in vec2 v_uv;
        in vec3 v_worldPos;

        uniform sampler2D u_baseColorTex;
        uniform int u_hasTex;
        uniform vec4 u_baseColorFactor;

        out vec4 outColor;

        void main() {
          vec4 base = u_baseColorFactor;
          if (u_hasTex == 1) {
            base *= texture(u_baseColorTex, v_uv);
          }
          if (base.a < 0.05) discard;

          // MToon-inspired anime cel-shading lighting
          vec3 N = normalize(v_normal);
          vec3 L = normalize(vec3(0.35, 0.65, 0.75));
          float ndl = dot(N, L) * 0.5 + 0.5;
          float toon = smoothstep(0.38, 0.48, ndl);
          vec3 shadeColor = base.rgb * 0.78;
          vec3 litColor = mix(shadeColor, base.rgb * 1.04, toon);

          // Soft rim light
          vec3 V = normalize(vec3(0.0, 0.1, 1.0));
          float rim = pow(max(0.0, 1.0 - abs(dot(N, V))), 3.2) * 0.18;
          outColor = vec4(litColor + vec3(rim), base.a);
        }
      `;

      const compile = (type, src) => {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        return sh;
      };
      const vs = compile(gl.VERTEX_SHADER, vsSource);
      const fs = compile(gl.FRAGMENT_SHADER, fsSource);
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      this.program = prog;
    }

    async _parseGLBOrGLTF(arrayBuffer, baseUrl) {
      const dv = new DataView(arrayBuffer);
      let json = null;
      let binChunk = null;

      // Check GLB magic 0x46546C67 ("glTF")
      if (dv.byteLength >= 12 && dv.getUint32(0, true) === 0x46546c67) {
        let offset = 12;
        while (offset + 8 <= dv.byteLength) {
          const chunkLen = dv.getUint32(offset, true);
          const chunkType = dv.getUint32(offset + 4, true);
          const chunkStart = offset + 8;
          if (chunkType === 0x4e4f534a) {
            // JSON
            const dec = new TextDecoder("utf-8");
            json = JSON.parse(dec.decode(new Uint8Array(arrayBuffer, chunkStart, chunkLen)));
          } else if (chunkType === 0x004e4942) {
            // BIN
            binChunk = new Uint8Array(arrayBuffer, chunkStart, chunkLen);
          }
          offset = chunkStart + chunkLen;
        }
      } else {
        const dec = new TextDecoder("utf-8");
        json = JSON.parse(dec.decode(new Uint8Array(arrayBuffer)));
      }

      if (!json) throw new Error("Invalid VRM/glTF file");

      const baseDir = baseUrl.substring(0, baseUrl.lastIndexOf("/") + 1);
      const buffers = [];
      for (let i = 0; i < (json.buffers || []).length; i++) {
        const b = json.buffers[i];
        if (i === 0 && binChunk && !b.uri) {
          buffers.push(binChunk);
        } else if (b.uri) {
          const uri = b.uri.startsWith("data:") || b.uri.startsWith("http") ? b.uri : baseDir + b.uri;
          const r = await fetch(uri);
          buffers.push(new Uint8Array(await r.arrayBuffer()));
        }
      }

      const getAccessorFloatArray = (accIdx) => {
        if (accIdx == null || !json.accessors || !json.accessors[accIdx]) return null;
        const acc = json.accessors[accIdx];
        if (acc.bufferView == null) return null;
        const bv = json.bufferViews[acc.bufferView];
        const buf = buffers[bv.buffer || 0];
        if (!buf) return null;

        const numComps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[acc.type] || 1;
        const count = acc.count || 0;
        const out = new Float32Array(count * numComps);
        const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
        const stride = bv.byteStride || numComps * 4;
        const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

        for (let i = 0; i < count; i++) {
          const base = byteOffset + i * stride;
          for (let c = 0; c < numComps; c++) {
            if (acc.componentType === 5126) {
              out[i * numComps + c] = view.getFloat32(base + c * 4, true);
            } else if (acc.componentType === 5123) {
              out[i * numComps + c] = view.getUint16(base + c * 2, true);
            } else if (acc.componentType === 5125) {
              out[i * numComps + c] = view.getUint32(base + c * 4, true);
            } else if (acc.componentType === 5121) {
              out[i * numComps + c] = view.getUint8(base + c);
            }
          }
        }
        return out;
      };

      const getIndexArray = (accIdx) => {
        if (accIdx == null || !json.accessors || !json.accessors[accIdx]) return null;
        const acc = json.accessors[accIdx];
        const bv = json.bufferViews[acc.bufferView];
        const buf = buffers[bv.buffer || 0];
        const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
        const count = acc.count || 0;
        const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        const out = new Uint32Array(count);
        for (let i = 0; i < count; i++) {
          if (acc.componentType === 5123) {
            out[i] = view.getUint16(byteOffset + i * 2, true);
          } else if (acc.componentType === 5125) {
            out[i] = view.getUint32(byteOffset + i * 4, true);
          } else {
            out[i] = view.getUint8(byteOffset + i);
          }
        }
        return out;
      };

      // Load textures
      const gl = this.gl;
      const glTextures = [];
      for (let i = 0; i < (json.images || []).length; i++) {
        const imgDef = json.images[i];
        let blobUrl = null;
        if (imgDef.bufferView != null) {
          const bv = json.bufferViews[imgDef.bufferView];
          const buf = buffers[bv.buffer || 0];
          const slice = buf.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
          blobUrl = URL.createObjectURL(new Blob([slice], { type: imgDef.mimeType || "image/png" }));
        } else if (imgDef.uri) {
          blobUrl = imgDef.uri.startsWith("data:") || imgDef.uri.startsWith("http")
            ? imgDef.uri
            : baseDir + imgDef.uri;
        }
        if (blobUrl) {
          const tex = await new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
              const t = gl.createTexture();
              gl.bindTexture(gl.TEXTURE_2D, t);
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
              gl.generateMipmap(gl.TEXTURE_2D);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
              if (blobUrl.startsWith("blob:")) URL.revokeObjectURL(blobUrl);
              resolve(t);
            };
            img.onerror = () => resolve(null);
            img.src = blobUrl;
          });
          glTextures.push(tex);
        } else {
          glTextures.push(null);
        }
      }

      // Parse VRM 0.x (`extensions.VRM`) or VRM 1.0 (`extensions.VRMC_vrm`) humanoid bones & expressions
      const extVRM0 = json.extensions && json.extensions.VRM;
      const extVRM1 = json.extensions && json.extensions.VRMC_vrm;
      if (extVRM0) {
        // VRM 0.x models face -Z by default; rotate 180 degrees around Y so they face the camera
        this.yawOffset = Math.PI;
        const humanBones = (extVRM0.humanoid && extVRM0.humanoid.humanBones) || [];
        for (const hb of humanBones) {
          if (hb.bone && hb.node != null) this.boneNodes[hb.bone] = hb.node;
        }
        const groups =
          (extVRM0.blendShapeMaster && extVRM0.blendShapeMaster.blendShapeGroups) || [];
        for (const g of groups) {
          const key = (g.presetName || g.name || "").toLowerCase();
          if (key && Array.isArray(g.binds)) {
            this.morphBindings[key] = g.binds.map((b) => ({
              mesh: b.mesh,
              index: b.index,
              weight: (b.weight != null ? b.weight : 100) / 100,
            }));
          }
        }
      } else if (extVRM1) {
        this.yawOffset = 0;
        const humanBones = (extVRM1.humanoid && extVRM1.humanoid.humanBones) || {};
        for (const [boneName, bObj] of Object.entries(humanBones)) {
          if (bObj && bObj.node != null) this.boneNodes[boneName] = bObj.node;
        }
        const preset = (extVRM1.expressions && extVRM1.expressions.preset) || {};
        for (const [expName, expObj] of Object.entries(preset)) {
          if (expObj && Array.isArray(expObj.morphTargetBinds)) {
            this.morphBindings[expName.toLowerCase()] = expObj.morphTargetBinds.map((b) => ({
              node: b.node,
              index: b.index,
              weight: b.weight != null ? b.weight : 1.0,
            }));
          }
        }
      }

      // Compute node hierarchy transforms
      const nodes = (json.nodes || []).map((n) => ({
        ...n,
        parent: null,
        worldMatrix: this._identityMat4(),
      }));
      nodes.forEach((n, idx) => {
        for (const ch of n.children || []) {
          if (nodes[ch]) nodes[ch].parent = idx;
        }
      });
      this.nodes = nodes;
      this._updateNodeHierarchy({ headYaw: 0, headPitch: 0, headRoll: 0, spinePitch: 0 });

      // Upload mesh primitives to WebGL2 VAOs
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

      const createVbo = (data, loc, size) => {
        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        return vbo;
      };

      for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
        const node = nodes[nodeIdx];
        if (node.mesh == null || !json.meshes || !json.meshes[node.mesh]) continue;
        const meshDef = json.meshes[node.mesh];

        for (const prim of meshDef.primitives || []) {
          const attrs = prim.attributes || {};
          const pos = getAccessorFloatArray(attrs.POSITION);
          if (!pos) continue;
          const vCount = pos.length / 3;
          const norm =
            getAccessorFloatArray(attrs.NORMAL) || new Float32Array(vCount * 3);
          const uv =
            getAccessorFloatArray(attrs.TEXCOORD_0) || new Float32Array(vCount * 2);
          const indices = getIndexArray(prim.indices);

          // Update bounding box in node space
          const wm = node.worldMatrix;
          for (let i = 0; i < vCount; i++) {
            const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
            const wx = wm[0] * x + wm[4] * y + wm[8] * z + wm[12];
            const wy = wm[1] * x + wm[5] * y + wm[9] * z + wm[13];
            const wz = wm[2] * x + wm[6] * y + wm[10] * z + wm[14];
            if (wx < minX) minX = wx;
            if (wy < minY) minY = wy;
            if (wz < minZ) minZ = wz;
            if (wx > maxX) maxX = wx;
            if (wy > maxY) maxY = wy;
            if (wz > maxZ) maxZ = wz;
          }

          // Extract up to 2 primary morph target position deltas (e.g. mouth open + blink)
          const targets = prim.targets || [];
          let morph0 = new Float32Array(vCount * 3);
          let morph1 = new Float32Array(vCount * 3);
          let morph0Idx = 0;
          let morph1Idx = 1;

          // Identify blink & mouth morph target indices if targetNames exist
          const targetNames = (meshDef.extras && meshDef.extras.targetNames) || [];
          for (let tIdx = 0; tIdx < targets.length; tIdx++) {
            const tName = (targetNames[tIdx] || "").toLowerCase();
            if (
              tName.includes("mouth") ||
              tName.includes("mth_a") ||
              tName.includes("aa") ||
              tName === "a" ||
              tName.includes("vrc.v_aa")
            ) {
              morph0Idx = tIdx;
            } else if (
              tName.includes("blink") ||
              tName.includes("eye_close") ||
              tName.includes("wink")
            ) {
              morph1Idx = tIdx;
            }
          }
          if (targets[morph0Idx] && targets[morph0Idx].POSITION != null) {
            morph0 = getAccessorFloatArray(targets[morph0Idx].POSITION) || morph0;
          }
          if (targets[morph1Idx] && targets[morph1Idx].POSITION != null) {
            morph1 = getAccessorFloatArray(targets[morph1Idx].POSITION) || morph1;
          }

          const vao = gl.createVertexArray();
          gl.bindVertexArray(vao);
          createVbo(pos, 0, 3);
          createVbo(norm, 1, 3);
          createVbo(uv, 2, 2);
          createVbo(morph0, 3, 3);
          createVbo(morph1, 4, 3);

          let indexBuffer = null;
          let indexCount = vCount;
          if (indices) {
            indexBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
            indexCount = indices.length;
          }
          gl.bindVertexArray(null);

          // Resolve material baseColor, unlit mode, and outline detection
          let baseColorFactor = [1, 1, 1, 1];
          let tex = null;
          let doubleSided = false;
          let isOutline = false;
          if (prim.material != null && json.materials && json.materials[prim.material]) {
            const mat = json.materials[prim.material];
            const matName = String(mat.name || "").trim().toLowerCase();
            const pbr = mat.pbrMetallicRoughness || {};
            const hasBaseTex = Boolean(
              pbr.baseColorTexture && pbr.baseColorTexture.index != null
            );
            if (
              matName.includes("outline") ||
              (!hasBaseTex && !Array.isArray(pbr.baseColorFactor) && matName.includes("line"))
            ) {
              isOutline = true;
            }
            if (Array.isArray(pbr.baseColorFactor)) {
              baseColorFactor = pbr.baseColorFactor;
            } else if (matName === "black" && !hasBaseTex) {
              baseColorFactor = [0.06, 0.06, 0.08, 1.0];
            }
            doubleSided = Boolean(mat.doubleSided);
            const texInfo = pbr.baseColorTexture;
            if (texInfo && texInfo.index != null && json.textures && json.textures[texInfo.index]) {
              const srcImgIdx = json.textures[texInfo.index].source;
              tex = glTextures[srcImgIdx] || null;
            }
          }

          if (isOutline) {
            // Skip inverted-hull / duplicate outline shells so inner textured meshes aren't occluded
            continue;
          }

          this.drawCalls.push({
            vao,
            indexBuffer,
            indexCount,
            nodeIdx,
            meshIdx: node.mesh,
            baseColorFactor,
            texture: tex,
            doubleSided,
          });
        }
      }

      if (isFinite(minX) && isFinite(maxX)) {
        const cx = (minX + maxX) * 0.5;
        const cy = (minY + maxY) * 0.5;
        const cz = (minZ + maxZ) * 0.5;
        const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.25);
        // Frame full character nicely and allow scroll-wheel zoom in/out
        this.bboxCenter = [cx, cy, cz];
        this.bboxRadius = span * 0.65;
      }
    }

    async _parseMQO(arrayBuffer, baseUrl) {
      const gl = this.gl;
      let text = "";
      try {
        text = new TextDecoder("shift_jis").decode(new Uint8Array(arrayBuffer));
      } catch (_) {
        text = new TextDecoder("utf-8").decode(new Uint8Array(arrayBuffer));
      }

      const baseDir = baseUrl.substring(0, baseUrl.lastIndexOf("/") + 1);
      const lines = text.split(/\r?\n/);

      const materials = [];
      const objects = [];
      let state = "root";
      let currentObj = null;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        if (state === "root") {
          if (line.startsWith("Thumbnail ") && line.endsWith("{")) {
            state = "skip_block";
          } else if (line.startsWith("Material ") && line.endsWith("{")) {
            state = "material";
          } else if (line.startsWith("Object ") && line.endsWith("{")) {
            const mName = line.match(/^Object\s+"([^"]*)"/);
            currentObj = {
              name: mName ? mName[1] : `obj_${objects.length}`,
              visible: 15,
              vertices: [],
              faces: [],
            };
            objects.push(currentObj);
            state = "object";
          }
        } else if (state === "skip_block") {
          if (line === "}") state = "root";
        } else if (state === "material") {
          if (line === "}") {
            state = "root";
            continue;
          }
          const mName = line.match(/^"([^"]*)"/);
          const mCol = line.match(/col\(\s*([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s*\)/);
          const mTex = line.match(/tex\("([^"]+)"\)/);
          materials.push({
            name: mName ? mName[1] : `mat_${materials.length}`,
            color: mCol
              ? [Number(mCol[1]), Number(mCol[2]), Number(mCol[3]), Number(mCol[4])]
              : [1, 1, 1, 1],
            texPath: mTex ? mTex[1].replace(/\\/g, "/") : null,
          });
        } else if (state === "object") {
          if (line === "}") {
            currentObj = null;
            state = "root";
          } else if (line.startsWith("visible ")) {
            currentObj.visible = parseInt(line.split(/\s+/)[1], 10);
          } else if (line.startsWith("vertex ") && line.endsWith("{")) {
            state = "vertex";
          } else if (line.startsWith("face ") && line.endsWith("{")) {
            state = "face";
          } else if (line.endsWith("{")) {
            state = "object_subblock";
          }
        } else if (state === "object_subblock") {
          if (line === "}") state = "object";
        } else if (state === "vertex") {
          if (line === "}") {
            state = "object";
            continue;
          }
          const parts = line.split(/\s+/);
          if (parts.length >= 3) {
            currentObj.vertices.push([
              parseFloat(parts[0]),
              parseFloat(parts[1]),
              parseFloat(parts[2]),
            ]);
          }
        } else if (state === "face") {
          if (line === "}") {
            state = "object";
            continue;
          }
          const nVerts = parseInt(line, 10);
          if (nVerts < 3) continue;
          const mV = line.match(/V\(([^)]+)\)/);
          if (!mV) continue;
          const vIdxs = mV[1].trim().split(/\s+/).map(Number);
          const mM = line.match(/M\((\d+)\)/);
          const matIdx = mM ? parseInt(mM[1], 10) : 0;
          const mUV = line.match(/UV\(([^)]+)\)/);
          const uvs = mUV ? mUV[1].trim().split(/\s+/).map(Number) : [];
          currentObj.faces.push({ nVerts, vIdxs, matIdx, uvs });
        }
      }

      // Load textures referenced by materials
      const glTextures = [];
      for (let i = 0; i < materials.length; i++) {
        const texPath = materials[i].texPath;
        if (!texPath) {
          glTextures.push(null);
          continue;
        }
        const texUrl =
          texPath.startsWith("http") || texPath.startsWith("/") ? texPath : baseDir + texPath;
        const tex = await new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = "anonymous";
          img.onload = () => {
            const t = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, t);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
            gl.generateMipmap(gl.TEXTURE_2D);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            resolve(t);
          };
          img.onerror = () => resolve(null);
          img.src = texUrl;
        });
        glTextures.push(tex);
      }

      this.nodes = [{ name: "mqo_root", parent: null, worldMatrix: this._identityMat4() }];

      // Group triangles by material index and compute smooth vertex normals + character bounds
      const buckets = new Map(); // matIdx -> { pos: [], norm: [], uv: [] }
      let charMinX = Infinity, charMinY = Infinity, charMinZ = Infinity;
      let charMaxX = -Infinity, charMaxY = -Infinity, charMaxZ = -Infinity;
      let allMinX = Infinity, allMinY = Infinity, allMinZ = Infinity;
      let allMaxX = -Infinity, allMaxY = -Infinity, allMaxZ = -Infinity;

      for (const obj of objects) {
        if (obj.visible === 0 || !obj.vertices.length || !obj.faces.length) continue;
        const verts = obj.vertices;
        const vNormals = new Float32Array(verts.length * 3);

        // Accumulate face normals into shared vertices for smooth shading
        for (const f of obj.faces) {
          const v0 = verts[f.vIdxs[0]];
          const v1 = verts[f.vIdxs[1]];
          const v2 = verts[f.vIdxs[2]];
          if (!v0 || !v1 || !v2) continue;
          // Metasequoia faces are clockwise; (v2 - v0) x (v1 - v0) points outward
          const ux = v2[0] - v0[0], uy = v2[1] - v0[1], uz = v2[2] - v0[2];
          const vx = v1[0] - v0[0], vy = v1[1] - v0[1], vz = v1[2] - v0[2];
          const nx = uy * vz - uz * vy;
          const ny = uz * vx - ux * vz;
          const nz = ux * vy - uy * vx;
          for (const vi of f.vIdxs) {
            vNormals[vi * 3] += nx;
            vNormals[vi * 3 + 1] += ny;
            vNormals[vi * 3 + 2] += nz;
          }
        }

        for (let vi = 0; vi < verts.length; vi++) {
          const nx = vNormals[vi * 3], ny = vNormals[vi * 3 + 1], nz = vNormals[vi * 3 + 2];
          const len = Math.hypot(nx, ny, nz) || 1.0;
          vNormals[vi * 3] = nx / len;
          vNormals[vi * 3 + 1] = ny / len;
          vNormals[vi * 3 + 2] = nz / len;
        }

        for (const f of obj.faces) {
          const mat = materials[f.matIdx] || { name: "", color: [1, 1, 1, 1], texPath: null };
          const texNameLower = String(mat.texPath || "").toLowerCase();
          // Skip giant stage/background dome planes so the VTuber avatar stands cleanly over the app background
          if (texNameLower.includes("background")) continue;

          const isCharacterTex =
            Boolean(mat.texPath) && !texNameLower.includes("background");

          let b = buckets.get(f.matIdx);
          if (!b) {
            b = { pos: [], norm: [], uv: [] };
            buckets.set(f.matIdx, b);
          }

          // Triangulate n-gon (3 or 4 vertices); reverse winding (0, k+1, k) to match OpenGL CCW
          const tris = f.nVerts === 4 ? [[0, 2, 1], [0, 3, 2]] : [[0, 2, 1]];
          for (const [a, c, d] of tris) {
            for (const corner of [a, c, d]) {
              const vi = f.vIdxs[corner];
              const v = verts[vi];
              if (!v) continue;
              b.pos.push(v[0], v[1], v[2]);
              b.norm.push(vNormals[vi * 3], vNormals[vi * 3 + 1], vNormals[vi * 3 + 2]);
              const u = f.uvs.length >= (corner + 1) * 2 ? f.uvs[corner * 2] : 0;
              const vCoord = f.uvs.length >= (corner + 1) * 2 ? f.uvs[corner * 2 + 1] : 0;
              b.uv.push(u, vCoord);

              if (v[0] < allMinX) allMinX = v[0];
              if (v[1] < allMinY) allMinY = v[1];
              if (v[2] < allMinZ) allMinZ = v[2];
              if (v[0] > allMaxX) allMaxX = v[0];
              if (v[1] > allMaxY) allMaxY = v[1];
              if (v[2] > allMaxZ) allMaxZ = v[2];

              if (isCharacterTex) {
                if (v[0] < charMinX) charMinX = v[0];
                if (v[1] < charMinY) charMinY = v[1];
                if (v[2] < charMinZ) charMinZ = v[2];
                if (v[0] > charMaxX) charMaxX = v[0];
                if (v[1] > charMaxY) charMaxY = v[1];
                if (v[2] > charMaxZ) charMaxZ = v[2];
              }
            }
          }
        }
      }

      const useCharBounds = isFinite(charMinY) && isFinite(charMaxY);
      const minX = useCharBounds ? charMinX : allMinX;
      const minY = useCharBounds ? charMinY : allMinY;
      const minZ = useCharBounds ? charMinZ : allMinZ;
      const maxX = useCharBounds ? charMaxX : allMaxX;
      const maxY = useCharBounds ? charMaxY : allMaxY;
      const maxZ = useCharBounds ? charMaxZ : allMaxZ;

      const headThresholdY = minY + (maxY - minY) * 0.72;
      const createVbo = (data, loc, size) => {
        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        return vbo;
      };

      for (const [matIdx, b] of buckets.entries()) {
        const vCount = b.pos.length / 3;
        if (!vCount) continue;

        // Filter out huge untextured stage/backdrop cages that span > 3x the character height
        const mat = materials[matIdx] || { color: [1, 1, 1, 1], texPath: null };
        if (!mat.texPath && useCharBounds) {
          let bMinY = Infinity, bMaxY = -Infinity;
          for (let i = 1; i < b.pos.length; i += 3) {
            if (b.pos[i] < bMinY) bMinY = b.pos[i];
            if (b.pos[i] > bMaxY) bMaxY = b.pos[i];
          }
          if (bMaxY - bMinY > (maxY - minY) * 2.2) {
            continue;
          }
        }

        const posArr = new Float32Array(b.pos);
        const normArr = new Float32Array(b.norm);
        const uvArr = new Float32Array(b.uv);
        const morph0 = new Float32Array(vCount * 3);
        const morph1 = new Float32Array(vCount * 3);

        // Add subtle procedural mouth/jaw motion in upper head region for static MQO meshes
        const texLower = String(mat.texPath || "").toLowerCase();
        if (texLower.includes("body") || texLower.includes("parts") || texLower.includes("face")) {
          const midX = (minX + maxX) * 0.5;
          const mouthY = minY + (maxY - minY) * 0.77;
          const radius = (maxY - minY) * 0.045;
          for (let i = 0; i < vCount; i++) {
            const vx = posArr[i * 3], vy = posArr[i * 3 + 1], vz = posArr[i * 3 + 2];
            if (vy > headThresholdY && vz > 0) {
              const d = Math.hypot(vx - midX, vy - mouthY);
              if (d < radius) {
                const w = 1.0 - d / radius;
                morph0[i * 3 + 1] = -w * (maxY - minY) * 0.012;
              }
            }
          }
        }

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        createVbo(posArr, 0, 3);
        createVbo(normArr, 1, 3);
        createVbo(uvArr, 2, 2);
        createVbo(morph0, 3, 3);
        createVbo(morph1, 4, 3);
        gl.bindVertexArray(null);

        this.drawCalls.push({
          vao,
          indexBuffer: null,
          indexCount: vCount,
          nodeIdx: 0,
          meshIdx: matIdx,
          baseColorFactor: mat.color || [1, 1, 1, 1],
          texture: glTextures[matIdx] || null,
          doubleSided: true,
        });
      }

      if (isFinite(minX) && isFinite(maxX)) {
        const cx = (minX + maxX) * 0.5;
        const cy = (minY + maxY) * 0.52;
        const cz = (minZ + maxZ) * 0.5;
        const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.25);
        this.bboxCenter = [cx, cy, cz];
        this.bboxRadius = span * 0.62;
      }
    }

    _identityMat4() {
      return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]);
    }

    _mulMat4(a, b) {
      const out = new Float32Array(16);
      for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
          out[c * 4 + r] =
            a[0 * 4 + r] * b[c * 4 + 0] +
            a[1 * 4 + r] * b[c * 4 + 1] +
            a[2 * 4 + r] * b[c * 4 + 2] +
            a[3 * 4 + r] * b[c * 4 + 3];
        }
      }
      return out;
    }

    _trsToMat4(t, q, s) {
      const [x, y, z, w] = q;
      const x2 = x + x, y2 = y + y, z2 = z + z;
      const xx = x * x2, xy = x * y2, xz = x * z2;
      const yy = y * y2, yz = y * z2, zz = z * z2;
      const wx = w * x2, wy = w * y2, wz = w * z2;
      const [sx, sy, sz] = s;
      return new Float32Array([
        (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
        (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
        (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
        t[0], t[1], t[2], 1,
      ]);
    }

    _eulerToQuat(rx, ry, rz) {
      const cx = Math.cos(rx * 0.5), sx = Math.sin(rx * 0.5);
      const cy = Math.cos(ry * 0.5), sy = Math.sin(ry * 0.5);
      const cz = Math.cos(rz * 0.5), sz = Math.sin(rz * 0.5);
      return [
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
      ];
    }

    _mulQuat(a, b) {
      return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
      ];
    }

    _updateNodeHierarchy(pose) {
      const headNodeIdx = this.boneNodes.head ?? this.boneNodes.Head;
      const neckNodeIdx = this.boneNodes.neck ?? this.boneNodes.Neck;
      const spineNodeIdx = this.boneNodes.spine ?? this.boneNodes.chest ?? this.boneNodes.upperChest;

      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        let local;
        if (Array.isArray(n.matrix) && n.matrix.length === 16) {
          local = new Float32Array(n.matrix);
        } else {
          const t = n.translation || [0, 0, 0];
          let q = n.rotation || [0, 0, 0, 1];
          const s = n.scale || [1, 1, 1];
          if (i === headNodeIdx) {
            q = this._mulQuat(q, this._eulerToQuat(pose.headPitch, pose.headYaw, pose.headRoll));
          } else if (i === neckNodeIdx) {
            q = this._mulQuat(q, this._eulerToQuat(pose.headPitch * 0.35, pose.headYaw * 0.35, 0));
          } else if (i === spineNodeIdx) {
            q = this._mulQuat(q, this._eulerToQuat(pose.spinePitch, 0, pose.headRoll * 0.25));
          }
          local = this._trsToMat4(t, q, s);
        }
        n.worldMatrix =
          n.parent != null && this.nodes[n.parent]
            ? this._mulMat4(this.nodes[n.parent].worldMatrix, local)
            : local;
      }
    }

    onResize() {
      if (!this.glCanvas || !this.containerEl) return;
      const dpr = window.devicePixelRatio || 1;
      const w = this.containerEl.clientWidth || window.innerWidth;
      const h = this.containerEl.clientHeight || window.innerHeight;
      this.glCanvas.width = Math.max(1, Math.floor(w * dpr));
      this.glCanvas.height = Math.max(1, Math.floor(h * dpr));
      if (this.gl) {
        this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
      }
    }

    setEmotion(emotionName) {
      this.currentEmotion = emotionName || "neutral";
    }

    playMotion() {}

    setLipSync(rms, visemes) {
      this.lipSyncRms = rms;
      if (visemes) this.visemes = visemes;
    }

    update(dt, life) {
      if (!this.gl || !this.program || !this.glCanvas) return;
      const gl = this.gl;
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      this._updateNodeHierarchy({
        headYaw: (life.gazeX * 0.32 + life.swayX * 0.12) * (this.yawOffset ? -1 : 1),
        headPitch: -life.gazeY * 0.22 + (life.breath - 0.5) * 0.04,
        headRoll: life.headTilt * 0.18,
        spinePitch: (life.breath - 0.5) * 0.03,
      });

      const cw = this.glCanvas.width;
      const ch = this.glCanvas.height;
      const aspect = cw / Math.max(1, ch);
      const fov = 30 * (Math.PI / 180);
      const f = 1.0 / Math.tan(fov * 0.5);
      const dist = (this.bboxRadius * 2.35) / Math.max(0.1, this.scale);
      const near = Math.max(0.01, dist * 0.05);
      const far = Math.max(100.0, dist * 8.0);

      const proj = new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) / (near - far), -1,
        0, 0, (2 * far * near) / (near - far), 0,
      ]);

      const rotY = this.yawOffset + this.userYaw + life.swayX * 0.08 + life.gazeX * 0.12;
      const cy = Math.cos(rotY), sy = Math.sin(rotY);
      const [bx, by, bz] = this.bboxCenter;
      const breathY = (life.breath - 0.5) * this.bboxRadius * 0.02;

      const rootTransform = new Float32Array([
        cy, 0, -sy, 0,
        0, 1, 0, 0,
        sy, 0, cy, 0,
        -bx * cy - bz * sy + this.posX,
        -by + breathY + this.posY,
        bx * sy - bz * cy - dist,
        1,
      ]);

      gl.useProgram(this.program);
      const locMvp = gl.getUniformLocation(this.program, "u_mvp");
      const locModel = gl.getUniformLocation(this.program, "u_model");
      const locMorph = gl.getUniformLocation(this.program, "u_morphWeights");
      const locBaseColor = gl.getUniformLocation(this.program, "u_baseColorFactor");
      const locHasTex = gl.getUniformLocation(this.program, "u_hasTex");
      const locTex = gl.getUniformLocation(this.program, "u_baseColorTex");

      const mouthWeight = Math.min(1.0, this.lipSyncRms * 1.35);
      const blinkWeight = life.blink;
      gl.uniform2f(locMorph, mouthWeight, blinkWeight);

      for (const dc of this.drawCalls) {
        const nodeWorld = this.nodes[dc.nodeIdx]?.worldMatrix || this._identityMat4();
        const modelMat = this._mulMat4(rootTransform, nodeWorld);
        const mvpMat = this._mulMat4(proj, modelMat);

        gl.uniformMatrix4fv(locModel, false, modelMat);
        gl.uniformMatrix4fv(locMvp, false, mvpMat);
        gl.uniform4fv(locBaseColor, dc.baseColorFactor);

        if (dc.texture) {
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, dc.texture);
          gl.uniform1i(locTex, 0);
          gl.uniform1i(locHasTex, 1);
        } else {
          gl.uniform1i(locHasTex, 0);
        }

        gl.bindVertexArray(dc.vao);
        if (dc.indexBuffer) {
          gl.drawElements(gl.TRIANGLES, dc.indexCount, gl.UNSIGNED_INT, 0);
        } else {
          gl.drawArrays(gl.TRIANGLES, 0, dc.indexCount);
        }
      }
      gl.bindVertexArray(null);
    }

    unload() {
      if (this.glCanvas && this.glCanvas.parentNode) {
        this.glCanvas.parentNode.removeChild(this.glCanvas);
      }
      this.glCanvas = null;
      this.gl = null;
      this.drawCalls = [];
      this.nodes = [];
    }
  }

  // ============================================================================
  // Part 3.1: Unified AvatarRuntimeManager (`IAvatarController` Orchestrator)
  // Bridges Cubism 3/4/5, Cubism 2.1, 3D VRM, 3D glTF, and 2D PNGTuber
  // ============================================================================
  class AvatarRuntimeManager {
    constructor() {
      this.activeFormat = "cubism3";
      this.activeController = null;
      this.modelInfo = null;
      this.lipSyncAnalyzer = new AudioLipSyncAnalyzer();
      this.lifeController = new ProceduralLifeController();
      this.rafId = null;
      this.lastTime = performance.now();
      this._initPointerTracking();
      this._startLoop();
    }

    detectFormat(modelInfo) {
      if (!modelInfo) return "cubism3";
      const explicit = String(modelInfo.format || "").toLowerCase();
      if (["cubism3", "cubism2", "vrm", "gltf", "mqo", "pngtuber"].includes(explicit)) {
        return explicit;
      }
      const url = String(modelInfo.url || "").toLowerCase();
      if (url.endsWith(".vrm")) return "vrm";
      if (url.endsWith(".glb") || url.endsWith(".gltf")) return "gltf";
      if (url.endsWith(".mqo")) return "mqo";
      if (url.endsWith(".model3.json")) return "cubism3";
      if (url.endsWith(".model.json")) return "cubism2";
      if (url.endsWith("pngtuber.json") || url.endsWith(".png") || url.endsWith(".webp")) {
        return "pngtuber";
      }
      return "cubism3";
    }

    async loadModel(modelInfo) {
      this.modelInfo = modelInfo;
      const fmt = this.detectFormat(modelInfo);
      this.activeFormat = fmt;

      if (this.activeController) {
        this.activeController.unload();
        this.activeController = null;
      }

      const wrapper =
        document.getElementById("live2d-internal-wrapper") || document.body;
      const cubismCanvas = document.getElementById("canvas");

      if (fmt === "cubism3") {
        if (cubismCanvas) cubismCanvas.style.opacity = "1";
        return false; // Let existing Cubism 3/4 pipeline load `.model3.json`, enhanced by our procedural loop
      }

      // Hide Cubism 3 canvas when rendering VRM / glTF / Cubism 2.1 / PNGTuber
      if (cubismCanvas) cubismCanvas.style.opacity = "0";
      if (window.LAppLive2DManager && window.LAppLive2DManager.releaseInstance) {
        try {
          window.LAppLive2DManager.releaseInstance();
        } catch (_) {}
      }

      try {
        if (fmt === "pngtuber") {
          this.activeController = new PNGTuberController(wrapper);
        } else if (fmt === "cubism2") {
          this.activeController = new Live2DCubism2Controller(wrapper);
        } else if (fmt === "vrm" || fmt === "gltf" || fmt === "mqo") {
          this.activeController = new VRMAndGLTF3DController(wrapper);
        }
        if (this.activeController && modelInfo?.url) {
          await this.activeController.load(modelInfo.url, modelInfo);
        }
      } catch (err) {
        console.error(`[AvatarRuntimeManager] Failed to load ${fmt} avatar:`, err);
      }
      return true;
    }

    setEmotion(emotionOrIndex) {
      const canonical = resolveCanonicalEmotion(emotionOrIndex, this.modelInfo);
      if (this.activeController) {
        this.activeController.setEmotion(canonical);
      }
    }

    attachAudio(audioEl) {
      this.lipSyncAnalyzer.attachAudio(audioEl);
    }

    stopAudio() {
      this.lipSyncAnalyzer.reset();
    }

    _initPointerTracking() {
      window.addEventListener(
        "pointermove",
        (e) => {
          const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
          const ny = -((e.clientY / Math.max(1, window.innerHeight)) * 2 - 1);
          this.lifeController.setPointerGaze(nx, ny);
        },
        { passive: true }
      );
      window.addEventListener(
        "resize",
        () => {
          if (this.activeController) this.activeController.onResize();
        },
        { passive: true }
      );
    }

    _startLoop() {
      const tick = (now) => {
        const dt = Math.min(0.1, Math.max(0.001, (now - this.lastTime) / 1000));
        this.lastTime = now;

        const life = this.lifeController.update(dt);
        const { rms, visemes } = this.lipSyncAnalyzer.sample(dt);

        if (this.activeController) {
          this.activeController.setLipSync(rms, visemes);
          this.activeController.update(dt, life);
        } else if (this.activeFormat === "cubism3") {
          // Enhance active Cubism 3/4 model with smooth pointer/saccade gaze when idle
          try {
            const mgr = window.getLive2DManager && window.getLive2DManager();
            const model = mgr && mgr.getModel && mgr.getModel(0);
            if (model && typeof model.setDragging === "function") {
              model.setDragging(life.gazeX * 0.45, life.gazeY * 0.45);
            }
          } catch (_) {}
        }

        this.rafId = requestAnimationFrame(tick);
      };
      this.rafId = requestAnimationFrame(tick);
    }
  }

  window.__olvAvatarRuntime = new AvatarRuntimeManager();
})();

