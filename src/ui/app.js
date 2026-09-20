// jepctl testbench client
// Zero-dependency pure vanilla JavaScript (ES6+)

(function () {
    "use strict";

    // Application State
    const state = {
        activeSection: "overview",
        hardwareInfo: null,
        activeModel: null,
        totalEmbeddings: 0,
        lastLatencyMs: 0,
        streamFps: 10,
        isStreaming: false,
        sseSource: null,
        webcamStream: null,
        nominalBaselineVector: null,
        alertThreshold: 0.45,
        webhookUrl: "",
        energyHistory: [],
        maxChartPoints: 50,
        audioContext: null,
        lastAlertSoundTime: 0,
        currentVector: null,
        patchGridSize: 14,
        authToken: localStorage.getItem("jepctl_auth_token") || "",
        gestureThreshold: 0.70,
        gestureMargin: 0.04,
        gestureSmoothing: 3,
        gestureHeatmap: true,
        gestureAudioEnabled: true,
        gestureHistory: {},
        activeGestureHold: null,
        holdStartTime: null,
        holdTriggered: false,
        slotHoldCounts: { 1: 0, 2: 0, 3: 0 },
        lastAudioToneTime: 0,
        registeredGestures: [],
        streamStarting: null,
        cameraRoi: null,
        roiTimer: null,
        roiObjectUrl: null,
        roiDrag: null,
        modelViewTimer: null,
        modelViewObjectUrl: null,
        modelViewSequence: null,
        lastGestureMatch: null,
    };

    const GESTURE_SLOT_COUNT = 4;
    const GESTURE_NEUTRAL_SLOT = 4;

    // DOM Elements Cache
    const el = {
        // Navigation
        navItems: document.querySelectorAll(".nav-item"),
        sections: document.querySelectorAll(".content-section"),
        
        // Header
        statusDot: document.getElementById("status-dot"),
        daemonStatusText: document.getElementById("daemon-status-text"),
        hardwareBadge: document.getElementById("hardware-badge"),
        hardwareName: document.getElementById("hardware-name"),
        memStats: document.getElementById("mem-stats"),
        memProgress: document.getElementById("mem-progress"),
        headerActiveModel: document.getElementById("header-active-model"),
        btnHeaderUnload: document.getElementById("btn-header-unload"),
        headerWeightsBadge: document.getElementById("header-weights-badge"),
        headerCameraChip: document.getElementById("header-camera-chip"),
        headerCameraText: document.getElementById("header-camera-text"),
        headerLatencyText: document.getElementById("header-latency-text"),
        toastRegion: document.getElementById("toast-region"),
        confirmDialog: document.getElementById("confirm-dialog"),
        confirmTitle: document.getElementById("confirm-title"),
        confirmMessage: document.getElementById("confirm-message"),
        confirmOk: document.getElementById("confirm-ok"),
        confirmCancel: document.getElementById("confirm-cancel"),
        apiDialog: document.getElementById("api-dialog"),
        apiDialogTitle: document.getElementById("api-dialog-title"),
        apiDialogDesc: document.getElementById("api-dialog-desc"),
        apiDialogCode: document.getElementById("api-dialog-code"),
        apiDialogClose: document.getElementById("api-dialog-close"),
        apiDialogCopy: document.getElementById("api-dialog-copy"),
        integrationBaseUrl: document.getElementById("integration-base-url"),
        integrationAuthMode: document.getElementById("integration-auth-mode"),
        integrationModel: document.getElementById("integration-model"),
        integrationExample: document.getElementById("integration-example"),
        btnCopyIntegration: document.getElementById("btn-copy-integration"),
        sseFilter: document.getElementById("sse-filter"),
        btnCopySse: document.getElementById("btn-copy-sse"),
        btnApiLoadModel: document.getElementById("btn-api-load-model"),
        btnApiMatch: document.getElementById("btn-api-match"),
        btnApiRegister: document.getElementById("btn-api-register"),
        btnGesturesExport: document.getElementById("btn-gestures-export"),
        roiCard: document.getElementById("roi-card"),
        roiStage: document.getElementById("roi-stage"),
        roiFullFrame: document.getElementById("roi-full-frame"),
        roiBox: document.getElementById("roi-box"),
        roiPlaceholder: document.getElementById("roi-placeholder"),
        roiValues: document.getElementById("roi-values"),
        roiStatusText: document.getElementById("roi-status-text"),
        btnRoiClear: document.getElementById("btn-roi-clear"),
        btnApiRoi: document.getElementById("btn-api-roi"),
        btnGesturesImport: document.getElementById("btn-gestures-import"),
        inputGesturesImport: document.getElementById("input-gestures-import"),

        // Overview
        kpiLatency: document.getElementById("kpi-latency"),
        kpiFps: document.getElementById("kpi-fps"),
        kpiTotalEmbeddings: document.getElementById("kpi-total-embeddings"),
        kpiUptime: document.getElementById("kpi-uptime"),
        telemetryDevice: document.getElementById("telemetry-device"),
        telemetryBackend: document.getElementById("telemetry-backend"),
        telemetryMemArch: document.getElementById("telemetry-mem-arch"),
        telemetryCpu: document.getElementById("telemetry-cpu"),
        telemetryPlatform: document.getElementById("telemetry-platform"),
        telemetryMem: document.getElementById("telemetry-mem"),
        btnQuickCamera: document.getElementById("btn-quick-camera"),
        btnQuickPullIjepa: document.getElementById("btn-quick-pull-ijepa"),
        btnQuickPullDinov2: document.getElementById("btn-quick-pull-dinov2"),
        verifiedGrid: document.getElementById("verified-grid"),
        appVersion: document.getElementById("app-version"),

        // Models
        pullRepoInput: document.getElementById("pull-repo-input"),
        btnStartPull: document.getElementById("btn-start-pull"),
        pullProgressBox: document.getElementById("pull-progress-box"),
        pullStatusText: document.getElementById("pull-status-text"),
        pullSpeedText: document.getElementById("pull-speed-text"),
        pullProgressFill: document.getElementById("pull-progress-fill"),
        installedModelsTbody: document.getElementById("installed-models-tbody"),
        jepafileJsonEditor: document.getElementById("jepafile-json-editor"),
        btnSaveJepafile: document.getElementById("btn-save-jepafile"),

        // Image Playground
        imageDropzone: document.getElementById("image-dropzone"),
        imageFileInput: document.getElementById("image-file-input"),
        canvasWrapper: document.getElementById("canvas-wrapper"),
        imageInspectCanvas: document.getElementById("image-inspect-canvas"),
        togglePatchGrid: document.getElementById("toggle-patch-grid"),
        outModelName: document.getElementById("out-model-name"),
        outEmbedDim: document.getElementById("out-embed-dim"),
        outEmbedLatency: document.getElementById("out-embed-latency"),
        heatmapCanvas: document.getElementById("heatmap-canvas"),
        vectorNumericView: document.getElementById("vector-numeric-view"),
        btnCopyVector: document.getElementById("btn-copy-vector"),
        btnExportJson: document.getElementById("btn-export-json"),

        // Video & Camera Stream
        cameraSelectWrap: document.getElementById("camera-select-wrap"),
        cameraDeviceSelect: document.getElementById("camera-device-select"),
        streamFpsSelect: document.getElementById("stream-fps-select"),
        btnStreamToggle: document.getElementById("btn-stream-toggle"),
        scrubberStripContainer: document.getElementById("scrubber-strip-container"),
        bufferCountBadge: document.getElementById("buffer-count-badge"),
        webcamPreviewElement: document.getElementById("webcam-preview-element"),
        streamFeedCanvas: document.getElementById("stream-feed-canvas"),
        previewPlaceholder: document.getElementById("preview-placeholder"),
        sseStreamLog: document.getElementById("sse-stream-log"),
        btnClearSse: document.getElementById("btn-clear-sse"),

        // Anomaly & Energy
        btnLockBaseline: document.getElementById("btn-lock-baseline"),
        baselineStatusLabel: document.getElementById("baseline-status-label"),
        sliderThreshold: document.getElementById("slider-threshold"),
        valThreshold: document.getElementById("val-threshold"),
        inputWebhookUrl: document.getElementById("input-webhook-url"),
        btnSaveWebhook: document.getElementById("btn-save-webhook"),
        energyChartCanvas: document.getElementById("energy-chart-canvas"),
        anomalyAlertBanner: document.getElementById("anomaly-alert-banner"),

        // Security & Keys
        toggleLanAccess: document.getElementById("toggle-lan-access"),
        lanWarningBox: document.getElementById("lan-warning-box"),
        tableApiKeys: document.getElementById("table-api-keys"),
        apiKeysTbody: document.getElementById("api-keys-tbody"),
        btnOpenCreateKey: document.getElementById("btn-open-create-key"),
        btnRefreshAudit: document.getElementById("btn-refresh-audit"),
        auditLogTbody: document.getElementById("audit-log-tbody"),
        createKeyModal: document.getElementById("create-key-modal"),
        modalKeyName: document.getElementById("modal-key-name"),
        modalKeyRole: document.getElementById("modal-key-role"),
        modalKeyExpire: document.getElementById("modal-key-expire"),
        btnCloseKeyModal: document.getElementById("btn-close-key-modal"),
        btnSubmitCreateKey: document.getElementById("btn-submit-create-key"),
        generatedTokenDisplay: document.getElementById("generated-token-display"),
        rawTokenValue: document.getElementById("raw-token-value"),
        btnCopyRawToken: document.getElementById("btn-copy-raw-token"),

        // Settings
        settingsBackend: document.getElementById("settings-backend"),
        settingsMemWatermark: document.getElementById("settings-mem-watermark"),
        settingsIdleTimeout: document.getElementById("settings-idle-timeout"),
        settingsStorageDir: document.getElementById("settings-storage-dir"),
        btnSaveSettings: document.getElementById("btn-save-settings"),

        // Gesture Sandbox
        gestureModelView: document.getElementById("gesture-model-view"),
        gestureHeatmap: document.getElementById("gesture-heatmap"),
        gestureViewPlaceholder: document.getElementById("gesture-view-placeholder"),
        toggleGestureHeatmap: document.getElementById("toggle-gesture-heatmap"),
        btnGestureCameraToggle: document.getElementById("btn-gesture-camera-toggle"),
        gestureCamBtnText: document.getElementById("gesture-cam-btn-text"),
        gestureDetectionBadge: document.getElementById("gesture-detection-badge"),
        gestureBadgeText: document.getElementById("gesture-badge-text"),
        gestureConfidenceText: document.getElementById("gesture-confidence-text"),
        gestureConfidenceBar: document.getElementById("gesture-confidence-bar"),
        sliderGestureThreshold: document.getElementById("slider-gesture-threshold"),
        valGestureThreshold: document.getElementById("val-gesture-threshold"),
        sliderGestureMargin: document.getElementById("slider-gesture-margin"),
        valGestureMargin: document.getElementById("val-gesture-margin"),
        sliderGestureSmoothing: document.getElementById("slider-gesture-smoothing"),
        valGestureSmoothing: document.getElementById("val-gesture-smoothing"),
        toggleGestureAudio: document.getElementById("toggle-gesture-audio"),
        gestureThemePill: document.getElementById("gesture-theme-pill"),
        gestureThemeText: document.getElementById("gesture-theme-text"),
        gestureHoldTimerText: document.getElementById("gesture-hold-timer-text"),
        gestureHoldBar: document.getElementById("gesture-hold-bar"),
        gestureCountBadge: document.getElementById("gesture-count-badge"),
        gestureVideoWrapper: document.getElementById("gesture-video-wrapper"),
        selectGestureActiveModel: document.getElementById("select-gesture-active-model"),
        btnGestureLoadModel: document.getElementById("btn-gesture-load-model"),
        btnGestureUnloadModel: document.getElementById("btn-gesture-unload-model"),
        gestureModelIndicator: document.getElementById("gesture-model-indicator"),
        gestureModelNameText: document.getElementById("gesture-model-name-text"),
        gestureWeightsBadge: document.getElementById("gesture-weights-badge"),
        gestureMethodBadge: document.getElementById("gesture-method-badge"),
        gestureReasoningBody: document.getElementById("gesture-reasoning-body"),
        gestureReasoningDecision: document.getElementById("gesture-reasoning-decision"),
        reasonMargin: document.getElementById("reason-margin"),
        reasonThreshold: document.getElementById("reason-threshold"),
        reasonLatency: document.getElementById("reason-latency"),
        reasonFrame: document.getElementById("reason-frame"),
        reasonGrid: document.getElementById("reason-grid"),
        btnGesturesClear: document.getElementById("btn-gestures-clear"),
    };

    // Loopback Session Token Management & Authenticated Fetch
    async function fetchSessionToken() {
        try {
            const res = await fetch("/api/auth/token");
            if (res.ok) {
                const data = await res.json();
                if (data.token) {
                    state.authToken = data.token;
                    localStorage.setItem("jepctl_auth_token", data.token);
                    return data.token;
                }
            }
        } catch (e) {
            console.warn("Session token acquisition failed:", e);
        }
        return state.authToken;
    }

    // ------------------------------------------------------------------
    // Feedback primitives: toasts (non-blocking), accessible confirm dialog,
    // and the API request viewer. Replaces window.alert/confirm everywhere.
    // ------------------------------------------------------------------

    function notify(message, kind = "info", timeoutMs = 5000) {
        if (!el.toastRegion) {
            console[kind === "error" ? "error" : "log"](message);
            return;
        }
        while (el.toastRegion.children.length >= 3) {
            el.toastRegion.removeChild(el.toastRegion.firstChild);
        }
        const toast = document.createElement("div");
        toast.className = `toast ${kind}`;
        toast.setAttribute("role", kind === "error" ? "alert" : "status");
        const icons = { success: "\u2713", error: "\u2717", warning: "!", info: "i" };
        toast.innerHTML = `<span class="toast-icon" aria-hidden="true">${icons[kind] || "i"}</span><span class="toast-msg"></span><button class="toast-close" aria-label="Dismiss">\u00d7</button>`;
        toast.querySelector(".toast-msg").textContent = message;
        const remove = () => { if (toast.parentNode) toast.parentNode.removeChild(toast); };
        toast.querySelector(".toast-close").addEventListener("click", remove);
        el.toastRegion.appendChild(toast);
        if (timeoutMs > 0) setTimeout(remove, kind === "error" ? Math.max(timeoutMs, 8000) : timeoutMs);
    }

    // Focus management shared by every dialog: trap Tab, close on Escape, restore focus.
    function openDialog(overlay, { initialFocus, onClose } = {}) {
        const previouslyFocused = document.activeElement;
        overlay.style.display = "flex";
        const focusables = () => [...overlay.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")].filter((n) => !n.disabled && n.offsetParent !== null);
        const onKey = (e) => {
            if (e.key === "Escape") {
                e.preventDefault();
                close();
            } else if (e.key === "Tab") {
                const f = focusables();
                if (!f.length) return;
                const first = f[0], last = f[f.length - 1];
                if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
                else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
            }
        };
        const onBackdrop = (e) => { if (e.target === overlay) close(); };
        function close() {
            overlay.style.display = "none";
            overlay.removeEventListener("keydown", onKey);
            overlay.removeEventListener("click", onBackdrop);
            if (onClose) onClose();
            if (previouslyFocused && previouslyFocused.focus) previouslyFocused.focus();
        }
        overlay.addEventListener("keydown", onKey);
        overlay.addEventListener("click", onBackdrop);
        (initialFocus || focusables()[0] || overlay).focus();
        return close;
    }

    function confirmDialog(message, { title = "Confirm", okLabel = "Confirm", danger = true } = {}) {
        if (!el.confirmDialog) return Promise.resolve(window.confirm(message));
        return new Promise((resolve) => {
            el.confirmTitle.textContent = title;
            el.confirmMessage.textContent = message;
            el.confirmOk.textContent = okLabel;
            el.confirmOk.className = danger ? "btn btn-danger" : "btn btn-primary";
            let settled = false;
            const close = openDialog(el.confirmDialog, { initialFocus: el.confirmCancel, onClose: () => { if (!settled) { settled = true; resolve(false); } } });
            const ok = () => { settled = true; cleanup(); close(); resolve(true); };
            const cancel = () => { settled = true; cleanup(); close(); resolve(false); };
            function cleanup() {
                el.confirmOk.removeEventListener("click", ok);
                el.confirmCancel.removeEventListener("click", cancel);
            }
            el.confirmOk.addEventListener("click", ok);
            el.confirmCancel.addEventListener("click", cancel);
        });
    }

    async function copyText(text, label = "Copied to clipboard") {
        try {
            await navigator.clipboard.writeText(text);
            notify(label, "success", 2500);
        } catch (e) {
            notify(`Copy failed: ${e.message}`, "error");
        }
    }

    // ------------------------------------------------------------------
    // API snippets: the exact request behind a UI action, in three languages.
    // ------------------------------------------------------------------

    function apiBaseUrl() {
        return window.location.origin;
    }

    function tokenPlaceholder() {
        return state.authToken === "no_auth" ? null : "$JEPA_TOKEN";
    }

    // request = { method, path, json?, multipartFile?, query? }
    function renderSnippet(lang, request) {
        const url = `${apiBaseUrl()}${request.path}`;
        const token = tokenPlaceholder();
        const bodyJson = request.json !== undefined ? JSON.stringify(request.json, null, 2) : null;

        if (lang === "curl") {
            const lines = [`curl -s -X ${request.method} "${url}" \\`];
            if (token) lines.push(`  -H "Authorization: Bearer ${token}" \\`);
            if (request.multipartFile) {
                lines.push(`  -F "file=@${request.multipartFile}"`);
            } else if (bodyJson) {
                lines.push(`  -H "Content-Type: application/json" \\`);
                lines.push(`  -d '${bodyJson.replace(/'/g, "'\\''")}'`);
            } else {
                lines[lines.length - 1] = lines[lines.length - 1].replace(/ \\$/, "");
            }
            return lines.join("\n");
        }

        if (lang === "js") {
            const headers = [];
            if (token) headers.push(`    "Authorization": \`Bearer \${process.env.JEPA_TOKEN}\``);
            if (bodyJson && !request.multipartFile) headers.push(`    "Content-Type": "application/json"`);
            let body = "";
            if (request.multipartFile) {
                body = `\nconst form = new FormData();\nform.append("file", fileBlob, "${request.multipartFile}");\n`;
            }
            return `${body}const res = await fetch("${url}", {\n  method: "${request.method}",\n  headers: {\n${headers.join(",\n")}\n  }${request.multipartFile ? ",\n  body: form" : bodyJson ? `,\n  body: JSON.stringify(${bodyJson.replace(/\n/g, "\n  ")})` : ""}\n});\nif (!res.ok) throw new Error(\`\${res.status} \${await res.text()}\`);\nconst data = await res.json();\nconsole.log(data);`;
        }

        // python
        const headers = [];
        if (token) headers.push(`"Authorization": f"Bearer {os.environ['JEPA_TOKEN']}"`);
        let call;
        if (request.multipartFile) {
            call = `requests.${request.method.toLowerCase()}(url, headers=headers, files={"file": open("${request.multipartFile}", "rb")})`;
        } else if (bodyJson) {
            call = `requests.${request.method.toLowerCase()}(url, headers=headers, json=${bodyJson.replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None")})`;
        } else {
            call = `requests.${request.method.toLowerCase()}(url, headers=headers)`;
        }
        return `import os, requests\n\nurl = "${url}"\nheaders = {${headers.join(", ")}}\nres = ${call}\nres.raise_for_status()\nprint(res.json())`;
    }

    let currentApiRequest = null;
    let currentApiLang = localStorage.getItem("jepctl_snippet_lang") || "curl";

    function showApiDialog(title, description, request) {
        if (!el.apiDialog) return;
        currentApiRequest = request;
        el.apiDialogTitle.textContent = title;
        el.apiDialogDesc.textContent = description;
        renderApiDialogCode();
        openDialog(el.apiDialog, { initialFocus: el.apiDialogClose });
    }

    function renderApiDialogCode() {
        if (!currentApiRequest) return;
        el.apiDialogCode.textContent = renderSnippet(currentApiLang, currentApiRequest);
        el.apiDialog.querySelectorAll(".snippet-lang").forEach((b) => {
            const on = b.dataset.lang === currentApiLang;
            b.classList.toggle("active", on);
            b.setAttribute("aria-selected", on ? "true" : "false");
        });
    }

    function setupApiDialog() {
        if (!el.apiDialog) return;
        el.apiDialog.querySelectorAll(".snippet-lang").forEach((b) => {
            b.addEventListener("click", () => {
                currentApiLang = b.dataset.lang;
                localStorage.setItem("jepctl_snippet_lang", currentApiLang);
                renderApiDialogCode();
                renderIntegrationExample();
            });
        });
        el.apiDialogCopy.addEventListener("click", () => copyText(el.apiDialogCode.textContent, "Snippet copied"));
        el.apiDialogClose.addEventListener("click", () => { el.apiDialog.style.display = "none"; });
    }

    function gestureMatchRequest() {
        return {
            method: "POST",
            path: "/api/gestures/match",
            json: { image_base64: "data:image/jpeg;base64,<...>", threshold: Number(state.gestureThreshold.toFixed(2)), margin: Number(state.gestureMargin.toFixed(2)) }
        };
    }

    function renderIntegrationExample() {
        if (!el.integrationExample) return;
        el.integrationBaseUrl.textContent = apiBaseUrl();
        el.integrationAuthMode.textContent = state.authToken === "no_auth" ? "disabled (--no-auth, loopback only)" : "Bearer token (inference or admin role)";
        el.integrationModel.textContent = state.activeModel || "none: load one in Models";
        const req = state.registeredGestures.length ? gestureMatchRequest() : { method: "POST", path: "/api/embed", multipartFile: "photo.jpg" };
        el.integrationExample.textContent = renderSnippet(currentApiLang, req);
        document.querySelectorAll("#section-security .snippet-lang").forEach((b) => {
            const on = b.dataset.lang === currentApiLang;
            b.classList.toggle("active", on);
            b.setAttribute("aria-selected", on ? "true" : "false");
        });
    }

    function setupIntegration() {
        document.querySelectorAll("#section-security .snippet-lang").forEach((b) => {
            b.addEventListener("click", () => {
                currentApiLang = b.dataset.lang;
                localStorage.setItem("jepctl_snippet_lang", currentApiLang);
                renderIntegrationExample();
            });
        });
        if (el.btnCopyIntegration) {
            el.btnCopyIntegration.addEventListener("click", () => copyText(el.integrationExample.textContent, "Example copied"));
        }
    }

    async function apiFetch(url, options = {}) {
        const opts = Object.assign({}, options);
        opts.headers = Object.assign({}, opts.headers);

        if (!state.authToken) {
            await fetchSessionToken();
        }

        if (state.authToken && state.authToken !== "no_auth" && !opts.headers["Authorization"]) {
            opts.headers["Authorization"] = `Bearer ${state.authToken}`;
        }

        let res = await fetch(url, opts);
        if (res.status === 401) {
            // Re-fetch token and retry once
            await fetchSessionToken();
            if (state.authToken && state.authToken !== "no_auth") {
                opts.headers["Authorization"] = `Bearer ${state.authToken}`;
                res = await fetch(url, opts);
            }
        }
        return res;
    }

    // Initialize Application
    async function init() {
        await fetchSessionToken();

        setupNavigation();
        setupApiDialog();
        setupIntegration();
        setupImagePlayground();
        setupVideoStream();
        setupAnomalyMonitor();
        setupSecurityAndKeys();
        setupSettings();
        setupGestureSandbox();
        setupRoiEditor();
        setupRobotTwin();
        setupHeader();

        // Initial fetch
        pollStatus();
        fetchModels();
        fetchCameras();
        fetchApiKeys();
        fetchAuditLog();
        fetchGesturesList();

        // Start Periodic Polling (every 2 seconds)
        setInterval(pollStatus, 2000);
        setInterval(refreshRingBufferThumbnails, 1000);

        // Render empty chart
        renderEnergyChart();
    }

    // Navigation Switcher
    function setupNavigation() {
        const items = [...el.navItems];
        items.forEach((item, idx) => {
            item.addEventListener("click", () => switchSection(item.getAttribute("data-section")));
            item.addEventListener("keydown", (e) => {
                let next = null;
                if (e.key === "ArrowDown") next = items[(idx + 1) % items.length];
                if (e.key === "ArrowUp") next = items[(idx - 1 + items.length) % items.length];
                if (e.key === "Home") next = items[0];
                if (e.key === "End") next = items[items.length - 1];
                if (next) {
                    e.preventDefault();
                    next.focus();
                    switchSection(next.getAttribute("data-section"));
                }
            });
        });

        // Quick action buttons
        if (el.btnQuickCamera) {
            el.btnQuickCamera.addEventListener("click", () => switchSection("video-stream"));
        }
        if (el.btnQuickPullIjepa) {
            el.btnQuickPullIjepa.addEventListener("click", () => pullModel("facebook/ijepa_vith14_1k"));
        }
        if (el.btnQuickPullDinov2) {
            el.btnQuickPullDinov2.addEventListener("click", () => pullModel("facebook/dinov2-small"));
        }
    }

    function switchSection(sectionId) {
        if (state.activeSection === "robot" && sectionId !== "robot") robotLeaveSection();
        state.activeSection = sectionId;

        el.navItems.forEach((item) => {
            const on = item.getAttribute("data-section") === sectionId;
            item.classList.toggle("active", on);
            item.setAttribute("aria-selected", on ? "true" : "false");
            item.tabIndex = on ? 0 : -1;
        });

        el.sections.forEach((sec) => {
            if (sec.id === `section-${sectionId}`) {
                sec.classList.add("active");
            } else {
                sec.classList.remove("active");
            }
        });

        if (sectionId === "models") {
            fetchModels();
        } else if (sectionId === "security") {
            fetchApiKeys();
            fetchAuditLog();
            renderIntegrationExample();
        } else if (sectionId === "gestures") {
            fetchGesturesList();
        } else if (sectionId === "robot") {
            robotEnterSection();
        }
    }

    // Header Controls
    function setupHeader() {
        if (el.btnHeaderUnload) {
            el.btnHeaderUnload.addEventListener("click", async () => {
                try {
                    await apiFetch("/api/models/unload", { method: "POST" });
                    pollStatus();
                    fetchModels();
                } catch (e) {
                    console.error("Failed to unload model:", e);
                }
            });
        }
    }

    // Status and Telemetry Polling
    async function pollStatus() {
        try {
            const res = await apiFetch("/api/status");
            if (!res.ok) throw new Error("Status endpoint error");
            const data = await res.json();

            // Update Header Status
            el.statusDot.style.backgroundColor = "var(--accent-green)";
            el.statusDot.style.boxShadow = "0 0 8px var(--accent-green)";
            el.daemonStatusText.textContent = "Daemon: Connected";

            // Hardware Telemetry
            const hw = data.hardware;
            state.hardwareInfo = hw;
            el.hardwareName.textContent = hw.device_name;
            
            const usedGb = (hw.memory_used_bytes / (1024 * 1024 * 1024)).toFixed(1);
            const totalGb = (hw.memory_total_bytes / (1024 * 1024 * 1024)).toFixed(1);
            const percent = hw.memory_percent.toFixed(0);

            el.memStats.textContent = `${usedGb} GB / ${totalGb} GB (${percent}%)`;
            el.memProgress.style.width = `${percent}%`;

            if (el.appVersion && data.version) el.appVersion.textContent = `v${data.version}`;
            if (el.headerCameraChip) {
                const on = !!data.camera_active;
                el.headerCameraChip.classList.toggle("on", on);
                el.headerCameraText.textContent = on ? `Camera ${state.streamFps || 10} fps` : "Camera off";
            }
            if (el.integrationModel) el.integrationModel.textContent = data.active_model || "none: load one in Models";

            // Active Model
            const modelChanged = state.activeModel !== data.active_model;
            state.activeModel = data.active_model;
            updateWeightsBadge(data.weights, data.active_model);
            if (modelChanged) {
                fetchGesturesList();
            }
            if (data.active_model) {
                el.headerActiveModel.textContent = data.active_model;
                el.btnHeaderUnload.style.display = "inline-flex";
                if (el.headerWeightsBadge && data.weights) {
                    const ok = data.weights.loaded === data.weights.expected && data.weights.expected > 0;
                    el.headerWeightsBadge.textContent = `${data.weights.loaded}/${data.weights.expected}`;
                    el.headerWeightsBadge.className = `weights-badge ${ok ? "ok" : "bad"}`;
                }
                if (el.gestureModelIndicator) el.gestureModelIndicator.className = "status-indicator status-active";
                if (el.gestureModelNameText) el.gestureModelNameText.textContent = data.active_model;
                if (el.selectGestureActiveModel && el.selectGestureActiveModel.value !== data.active_model) {
                    // Only update if option exists
                    for (let i = 0; i < el.selectGestureActiveModel.options.length; i++) {
                        if (el.selectGestureActiveModel.options[i].value === data.active_model) {
                            el.selectGestureActiveModel.selectedIndex = i;
                            break;
                        }
                    }
                }
            } else {
                el.headerActiveModel.textContent = "none loaded";
                el.btnHeaderUnload.style.display = "none";
                if (el.headerWeightsBadge) {
                    el.headerWeightsBadge.textContent = "--";
                    el.headerWeightsBadge.className = "weights-badge";
                }
                if (el.gestureModelIndicator) el.gestureModelIndicator.className = "status-indicator status-idle";
                if (el.gestureModelNameText) el.gestureModelNameText.textContent = "No model loaded";
            }

            // Overview KPIs
            state.totalEmbeddings = data.embeddings_computed_total;
            el.kpiTotalEmbeddings.textContent = data.embeddings_computed_total.toLocaleString();
            el.kpiUptime.textContent = formatUptime(data.uptime_seconds);

            el.telemetryDevice.textContent = hw.device_name;
            if (el.telemetryBackend) {
                if (hw.backend === "Metal") {
                    el.telemetryBackend.innerHTML = '<span style="color: #4ade80; font-weight: 600;">Apple Metal GPU (Hardware Accelerated)</span>';
                    if (el.hardwareBadge) {
                        el.hardwareBadge.style.borderColor = "rgba(74, 222, 128, 0.5)";
                        el.hardwareBadge.style.backgroundColor = "rgba(74, 222, 128, 0.1)";
                    }
                } else if (hw.backend === "Cuda") {
                    el.telemetryBackend.innerHTML = '<span style="color: #4ade80; font-weight: 600;">NVIDIA CUDA GPU (Hardware Accelerated)</span>';
                    if (el.hardwareBadge) {
                        el.hardwareBadge.style.borderColor = "rgba(74, 222, 128, 0.5)";
                        el.hardwareBadge.style.backgroundColor = "rgba(74, 222, 128, 0.1)";
                    }
                } else {
                    el.telemetryBackend.innerHTML = '<span style="color: #facc15; font-weight: 600;">CPU Multithreaded (Fallback Mode)</span>';
                    if (el.hardwareBadge) {
                        el.hardwareBadge.style.borderColor = "rgba(250, 204, 21, 0.4)";
                        el.hardwareBadge.style.backgroundColor = "rgba(250, 204, 21, 0.1)";
                    }
                }
            }
            if (el.telemetryMemArch) {
                if (hw.backend === "Metal") {
                    el.telemetryMemArch.textContent = "Apple Unified Memory Architecture (UMA: CPU and GPU Shared)";
                } else {
                    el.telemetryMemArch.textContent = "Standard Host Memory";
                }
            }
            el.telemetryCpu.textContent = `${hw.cpu_threads} Logical Cores`;
            el.telemetryPlatform.textContent = data.platform;
            el.telemetryMem.textContent = `${usedGb} GB used of ${totalGb} GB total`;

        } catch (err) {
            el.statusDot.style.backgroundColor = "var(--accent-red)";
            el.statusDot.style.boxShadow = "0 0 8px var(--accent-red)";
            el.daemonStatusText.textContent = "Daemon: Disconnected";
        }
    }

    function formatUptime(seconds) {
        const h = Math.floor(seconds / 3600).toString().padStart(2, "0");
        const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
        const s = (seconds % 60).toString().padStart(2, "0");
        return `${h}:${m}:${s}`;
    }

    // Models Catalog Manager
    async function fetchModels() {
        try {
            const res = await apiFetch("/api/tags");
            if (!res.ok) throw new Error("Failed to load models");
            const models = await res.json();
            renderModelsTable(models);
            updateGestureModelSelector(models);
            await renderVerifiedCatalog(models);
        } catch (e) {
            console.error("fetchModels error:", e);
        }
    }

    // The verified catalog comes from the server so the UI can never advertise a model
    // the engine does not actually load.
    async function renderVerifiedCatalog(installed) {
        if (!el.verifiedGrid) return;
        try {
            const res = await apiFetch("/api/catalog");
            if (!res.ok) return;
            const catalog = await res.json();
            const installedNames = new Set((installed || []).filter((m) => m.disk_size_bytes > 0).map((m) => m.name));
            el.verifiedGrid.innerHTML = "";
            catalog.forEach((m) => {
                const card = document.createElement("div");
                card.className = "verified-card";
                const org = m.name.split("/")[0];
                const isInstalled = installedNames.has(m.name);
                const sizeGb = m.disk_size_bytes ? (m.disk_size_bytes / 1e9).toFixed(2) : "?";
                card.innerHTML = `
                    <div class="badge-row">
                        <span class="badge badge-image">${escapeHtml(org)}</span>
                        <span class="badge badge-dim">${m.embed_dim} dims</span>
                        <span class="badge badge-dim">${escapeHtml(m.modality || "image")}</span>
                        <span class="badge badge-dim">${escapeHtml(m.variant || "plain")}</span>
                    </div>
                    <h4>${escapeHtml(m.name)}</h4>
                    <p>${escapeHtml(m.architecture)} · ${escapeHtml(m.parameter_count)} · ${m.image_size}px · ${escapeHtml(m.normalization || "imagenet")} norm · ${sizeGb} GB</p>
                    <div class="card-action"></div>`;
                const action = card.querySelector(".card-action");
                const btn = document.createElement("button");
                btn.className = isInstalled ? "btn btn-sm btn-outline" : "btn btn-sm btn-primary";
                btn.textContent = isInstalled ? "Installed" : "Pull Checkpoint";
                btn.disabled = isInstalled;
                btn.addEventListener("click", () => pullModel(m.name));
                action.appendChild(btn);
                el.verifiedGrid.appendChild(card);
            });
        } catch (e) {
            console.error("renderVerifiedCatalog error:", e);
        }
    }

    function updateGestureModelSelector(models) {
        if (!el.selectGestureActiveModel) return;
        el.selectGestureActiveModel.innerHTML = "";

        if (!models || models.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "No model installed: pull one in Models";
            el.selectGestureActiveModel.appendChild(opt);
        } else {
            models.forEach((m) => {
                const opt = document.createElement("option");
                opt.value = m.name;
                opt.textContent = `${m.name} (${m.architecture} | ${m.embed_dim}d | ${m.parameter_count || 'ViT'})`;
                if (state.activeModel === m.name) {
                    opt.selected = true;
                }
                el.selectGestureActiveModel.appendChild(opt);
            });
        }

        if (el.gestureModelIndicator && el.gestureModelNameText) {
            if (state.activeModel) {
                el.gestureModelIndicator.className = "status-indicator status-active";
                el.gestureModelNameText.textContent = state.activeModel;
            } else {
                el.gestureModelIndicator.className = "status-indicator status-idle";
                el.gestureModelNameText.textContent = "No model loaded";
            }
        }
    }

    function renderModelsTable(models) {
        if (!el.installedModelsTbody) return;
        el.installedModelsTbody.innerHTML = "";

        if (models.length === 0) {
            el.installedModelsTbody.innerHTML = `<tr><td colspan="6" class="text-center" style="color: var(--text-dim);">Aucun modele installe localement. Telechargez un checkpoint verifie ci-dessus.</td></tr>`;
            return;
        }

        models.forEach((m) => {
            const tr = document.createElement("tr");
            const isLoaded = state.activeModel === m.name;
            const sizeMb = (m.disk_size_bytes / (1024 * 1024)).toFixed(1);

            tr.innerHTML = `
                <td><strong>${m.name}</strong> ${isLoaded ? '<span class="badge badge-image">Active</span>' : ''}</td>
                <td><span class="badge badge-${m.modality}">${m.modality}</span></td>
                <td><code>${m.embed_dim}</code></td>
                <td>${m.parameter_count}</td>
                <td>${sizeMb} MB</td>
                <td>
                    ${isLoaded ? 
                        `<button class="btn btn-sm btn-outline" data-action="unload" data-name="${m.name}">Decharger</button>` :
                        `<button class="btn btn-sm btn-primary" data-action="load" data-name="${m.name}">Charger</button>`
                    }
                    <button class="btn btn-sm btn-danger" data-action="delete" data-name="${m.name}" style="margin-left: 6px;">Supprimer</button>
                </td>
            `;

            tr.querySelectorAll("button").forEach((btn) => {
                btn.addEventListener("click", () => {
                    const action = btn.getAttribute("data-action");
                    const name = btn.getAttribute("data-name");
                    if (action === "load") loadModel(name);
                    else if (action === "unload") unloadModel();
                    else if (action === "delete") deleteModel(name);
                });
            });

            el.installedModelsTbody.appendChild(tr);
        });
    }

    async function loadModel(name) {
        try {
            const res = await apiFetch("/api/models/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: name })
            });
            if (res.ok) {
                const data = await res.json().catch(() => ({}));
                const w = data.weights;
                notify(w ? `${name} loaded: ${w.loaded}/${w.expected} tensors from ${w.source}` : `${name} loaded`, "success");
                await pollStatus();
                await fetchModels();
            } else {
                const err = await res.json().catch(() => ({}));
                notify(`Could not load model: ${err.error || "unknown error"}`, "error");
            }
        } catch (e) {
            console.error("Load model error:", e);
            notify(`Network error: ${e.message}`, "error");
        }
    }

    async function unloadModel() {
        try {
            await apiFetch("/api/models/unload", { method: "POST" });
            notify("Model unloaded", "info", 2500);
            await pollStatus();
            await fetchModels();
        } catch (e) {
            console.error("Unload error:", e);
        }
    }

    async function deleteModel(name) {
        if (!(await confirmDialog(`Delete model "${name}" from local storage? The checkpoint will have to be pulled again.`, { title: "Delete model", okLabel: "Delete" }))) return;
        try {
            let res = await apiFetch(`/api/models?name=${encodeURIComponent(name)}`, { method: "DELETE" });
            if (!res.ok) {
                res = await apiFetch(`/api/models/${encodeURIComponent(name)}`, { method: "DELETE" });
            }
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                notify(`Delete failed: ${err.error || "unknown error"}`, "error");
            } else {
                console.log(`Modele ${name} supprime avec succes.`);
            }
            await pollStatus();
            await fetchModels();
        } catch (e) {
            console.error("Delete error:", e);
            notify(`Network error: ${e.message}`, "error");
        }
    }

    // Gesture Sandbox Quick Model Load/Unload
    if (el.btnGestureLoadModel) {
        el.btnGestureLoadModel.addEventListener("click", () => {
            const chosen = el.selectGestureActiveModel ? el.selectGestureActiveModel.value : null;
            if (chosen) {
                loadModel(chosen);
            } else {
                notify("Select an installed model first.", "warning");
            }
        });
    }

    if (el.btnGestureUnloadModel) {
        el.btnGestureUnloadModel.addEventListener("click", () => {
            unloadModel();
        });
    }

    // Pull Model
    if (el.btnStartPull) {
        el.btnStartPull.addEventListener("click", () => {
            const repo = el.pullRepoInput.value.trim();
            if (repo) pullModel(repo);
        });
    }

    document.querySelectorAll("[data-pull]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const repo = btn.getAttribute("data-pull");
            pullModel(repo);
        });
    });

    async function pullModel(repoId) {
        el.pullProgressBox.style.display = "block";
        el.pullStatusText.textContent = `Initiating pull for ${repoId}...`;
        el.pullSpeedText.textContent = "0.0 MB/s";
        el.pullProgressFill.style.width = "0%";

        try {
            const res = await apiFetch("/api/pull", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repo_id: repoId })
            });

            if (!res.ok) {
                let errMsg = `Failed to start pull (HTTP ${res.status})`;
                try {
                    const err = await res.json();
                    if (err && err.error) errMsg = err.error;
                } catch (_) {}
                throw new Error(errMsg);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split("\n").filter((l) => l.trim().length > 0);

                for (const line of lines) {
                    try {
                        const event = JSON.parse(line);
                        if (event.status === "error") {
                            el.pullStatusText.textContent = `Error: ${event.error || "Download failed"}`;
                            return;
                        }

                        el.pullStatusText.textContent = `Downloading ${event.repo_id}: ${event.percentage.toFixed(1)}%`;
                        el.pullSpeedText.textContent = `${event.speed_mb_s.toFixed(1)} MB/s`;
                        el.pullProgressFill.style.width = `${event.percentage}%`;

                        if (event.finished) {
                            el.pullStatusText.textContent = `Successfully pulled ${event.repo_id}`;
                            setTimeout(() => {
                                el.pullProgressBox.style.display = "none";
                                fetchModels();
                                pollStatus();
                            }, 1500);
                        }
                    } catch (err) {
                        // ignore unparseable chunk
                    }
                }
            }
        } catch (e) {
            el.pullStatusText.textContent = `Error: ${e.message}`;
        }
    }

    // Custom Jepafile Registration
    if (el.btnSaveJepafile) {
        el.btnSaveJepafile.addEventListener("click", async () => {
            try {
                const raw = el.jepafileJsonEditor.value;
                const parsed = JSON.parse(raw);
                const res = await apiFetch("/api/manifests", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(parsed)
                });
                if (res.ok) {
                    notify("Jepafile registered.", "success");
                    fetchModels();
                } else {
                    const err = await res.json().catch(() => ({}));
                    notify(`Could not register manifest: ${err.error || "unknown error"}`, "error");
                }
            } catch (e) {
                notify(`Invalid JSON: ${e.message}`, "error");
            }
        });
    }

    // SECTION 3: IMAGE PLAYGROUND
    function setupImagePlayground() {
        if (!el.imageDropzone) return;

        el.imageDropzone.addEventListener("click", () => {
            el.imageFileInput.click();
        });

        el.imageDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            el.imageDropzone.classList.add("dragover");
        });

        el.imageDropzone.addEventListener("dragleave", () => {
            el.imageDropzone.classList.remove("dragover");
        });

        el.imageDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            el.imageDropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processUploadedImage(e.dataTransfer.files[0]);
            }
        });

        el.imageFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                processUploadedImage(e.target.files[0]);
            }
        });

        if (el.togglePatchGrid) {
            el.togglePatchGrid.addEventListener("change", () => {
                if (window.currentInspectionImg) {
                    drawInspectionCanvas(window.currentInspectionImg);
                }
            });
        }

        if (el.btnCopyVector) {
            el.btnCopyVector.addEventListener("click", () => {
                if (state.currentVector) {
                    navigator.clipboard.writeText(JSON.stringify(state.currentVector));
                    notify("Vector copied", "success", 2500);
                }
            });
        }

        if (el.btnExportJson) {
            el.btnExportJson.addEventListener("click", () => {
                if (state.currentVector) {
                    const blob = new Blob([JSON.stringify({
                        model: state.activeModel,
                        dimension: state.currentVector.length,
                        embedding: state.currentVector
                    }, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `jepctl-embedding-${Date.now()}.json`;
                    a.click();
                }
            });
        }
    }

    async function processUploadedImage(file) {
        const isImage = /^image\/(png|jpeg|webp)$/.test(file.type);
        if (isImage) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    window.currentInspectionImg = img;
                    el.canvasWrapper.style.display = "block";
                    drawInspectionCanvas(img);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        } else {
            // Clips and audio have no still to inspect; the vector panel is the output.
            el.canvasWrapper.style.display = "none";
            notify(`Embedding ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`, "info", 3000);
        }

        // Upload and embed
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await apiFetch("/api/embed", {
                method: "POST",
                body: formData
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || "Embed request failed");
            }
            const data = await res.json();

            state.currentVector = data.embedding;
            state.lastLatencyMs = data.latency_ms;
            el.outModelName.textContent = data.model;
            el.outEmbedDim.textContent = data.dimension;
            el.outEmbedLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;
            el.kpiLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;

            // Render vector heatmap
            renderVectorHeatmap(data.embedding);

            // Populate numeric array preview
            el.vectorNumericView.value = JSON.stringify(data.embedding.slice(0, 32), null, 2) + "\n... (truncated for display)";

            // Update anomaly baseline if locked
            if (state.nominalBaselineVector) {
                calculateAndRecordEnergy(data.embedding);
            }
        } catch (e) {
            notify(`Inference failed: ${e.message}`, "error");
        }
    }

    function drawInspectionCanvas(img) {
        const canvas = el.imageInspectCanvas;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw image resized to 224x224
        ctx.drawImage(img, 0, 0, 224, 224);

        // Draw ViT patch grid overlay if toggled
        if (el.togglePatchGrid && el.togglePatchGrid.checked) {
            const patchSize = 224 / state.patchGridSize; // 14x14 or 16x16
            ctx.strokeStyle = "rgba(6, 182, 212, 0.45)";
            ctx.lineWidth = 1;

            for (let x = 0; x <= 224; x += patchSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, 224);
                ctx.stroke();
            }

            for (let y = 0; y <= 224; y += patchSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(224, y);
                ctx.stroke();
            }
        }
    }

    function renderVectorHeatmap(vec) {
        const canvas = el.heatmapCanvas;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const step = w / vec.length;
        for (let i = 0; i < vec.length; i++) {
            const val = vec[i];
            // Normalize float to 0..1 color range
            const norm = Math.max(0, Math.min(1, (val + 2.0) / 4.0));
            const r = Math.floor(norm * 255);
            const b = Math.floor((1 - norm) * 255);
            const g = Math.floor(Math.sin(norm * Math.PI) * 200);

            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(i * step, 0, Math.max(1, step), h);
        }
    }

    // SECTION 4: VIDEO & CAMERA STREAM
    function setupVideoStream() {
        // Toggle camera device
        if (el.btnStreamToggle) {
            el.btnStreamToggle.addEventListener("click", () => {
                if (state.isStreaming) {
                    stopLiveStream();
                } else {
                    startLiveStream();
                }
            });
        }

        if (el.cameraDeviceSelect) {
            el.cameraDeviceSelect.addEventListener("change", () => {
                if (state.isStreaming) {
                    stopLiveStream();
                    setTimeout(startLiveStream, 300);
                }
            });
        }

        if (el.streamFpsSelect) {
            el.streamFpsSelect.addEventListener("change", (e) => {
                state.streamFps = parseInt(e.target.value, 10);
                el.kpiFps.textContent = `${state.streamFps} FPS`;
                if (state.isStreaming) {
                    stopLiveStream();
                    setTimeout(startLiveStream, 300);
                }
            });
        }

        if (el.btnClearSse) {
            el.btnClearSse.addEventListener("click", () => {
                el.sseStreamLog.innerHTML = '<div class="sse-placeholder">Awaiting live stream connection...</div>';
            });
        }
    }

    async function fetchCameras() {
        try {
            const res = await apiFetch("/api/cameras");
            if (!res.ok) return;
            const cams = await res.json();
            if (el.cameraDeviceSelect) {
                const prevVal = el.cameraDeviceSelect.value;
                el.cameraDeviceSelect.innerHTML = "";
                cams.forEach((c) => {
                    const opt = document.createElement("option");
                    opt.value = c.index;
                    opt.textContent = `${c.name} (Device #${c.index})`;
                    el.cameraDeviceSelect.appendChild(opt);
                });
                if (prevVal && cams.some((c) => c.index == prevVal)) {
                    el.cameraDeviceSelect.value = prevVal;
                }
            }
        } catch (e) {
            console.error("Camera fetch error:", e);
        }
    }

    // Serialise concurrent start requests (e.g. "Capture" clicked while the camera is
    // still starting): every caller awaits the same in-flight start.
    function startLiveStream() {
        if (state.isStreaming) return Promise.resolve();
        if (!state.streamStarting) {
            state.streamStarting = startLiveStreamInner().finally(() => {
                state.streamStarting = null;
            });
        }
        return state.streamStarting;
    }

    async function startLiveStreamInner() {
        const camIdx = el.cameraDeviceSelect ? parseInt(el.cameraDeviceSelect.value, 10) : 0;
        const fps = state.streamFps;

        try {
            // 1. Trigger backend camera daemon capture
            await apiFetch(`/api/camera/start?device=${camIdx}&fps=${fps}`, { method: "POST" });

            // 2. Open Server-Sent Events (SSE) listener
            if (!state.authToken) await fetchSessionToken();
            const tokenParam = state.authToken && state.authToken !== "no_auth" ? `&token=${encodeURIComponent(state.authToken)}` : "";
            const sseUrl = `/api/embed/stream?fps=${fps}&threshold=${state.gestureThreshold}&margin=${state.gestureMargin}${tokenParam}`;
            state.sseSource = new EventSource(sseUrl);

            state.sseSource.onmessage = (e) => {
                try {
                    const event = JSON.parse(e.data);
                    logSseEvent(event);

                    state.lastLatencyMs = event.latency_ms;
                    el.kpiLatency.textContent = `${event.latency_ms.toFixed(1)} ms`;
                    if (el.headerLatencyText) el.headerLatencyText.textContent = `${event.latency_ms.toFixed(0)} ms`;

                    if (event.embedding) {
                        state.currentVector = event.embedding;
                        calculateAndRecordEnergy(event.embedding);
                    }

                    processGestureRecognition(event);
                } catch (err) {
                    console.error("SSE parse error:", err);
                }
            };

            state.sseSource.addEventListener("error", (evt) => {
                if (evt && evt.data) {
                    try {
                        const err = JSON.parse(evt.data);
                        showStreamError(err.error || "Stream error");
                        logSseEvent({ error: err.error || "Stream error" });
                    } catch (_) {
                        showStreamError("Stream error");
                    }
                } else {
                    console.warn("SSE connection interrupted.");
                }
            });
            startModelViewPolling();

            // 3. Streaming is live as soon as the server camera and SSE are up.
            state.isStreaming = true;
            el.btnStreamToggle.textContent = "Pause Live Stream";
            el.btnStreamToggle.classList.remove("btn-primary");
            el.btnStreamToggle.classList.add("btn-danger");
            el.kpiFps.textContent = `${fps} FPS`;

            if (el.gestureCamBtnText) el.gestureCamBtnText.textContent = "Stop camera";
            if (el.btnGestureCameraToggle) {
                el.btnGestureCameraToggle.classList.remove("btn-primary");
                el.btnGestureCameraToggle.classList.add("btn-danger");
            }

            // 4. Optional browser-side preview for the Video & Camera section. This may
            // wait on a permission prompt, so it must never block the stream itself.
            attachBrowserPreview(camIdx);
        } catch (e) {
            notify(`Could not start the stream: ${e.message}`, "error");
        }
    }

    // Cosmetic full-resolution preview for section 4 via getUserMedia. Inference never
    // uses these pixels: the server camera feeds the ring buffer and the model.
    async function attachBrowserPreview(camIdx) {
        const showBackendFeed = () => {
            if (el.webcamPreviewElement) {
                el.webcamPreviewElement.srcObject = null;
                el.webcamPreviewElement.style.display = "none";
            }
            if (el.streamFeedCanvas) el.streamFeedCanvas.style.display = "block";
            if (el.previewPlaceholder) el.previewPlaceholder.style.display = "none";
            renderWebcamCanvasLoop();
        };

        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            showBackendFeed();
            return;
        }

        try {
            const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
            const videoDevices = devices.filter((d) => d.kind === "videoinput");
            const selectedOpt = el.cameraDeviceSelect && el.cameraDeviceSelect.selectedOptions[0];
            const selectedName = selectedOpt ? selectedOpt.textContent.toLowerCase() : "";

            const matchedDevice = videoDevices.find((d) => {
                const lbl = d.label.toLowerCase();
                if (selectedName.includes("iphone") && lbl.includes("iphone")) return true;
                if (selectedName.includes("macbook") && (lbl.includes("facetime") || lbl.includes("macbook") || lbl.includes("built-in"))) return true;
                return false;
            }) || videoDevices[camIdx];

            const videoConstraints = {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                aspectRatio: { ideal: 1.7777777778 }
            };
            if (matchedDevice && matchedDevice.deviceId) {
                videoConstraints.deviceId = { ideal: matchedDevice.deviceId };
            }

            const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
            if (!state.isStreaming) {
                // Stream was stopped while the permission prompt was open.
                stream.getTracks().forEach((t) => t.stop());
                return;
            }
            state.webcamStream = stream;
            stream.getVideoTracks().forEach((track) => {
                track.onended = () => {
                    if (state.isStreaming) stopLiveStream();
                };
            });
            el.webcamPreviewElement.srcObject = stream;
            el.webcamPreviewElement.style.display = "block";
            if (el.streamFeedCanvas) el.streamFeedCanvas.style.display = "none";
            if (el.previewPlaceholder) el.previewPlaceholder.style.display = "none";
        } catch (camErr) {
            console.warn("Client direct webcam access fallback to backend feed:", camErr);
            showBackendFeed();
        }
    }

    function stopLiveStream() {
        if (state.sseSource) {
            state.sseSource.close();
            state.sseSource = null;
        }

        apiFetch("/api/camera/stop", { method: "POST" }).catch(() => {});
        stopModelViewPolling();

        if (state.webcamStream) {
            state.webcamStream.getTracks().forEach((t) => t.stop());
            state.webcamStream = null;
        }

        if (el.webcamPreviewElement) {
            el.webcamPreviewElement.srcObject = null;
            el.webcamPreviewElement.style.display = "none";
        }
        if (el.streamFeedCanvas) {
            el.streamFeedCanvas.style.display = "none";
        }
        if (el.previewPlaceholder) {
            el.previewPlaceholder.style.display = "block";
        }

        state.isStreaming = false;
        el.btnStreamToggle.textContent = "Start Live Stream";
        el.btnStreamToggle.classList.remove("btn-danger");
        el.btnStreamToggle.classList.add("btn-primary");
        el.kpiFps.textContent = "-- FPS";

        if (el.gestureCamBtnText) el.gestureCamBtnText.textContent = "Start camera";
        if (el.btnGestureCameraToggle) {
            el.btnGestureCameraToggle.classList.remove("btn-danger");
            el.btnGestureCameraToggle.classList.add("btn-primary");
        }
    }

    function renderWebcamCanvasLoop() {
        if (!state.isStreaming) return;
        if (!state.webcamStream) {
            if (el.streamFeedCanvas && state.latestThumbImg && state.latestThumbImg.complete) {
                const ctx = el.streamFeedCanvas.getContext("2d");
                ctx.drawImage(state.latestThumbImg, 0, 0, el.streamFeedCanvas.width, el.streamFeedCanvas.height);
            }
            requestAnimationFrame(renderWebcamCanvasLoop);
        }
    }

    function logSseEvent(ev) {
        if (!el.sseStreamLog) return;
        const placeholder = el.sseStreamLog.querySelector(".sse-placeholder");
        if (placeholder) {
            placeholder.remove();
        }

        const entry = document.createElement("div");
        entry.className = "sse-log-entry sse-entry";
        const time = new Date().toLocaleTimeString();
        let kind = "frame";
        let text = `[${time}] frame #${ev.frame_index} · ${ev.latency_ms.toFixed(1)} ms · ${ev.embedding ? ev.embedding.length : 0}d`;
        if (ev.error) {
            kind = "error";
            text = `[${time}] error · ${ev.error}`;
            entry.classList.add("is-error");
        } else if (ev.gesture_match && ev.gesture_match.detected) {
            kind = "detection";
            text += ` · detected ${ev.gesture_match.matched} (${ev.gesture_match.confidence.toFixed(2)})`;
            entry.classList.add("is-detection");
        } else if (ev.gesture_match) {
            text += ` · ${ev.gesture_match.reason}`;
        }
        entry.dataset.kind = kind;
        entry.textContent = text;
        state.lastSseLine = text;
        const filter = el.sseFilter ? el.sseFilter.value : "all";
        if ((filter === "errors" && kind !== "error") || (filter === "detections" && kind !== "detection")) {
            entry.style.display = "none";
        }
        el.sseStreamLog.prepend(entry);

        // Limit log entries
        while (el.sseStreamLog.children.length > 50) {
            el.sseStreamLog.removeChild(el.sseStreamLog.lastChild);
        }
    }

    // Continuous Ring Buffer Scrubber Refresh
    async function refreshRingBufferThumbnails() {
        if (!el.scrubberStripContainer) return;
        try {
            const res = await apiFetch("/api/ring-buffer");
            if (!res.ok) return;
            const data = await res.json();

            el.bufferCountBadge.textContent = `Buffer: ${data.count} / 16 Frames`;
            el.scrubberStripContainer.innerHTML = "";

            if (!data.thumbnails || data.thumbnails.length === 0) {
                el.scrubberStripContainer.innerHTML = `<div style="color: var(--text-dim); font-size: 12px; padding: 14px 8px; width: 100%;">Camera buffer empty. Click 'Start Live Stream' above to capture live video frames.</div>`;
                return;
            }

            data.thumbnails.forEach((thumb, idx) => {
                const item = document.createElement("div");
                item.className = "scrubber-frame";
                item.innerHTML = `
                    <img src="${thumb}" alt="Frame ${idx + 1}" loading="eager">
                    <span class="scrubber-index">#${idx + 1}</span>
                `;
                el.scrubberStripContainer.appendChild(item);
            });

            if (data.thumbnails && data.thumbnails.length > 0) {
                const latest = data.thumbnails[data.thumbnails.length - 1];
                if (!state.latestThumbImg) {
                    state.latestThumbImg = new Image();
                }
                state.latestThumbImg.src = latest;
            }
        } catch (e) {
            // silent ignore during idle
        }
    }

    // SECTION 5: ANOMALY & ENERGY MONITOR
    function setupAnomalyMonitor() {
        if (el.btnLockBaseline) {
            el.btnLockBaseline.addEventListener("click", () => {
                if (state.currentVector) {
                    state.nominalBaselineVector = [...state.currentVector];
                    el.baselineStatusLabel.textContent = "Locked (Active)";
                    el.baselineStatusLabel.style.color = "var(--accent-green)";
                    notify("Baseline locked on the current frame.", "success");
                } else {
                    notify("No embedding yet: embed an image or start the camera first.", "warning");
                }
            });
        }

        if (el.sliderThreshold) {
            el.sliderThreshold.addEventListener("input", (e) => {
                state.alertThreshold = parseFloat(e.target.value);
                el.valThreshold.textContent = state.alertThreshold.toFixed(2);
                renderEnergyChart();
            });
        }

        if (el.btnSaveWebhook) {
            el.btnSaveWebhook.addEventListener("click", () => {
                state.webhookUrl = el.inputWebhookUrl.value.trim();
                notify("Webhook URL saved.", "success");
            });
        }
    }

    function calculateAndRecordEnergy(incomingVec) {
        if (!state.nominalBaselineVector) return;

        // Compute L2 Euclidean Distance: sqrt(sum((v1 - v2)^2))
        let sumSq = 0;
        const len = Math.min(incomingVec.length, state.nominalBaselineVector.length);
        for (let i = 0; i < len; i++) {
            const diff = incomingVec[i] - state.nominalBaselineVector[i];
            sumSq += diff * diff;
        }
        const distance = Math.sqrt(sumSq) / Math.sqrt(len);

        state.energyHistory.push(distance);
        if (state.energyHistory.length > state.maxChartPoints) {
            state.energyHistory.shift();
        }

        // Anomaly Evaluation
        const isAnomaly = distance > state.alertThreshold;
        if (isAnomaly) {
            triggerAnomalyAlert(distance);
        } else {
            el.anomalyAlertBanner.style.display = "none";
        }

        renderEnergyChart();
    }

    function triggerAnomalyAlert(dist) {
        el.anomalyAlertBanner.style.display = "flex";
        el.anomalyAlertBanner.querySelector(".alert-msg").textContent = 
            `ANOMALY DETECTED: Latent energy distance (${dist.toFixed(3)}) exceeded alert threshold (${state.alertThreshold.toFixed(3)})!`;

        // Audio synthesizer beep alert via Web Audio API
        const now = Date.now();
        if (now - state.lastAlertSoundTime > 1500) {
            playAlertTone();
            state.lastAlertSoundTime = now;
        }

        // Webhook trigger
        if (state.webhookUrl) {
            fetch(state.webhookUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    event: "anomaly_detected",
                    distance: dist,
                    threshold: state.alertThreshold,
                    model: state.activeModel,
                    timestamp: new Date().toISOString()
                })
            }).catch(() => {});
        }
    }

    function playAlertTone() {
        try {
            if (!state.audioContext) {
                state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            const ctx = state.audioContext;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(880, ctx.currentTime); // 880 Hz A5 note
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);

            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.3);
        } catch (e) {
            // ignore audio failure if blocked by browser policy
        }
    }

    function renderEnergyChart() {
        const canvas = el.energyChartCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Background Grid
        ctx.strokeStyle = "rgba(43, 49, 66, 0.4)";
        ctx.lineWidth = 1;
        for (let y = 0; y <= h; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Alert Threshold Line
        const thresholdY = h - (state.alertThreshold / 1.5) * h;
        ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(0, thresholdY);
        ctx.lineTo(w, thresholdY);
        ctx.stroke();
        ctx.setLineDash([]); // Reset line dash

        // Render Energy History Curve
        if (state.energyHistory.length > 1) {
            ctx.strokeStyle = "var(--accent-cyan)";
            ctx.lineWidth = 2.5;
            ctx.beginPath();

            const step = w / (state.maxChartPoints - 1);
            state.energyHistory.forEach((val, i) => {
                const x = i * step;
                const y = h - (Math.min(1.5, val) / 1.5) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
    }

    // SECTION 6: SECURITY & KEYS
    function setupSecurityAndKeys() {
        let closeKeyModal = null;
        if (el.btnOpenCreateKey) {
            el.btnOpenCreateKey.addEventListener("click", () => {
                el.generatedTokenDisplay.style.display = "none";
                closeKeyModal = openDialog(el.createKeyModal, { initialFocus: el.modalKeyName, onClose: () => { closeKeyModal = null; } });
            });
        }

        if (el.btnCloseKeyModal) {
            el.btnCloseKeyModal.addEventListener("click", () => {
                if (closeKeyModal) closeKeyModal();
                else el.createKeyModal.style.display = "none";
            });
        }

        if (el.btnSubmitCreateKey) {
            el.btnSubmitCreateKey.addEventListener("click", async () => {
                const name = el.modalKeyName.value.trim() || "API Token";
                const role = el.modalKeyRole.value;
                const days = el.modalKeyExpire.value ? parseInt(el.modalKeyExpire.value, 10) : null;

                try {
                    const res = await apiFetch("/api/keys", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ name, role, expire_days: days })
                    });
                    if (!res.ok) {
                        const err = await res.json().catch(() => ({}));
                        throw new Error(err.error || "Failed to create key");
                    }
                    const data = await res.json();

                    el.rawTokenValue.value = data.raw_token;
                    el.generatedTokenDisplay.style.display = "block";
                    fetchApiKeys();
                } catch (e) {
                    notify(`Could not create key: ${e.message}`, "error");
                }
            });
        }

        if (el.btnCopyRawToken) {
            el.btnCopyRawToken.addEventListener("click", () => {
                navigator.clipboard.writeText(el.rawTokenValue.value);
                notify("Token copied", "success", 2500);
            });
        }

        if (el.toggleLanAccess) {
            el.toggleLanAccess.addEventListener("change", (e) => {
                if (e.target.checked) {
                    confirmDialog("Binding to 0.0.0.0 lets other machines on your network call inference. Authentication stays required. Continue?", { title: "Enable LAN access", okLabel: "Enable" }).then((ok) => {
                        if (!ok) {
                            e.target.checked = false;
                            return;
                        }
                        el.lanWarningBox.style.display = "block";
                    });
                } else {
                    el.lanWarningBox.style.display = "none";
                }
            });
        }

        if (el.btnRefreshAudit) {
            el.btnRefreshAudit.addEventListener("click", fetchAuditLog);
        }
    }

    async function fetchApiKeys() {
        if (!el.apiKeysTbody) return;
        try {
            const res = await apiFetch("/api/keys");
            if (!res.ok) return;
            const keys = await res.json();

            el.apiKeysTbody.innerHTML = "";
            keys.forEach((k) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><code>${k.key_prefix}...</code></td>
                    <td>${k.name}</td>
                    <td><span class="badge badge-dim">${k.role}</span></td>
                    <td>${new Date(k.created_at).toLocaleDateString()}</td>
                    <td>${k.expires_at ? new Date(k.expires_at).toLocaleDateString() : "Never"}</td>
                    <td><button class="btn btn-sm btn-danger" data-prefix="${k.key_prefix}">Revoke</button></td>
                `;

                tr.querySelector("button").addEventListener("click", async () => {
                    if (await confirmDialog(`Revoke key ${k.key_prefix}? Clients using it will get 403 immediately.`, { title: "Revoke key", okLabel: "Revoke" })) {
                        await apiFetch(`/api/keys/${k.key_prefix}`, { method: "DELETE" });
                        notify(`Key ${k.key_prefix} revoked`, "success");
                        fetchApiKeys();
                    }
                });

                el.apiKeysTbody.appendChild(tr);
            });
        } catch (e) {
            console.error("fetchApiKeys error:", e);
        }
    }

    async function fetchAuditLog() {
        if (!el.auditLogTbody) return;
        try {
            const res = await apiFetch("/api/audit");
            if (!res.ok) return;
            const logs = await res.json();

            el.auditLogTbody.innerHTML = "";
            logs.slice(0, 50).forEach((entry) => {
                const tr = document.createElement("tr");
                const time = new Date(entry.timestamp).toLocaleTimeString();
                tr.innerHTML = `
                    <td>${time}</td>
                    <td><code>${entry.method}</code></td>
                    <td>${entry.path}</td>
                    <td>${entry.client_ip}</td>
                    <td><span class="badge ${entry.status_code < 400 ? 'badge-image' : 'badge-video'}">${entry.status_code}</span></td>
                    <td>${entry.latency_ms.toFixed(1)} ms</td>
                `;
                el.auditLogTbody.appendChild(tr);
            });
        } catch (e) {
            console.error("fetchAuditLog error:", e);
        }
    }

    // SECTION 7: SETTINGS
    async function setupSettings() {
        try {
            const res = await apiFetch("/api/settings");
            if (res.ok) {
                const s = await res.json();
                if (el.settingsBackend && s.compute_backend) {
                    el.settingsBackend.value = s.compute_backend;
                }
                if (el.settingsMemWatermark && s.gpu_memory_high_watermark) {
                    el.settingsMemWatermark.value = s.gpu_memory_high_watermark;
                }
                if (el.settingsIdleTimeout && s.idle_unload_timeout_minutes) {
                    el.settingsIdleTimeout.value = s.idle_unload_timeout_minutes;
                }
                if (el.settingsStorageDir && s.storage_dir) {
                    el.settingsStorageDir.value = s.storage_dir;
                }
            }
        } catch (_) {}

        if (el.btnSaveSettings) {
            el.btnSaveSettings.addEventListener("click", async () => {
                const backend = el.settingsBackend.value;
                const memRatio = parseFloat(el.settingsMemWatermark.value);
                const timeout = parseInt(el.settingsIdleTimeout.value, 10);

                try {
                    const res = await apiFetch("/api/settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            compute_backend: backend,
                            gpu_memory_high_watermark: memRatio,
                            idle_unload_timeout_minutes: timeout,
                            storage_dir: el.settingsStorageDir.value
                        })
                    });
                    if (res.ok) {
                        notify("Settings saved.", "success");
                    } else {
                        const err = await res.json().catch(() => ({}));
                        notify(`Could not save settings: ${err.error || "unknown error"}`, "error");
                    }
                } catch (e) {
                    notify(`Could not save settings: ${e.message}`, "error");
                }
            });
        }
    }

    // SECTION 8: FEW-SHOT GESTURE SANDBOX
    // ------------------------------------------------------------------
    // Gesture Sandbox
    //
    // Registration and live matching both run on the *server* camera frame
    // (see /api/camera/frame and POST /api/gestures {from_camera:true}), so the
    // reference prototypes and the live embeddings always share one pipeline.
    // The server returns a full GestureMatchResult per frame; the UI only adds
    // temporal smoothing and its own threshold/margin on top of it.
    // ------------------------------------------------------------------

    function setupGestureSandbox() {
        if (el.btnGestureCameraToggle) {
            el.btnGestureCameraToggle.addEventListener("click", () => {
                if (state.isStreaming) {
                    stopLiveStream();
                } else {
                    startLiveStream();
                }
            });
        }

        const bindSlider = (slider, label, key, fmt) => {
            if (!slider) return;
            slider.addEventListener("input", (e) => {
                const val = parseFloat(e.target.value);
                state[key] = val;
                if (label) label.textContent = fmt(val);
            });
        };
        bindSlider(el.sliderGestureThreshold, el.valGestureThreshold, "gestureThreshold", (v) => v.toFixed(2));
        bindSlider(el.sliderGestureMargin, el.valGestureMargin, "gestureMargin", (v) => v.toFixed(2));
        bindSlider(el.sliderGestureSmoothing, el.valGestureSmoothing, "gestureSmoothing", (v) => String(Math.round(v)));

        if (el.toggleGestureAudio) {
            el.toggleGestureAudio.addEventListener("change", (e) => {
                state.gestureAudioEnabled = e.target.checked;
            });
        }
        if (el.toggleGestureHeatmap) {
            el.toggleGestureHeatmap.addEventListener("change", (e) => {
                state.gestureHeatmap = e.target.checked;
                if (!state.gestureHeatmap) clearHeatmap();
            });
        }

        for (let i = 1; i <= GESTURE_SLOT_COUNT; i++) {
            const btnCap = document.getElementById(`btn-capture-${i}`);
            if (btnCap) btnCap.addEventListener("click", () => captureSlotPose(i));
            const btnDel = document.getElementById(`btn-delete-${i}`);
            if (btnDel) btnDel.addEventListener("click", () => deleteSlotPose(i));
        }

        if (el.btnApiLoadModel) {
            el.btnApiLoadModel.addEventListener("click", () => {
                const chosen = (el.selectGestureActiveModel && el.selectGestureActiveModel.value) || state.activeModel || "facebook/dinov2-small";
                showApiDialog("Load a model", "Admin role. Returns the checkpoint coverage report; 409 if the model is not pulled, 422 if its layout is unsupported.",
                    { method: "POST", path: "/api/models/load", json: { model_name: chosen } });
            });
        }
        if (el.btnApiRegister) {
            el.btnApiRegister.addEventListener("click", () => {
                const name = slotName(1) || "Open Hand";
                showApiDialog("Register a gesture sample", "Inference role. Call it several times with the same name to add samples. `from_camera: true` uses the server camera; you can also send `image_base64` or a raw `embedding`.",
                    { method: "POST", path: "/api/gestures", json: { name, from_camera: true, is_neutral: false } });
            });
        }
        if (el.btnApiMatch) {
            el.btnApiMatch.addEventListener("click", () => {
                showApiDialog("Match a frame", "Inference role. Threshold and margin below are the ones currently set in this panel. The response is the same decision trace shown here.", gestureMatchRequest());
            });
        }
        if (el.btnGesturesExport) {
            el.btnGesturesExport.addEventListener("click", async () => {
                try {
                    const q = new URLSearchParams({ threshold: state.gestureThreshold.toFixed(2), margin: state.gestureMargin.toFixed(2) });
                    const res = await apiFetch(`/api/gestures/export?${q}`);
                    if (!res.ok) throw new Error(`${res.status}`);
                    const bundle = await res.json();
                    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
                    const a = document.createElement("a");
                    a.href = URL.createObjectURL(blob);
                    const model = (state.activeModel || "gestures").replace(/[^a-z0-9]+/gi, "-");
                    a.download = `jepctl-gestures-${model}.json`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
                    notify(`Exported ${bundle.gestures.length} gesture(s). Import it with POST /api/gestures/import or \`jepctl gestures import\`.`, "success", 7000);
                } catch (e) {
                    notify(`Export failed: ${e.message}`, "error");
                }
            });
        }
        if (el.btnGesturesImport && el.inputGesturesImport) {
            el.btnGesturesImport.addEventListener("click", () => el.inputGesturesImport.click());
            el.inputGesturesImport.addEventListener("change", async () => {
                const file = el.inputGesturesImport.files && el.inputGesturesImport.files[0];
                el.inputGesturesImport.value = "";
                if (!file) return;
                try {
                    const text = await file.text();
                    const bundle = JSON.parse(text);
                    const n = Array.isArray(bundle.gestures) ? bundle.gestures.length : 0;
                    const replace = await confirmDialog(`Import ${n} gesture(s)? Existing gestures of the same models will be replaced.`, { title: "Import bundle", okLabel: "Replace and import", danger: false });
                    if (!replace) return;
                    const res = await apiFetch("/api/gestures/import?replace=true", { method: "POST", headers: { "Content-Type": "application/json" }, body: text });
                    const report = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(report.error || `${res.status}`);
                    notify(`Imported ${report.imported} gesture(s) for ${(report.models || []).join(", ")}`, "success");
                    if (typeof bundle.threshold === "number" && el.sliderGestureThreshold) {
                        el.sliderGestureThreshold.value = bundle.threshold;
                        el.sliderGestureThreshold.dispatchEvent(new Event("input"));
                    }
                    if (typeof bundle.margin === "number" && el.sliderGestureMargin) {
                        el.sliderGestureMargin.value = bundle.margin;
                        el.sliderGestureMargin.dispatchEvent(new Event("input"));
                    }
                    state.gestureHistory = {};
                    await fetchGesturesList();
                } catch (e) {
                    notify(`Import failed: ${e.message}`, "error");
                }
            });
        }
        if (el.sseFilter) {
            el.sseFilter.addEventListener("change", () => {
                const f = el.sseFilter.value;
                el.sseStreamLog.querySelectorAll(".sse-entry").forEach((n) => {
                    const k = n.dataset.kind;
                    n.style.display = f === "all" || (f === "errors" && k === "error") || (f === "detections" && k === "detection") ? "" : "none";
                });
            });
        }
        if (el.btnCopySse) {
            el.btnCopySse.addEventListener("click", () => {
                if (state.lastSseLine) copyText(state.lastSseLine, "Event copied");
                else notify("No event yet.", "info", 2500);
            });
        }

        if (el.btnGesturesClear) {
            el.btnGesturesClear.addEventListener("click", async () => {
                if (!(await confirmDialog("Delete every gesture registered with the active model?", { title: "Clear gestures", okLabel: "Delete all" }))) return;
                try {
                    const res = await apiFetch("/api/gestures", { method: "DELETE" });
                    if (!res.ok) {
                        const err = await res.json().catch(() => ({}));
                        notify(`Error: ${err.error || "request failed"}`, "error");
                    }
                } catch (e) {
                    notify(`Error: ${e.message}`, "error");
                }
                state.gestureHistory = {};
                await fetchGesturesList();
            });
        }
    }

    function updateWeightsBadge(weights, activeModel) {
        if (!el.gestureWeightsBadge) return;
        if (!activeModel || !weights) {
            el.gestureWeightsBadge.textContent = "weights: --";
            el.gestureWeightsBadge.className = "weights-badge";
            el.gestureWeightsBadge.title = "No model loaded";
            return;
        }
        const ok = weights.loaded === weights.expected && weights.expected > 0;
        el.gestureWeightsBadge.textContent = `weights: ${weights.loaded}/${weights.expected} (${weights.source})`;
        el.gestureWeightsBadge.className = `weights-badge ${ok ? "ok" : "bad"}`;
        el.gestureWeightsBadge.title = ok
            ? "Every parameter comes from the checkpoint"
            : "Incomplete checkpoint: embeddings are not reliable";
    }

    function showStreamError(message) {
        if (el.gestureDetectionBadge) {
            el.gestureDetectionBadge.className = "gesture-badge gesture-badge-idle";
        }
        if (el.gestureBadgeText) el.gestureBadgeText.textContent = `Erreur : ${message}`;
        if (el.gestureReasoningDecision) {
            el.gestureReasoningDecision.textContent = `Stream interrupted: ${message}`;
            el.gestureReasoningDecision.classList.remove("detected");
        }
    }

    // --- Model view polling (what the network actually receives) ---------

    function startModelViewPolling() {
        stopModelViewPolling();
        if (el.gestureViewPlaceholder) el.gestureViewPlaceholder.style.display = "none";
        const interval = Math.max(100, Math.round(1000 / (state.streamFps || 10)));
        const tick = async () => {
            if (!state.isStreaming) return;
            try {
                const res = await apiFetch("/api/camera/frame");
                if (res.ok) {
                    const seq = res.headers.get("x-frame-sequence");
                    if (seq !== state.modelViewSequence) {
                        state.modelViewSequence = seq;
                        const blob = await res.blob();
                        const url = URL.createObjectURL(blob);
                        if (el.gestureModelView) el.gestureModelView.src = url;
                        if (state.modelViewObjectUrl) URL.revokeObjectURL(state.modelViewObjectUrl);
                        state.modelViewObjectUrl = url;
                    }
                }
            } catch (e) {
                console.warn("model view fetch failed:", e);
            }
        };
        tick();
        state.modelViewTimer = setInterval(tick, interval);
        startRoiPolling();
    }

    function stopModelViewPolling() {
        if (state.modelViewTimer) {
            clearInterval(state.modelViewTimer);
            state.modelViewTimer = null;
        }
        stopRoiPolling();
        if (el.gestureViewPlaceholder) el.gestureViewPlaceholder.style.display = "flex";
        clearHeatmap();
    }

    function clearHeatmap() {
        if (!el.gestureHeatmap) return;
        const ctx = el.gestureHeatmap.getContext("2d");
        ctx.clearRect(0, 0, el.gestureHeatmap.width, el.gestureHeatmap.height);
    }

    // Per-patch dissimilarity to the best prototype, drawn as a grid overlay.
    function drawHeatmap(patchDiff, gridSize) {
        if (!el.gestureHeatmap || !state.gestureHeatmap) return;
        if (!Array.isArray(patchDiff) || !gridSize || patchDiff.length !== gridSize * gridSize) {
            clearHeatmap();
            return;
        }
        const canvas = el.gestureHeatmap;
        const ctx = canvas.getContext("2d");
        const cell = canvas.width / gridSize;
        // Normalise against the frame's own max so the map always has contrast.
        const max = Math.max(1e-6, ...patchDiff);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < patchDiff.length; i++) {
            const v = Math.max(0, Math.min(1, patchDiff[i] / max));
            const x = (i % gridSize) * cell;
            const y = Math.floor(i / gridSize) * cell;
            // Dark -> amber -> red, alpha grows with difference.
            const r = Math.round(255 * Math.min(1, v * 1.4));
            const g = Math.round(180 * Math.max(0, 1 - Math.abs(v - 0.5) * 2));
            ctx.fillStyle = `rgba(${r}, ${g}, 40, ${(0.15 + 0.7 * v).toFixed(2)})`;
            ctx.fillRect(x, y, cell, cell);
        }
    }

    // --- Region of interest editor -------------------------------------------

    function renderRoi() {
        const roi = state.cameraRoi;
        if (el.roiValues) el.roiValues.textContent = roi ? `x ${roi.x.toFixed(3)} · y ${roi.y.toFixed(3)} · w ${roi.w.toFixed(3)} · h ${roi.h.toFixed(3)}` : "none";
        if (el.roiStatusText) el.roiStatusText.textContent = roi ? `${Math.round(roi.w * 100)}% × ${Math.round(roi.h * 100)}% crop` : "full frame";
        if (el.roiBox) {
            if (roi) {
                el.roiBox.style.display = "block";
                el.roiBox.style.left = `${roi.x * 100}%`;
                el.roiBox.style.top = `${roi.y * 100}%`;
                el.roiBox.style.width = `${roi.w * 100}%`;
                el.roiBox.style.height = `${roi.h * 100}%`;
            } else {
                el.roiBox.style.display = "none";
            }
        }
    }

    async function fetchRoi() {
        try {
            const res = await apiFetch("/api/camera/roi");
            if (res.ok) {
                const data = await res.json();
                state.cameraRoi = data.roi || null;
                renderRoi();
            }
        } catch (e) {
            console.warn("roi fetch failed:", e);
        }
    }

    async function saveRoi(roi) {
        try {
            const res = roi
                ? await apiFetch("/api/camera/roi", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(roi) })
                : await apiFetch("/api/camera/roi", { method: "DELETE" });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.error || `${res.status}`);
            state.cameraRoi = data.roi || null;
            renderRoi();
            state.gestureHistory = {};
            notify(roi ? "Region of interest saved: re-capture your samples with this crop." : "Using the full frame.", "success");
        } catch (e) {
            notify(`Could not save the region: ${e.message}`, "error");
        }
    }

    function startRoiPolling() {
        stopRoiPolling();
        if (el.roiPlaceholder) el.roiPlaceholder.style.display = "none";
        const tick = async () => {
            if (!state.isStreaming || !el.roiCard || !el.roiCard.open) return;
            try {
                const res = await apiFetch("/api/camera/frame?full=true");
                if (!res.ok) return;
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                el.roiFullFrame.src = url;
                if (state.roiObjectUrl) URL.revokeObjectURL(state.roiObjectUrl);
                state.roiObjectUrl = url;
            } catch (e) {
                console.warn("roi frame fetch failed:", e);
            }
        };
        tick();
        state.roiTimer = setInterval(tick, 500);
    }

    function stopRoiPolling() {
        if (state.roiTimer) {
            clearInterval(state.roiTimer);
            state.roiTimer = null;
        }
        if (el.roiPlaceholder) el.roiPlaceholder.style.display = "flex";
    }

    function setupRoiEditor() {
        if (!el.roiStage) return;
        fetchRoi();
        if (el.roiCard) el.roiCard.addEventListener("toggle", () => { if (el.roiCard.open && state.isStreaming) startRoiPolling(); });

        const pointFrom = (e) => {
            const r = el.roiStage.getBoundingClientRect();
            return {
                x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
                y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
            };
        };
        el.roiStage.addEventListener("pointerdown", (e) => {
            if (!state.isStreaming) return;
            state.roiDrag = { start: pointFrom(e) };
            el.roiStage.setPointerCapture(e.pointerId);
        });
        el.roiStage.addEventListener("pointermove", (e) => {
            if (!state.roiDrag) return;
            const a = state.roiDrag.start, b = pointFrom(e);
            const roi = { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y) };
            state.roiDrag.current = roi;
            el.roiBox.style.display = "block";
            el.roiBox.style.left = `${roi.x * 100}%`;
            el.roiBox.style.top = `${roi.y * 100}%`;
            el.roiBox.style.width = `${roi.w * 100}%`;
            el.roiBox.style.height = `${roi.h * 100}%`;
        });
        const finish = () => {
            if (!state.roiDrag) return;
            const roi = state.roiDrag.current;
            state.roiDrag = null;
            if (roi && roi.w > 0.02 && roi.h > 0.02) {
                saveRoi({ x: +roi.x.toFixed(4), y: +roi.y.toFixed(4), w: +roi.w.toFixed(4), h: +roi.h.toFixed(4) });
            } else {
                renderRoi();
            }
        };
        el.roiStage.addEventListener("pointerup", finish);
        el.roiStage.addEventListener("pointercancel", finish);
        if (el.btnRoiClear) el.btnRoiClear.addEventListener("click", () => saveRoi(null));
        if (el.btnApiRoi) {
            el.btnApiRoi.addEventListener("click", () => {
                const roi = state.cameraRoi || { x: 0.3, y: 0.2, w: 0.4, h: 0.6 };
                showApiDialog("Set the camera region of interest", "Inference role. Normalised coordinates on the raw camera frame. DELETE /api/camera/roi restores the full frame. Exported bundles carry the ROI so deployments crop identically.",
                    { method: "PUT", path: "/api/camera/roi", json: roi });
            });
        }
    }

    // --- Registration ------------------------------------------------------

    function slotName(slotIndex) {
        const input = document.getElementById(`gesture-name-${slotIndex}`);
        return input ? input.value.trim() : "";
    }

    function gestureForSlot(slotIndex) {
        const name = slotName(slotIndex);
        return state.registeredGestures.find((g) => g.name === name) || null;
    }

    async function captureSlotPose(slotIndex) {
        const name = slotName(slotIndex);
        if (!name) {
            notify("Give the gesture a name first.", "warning");
            return;
        }

        if (!state.isStreaming) {
            try {
                await startLiveStream();
                await new Promise((r) => setTimeout(r, 800));
            } catch (e) {
                notify("Start the camera to capture a reference gesture.", "warning");
                return;
            }
        }

        const btnCap = document.getElementById(`btn-capture-${slotIndex}`);
        const originalHtml = btnCap ? btnCap.innerHTML : "Capture";
        if (btnCap) {
            btnCap.disabled = true;
            btnCap.innerHTML = `<span class="btn-icon">&#9203;</span> Encoding...`;
        }

        try {
            const res = await apiFetch("/api/gestures", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name,
                    from_camera: true,
                    is_neutral: slotIndex === GESTURE_NEUTRAL_SLOT
                })
            });
            if (res.ok) {
                await fetchGesturesList();
            } else {
                const err = await res.json().catch(() => ({}));
                notify(`Could not register: ${err.error || "request failed"}`, "error");
            }
        } catch (err) {
            console.error("captureSlotPose error:", err);
            notify(`Capture failed: ${err.message}`, "error");
        } finally {
            if (btnCap) {
                btnCap.disabled = false;
                btnCap.innerHTML = originalHtml;
            }
        }
    }

    async function deleteSlotPose(slotIndex) {
        const g = gestureForSlot(slotIndex);
        if (!g) return;
        if (!(await confirmDialog(`Delete gesture "${g.name}" (${g.sample_count} sample(s))?`, { title: "Delete gesture", okLabel: "Delete" }))) return;

        try {
            const res = await apiFetch(`/api/gestures/${encodeURIComponent(g.name)}`, { method: "DELETE" });
            if (res.ok) {
                state.slotHoldCounts[slotIndex] = 0;
                const counterEl = document.getElementById(`slot-counter-${slotIndex}`);
                if (counterEl) counterEl.textContent = "0";
                delete state.gestureHistory[g.name];
                await fetchGesturesList();
            } else {
                const err = await res.json().catch(() => ({}));
                notify(`Delete failed: ${err.error || "request failed"}`, "error");
            }
        } catch (e) {
            console.error("deleteSlotPose error:", e);
            notify(`Error: ${e.message}`, "error");
        }
    }

    async function fetchGesturesList() {
        try {
            const res = await apiFetch("/api/gestures");
            if (!res.ok) return;
            const gestures = await res.json();
            state.registeredGestures = Array.isArray(gestures) ? gestures : [];

            if (el.gestureCountBadge) {
                const n = state.registeredGestures.length;
                el.gestureCountBadge.textContent = `${n} gesture${n === 1 ? "" : "s"}`;
            }

            const activeNames = new Set(state.registeredGestures.map((g) => g.name));
            for (const k of Object.keys(state.gestureHistory)) {
                if (!activeNames.has(k)) delete state.gestureHistory[k];
            }

            // Fill slots: a slot shows the gesture whose name matches its input; unmatched
            // gestures are assigned to remaining empty slots (neutral ones to the neutral slot).
            const assigned = new Set();
            const slotGesture = {};
            for (let i = 1; i <= GESTURE_SLOT_COUNT; i++) {
                const g = state.registeredGestures.find((item) => item.name === slotName(i) && !assigned.has(item.name));
                if (g) {
                    slotGesture[i] = g;
                    assigned.add(g.name);
                }
            }
            for (const g of state.registeredGestures) {
                if (assigned.has(g.name)) continue;
                for (let i = 1; i <= GESTURE_SLOT_COUNT; i++) {
                    const isNeutralSlot = i === GESTURE_NEUTRAL_SLOT;
                    if (slotGesture[i] || isNeutralSlot !== !!g.is_neutral) continue;
                    slotGesture[i] = g;
                    assigned.add(g.name);
                    const input = document.getElementById(`gesture-name-${i}`);
                    if (input) input.value = g.name;
                    break;
                }
            }

            for (let i = 1; i <= GESTURE_SLOT_COUNT; i++) {
                renderSlot(i, slotGesture[i] || null);
            }

            if (state.registeredGestures.length === 0 && el.gestureBadgeText) {
                el.gestureBadgeText.textContent = state.isStreaming
                    ? "Waiting: capture a slot on the right"
                    : "No gesture registered (start the camera)";
            }
        } catch (e) {
            console.error("fetchGesturesList error:", e);
        }
    }

    function renderSlot(i, g) {
        const card = document.getElementById(`slot-card-${i}`);
        const stateBadge = document.getElementById(`slot-state-${i}`);
        const imgEl = document.getElementById(`slot-img-${i}`);
        const emptyEl = document.getElementById(`slot-empty-${i}`);
        const metaEl = document.getElementById(`slot-meta-${i}`);
        const btnCap = document.getElementById(`btn-capture-${i}`);
        const btnDel = document.getElementById(`btn-delete-${i}`);
        const isNeutralSlot = i === GESTURE_NEUTRAL_SLOT;

        if (g) {
            if (card) card.classList.add("has-gesture");
            if (stateBadge) {
                stateBadge.textContent = `${g.sample_count} sample${g.sample_count === 1 ? "" : "s"}`;
                stateBadge.className = "slot-state-badge registered";
            }
            if (imgEl) {
                if (g.thumbnail) {
                    imgEl.src = g.thumbnail;
                    imgEl.style.display = "block";
                } else {
                    imgEl.style.display = "none";
                }
            }
            if (emptyEl) emptyEl.style.display = g.thumbnail ? "none" : "flex";
            if (metaEl) {
                const t = g.updated_at ? new Date(g.updated_at * 1000).toLocaleTimeString() : "--";
                metaEl.textContent = `${g.dimension}d · ${g.sample_count} sample(s) · ${g.model_name} · ${t}`;
            }
            if (btnCap) btnCap.innerHTML = `<span class="btn-icon">&#10133;</span> Add sample`;
            if (btnDel) btnDel.style.display = "inline-flex";
        } else {
            if (card) {
                card.classList.remove("has-gesture");
                card.classList.remove("matched-active");
            }
            if (stateBadge) {
                stateBadge.textContent = "Empty";
                stateBadge.className = "slot-state-badge";
            }
            if (imgEl) {
                imgEl.src = "";
                imgEl.style.display = "none";
            }
            if (emptyEl) emptyEl.style.display = "flex";
            if (metaEl) {
                metaEl.textContent = isNeutralSlot
                    ? "Rest pose: absorbs frames with no intentional gesture. Never reported as a detection."
                    : "Capture 3 to 5 samples while moving slightly.";
            }
            if (btnCap) btnCap.innerHTML = `<span class="btn-icon">&#128247;</span> Capture`;
            if (btnDel) btnDel.style.display = "none";
            const scoreValEl = document.getElementById(`slot-score-val-${i}`);
            const scoreBarEl = document.getElementById(`slot-score-bar-${i}`);
            if (scoreValEl) scoreValEl.textContent = "--%";
            if (scoreBarEl) scoreBarEl.style.width = "0%";
        }
    }

    // --- Live recognition --------------------------------------------------

    function processGestureRecognition(event) {
        const match = event && event.gesture_match ? event.gesture_match : null;
        state.lastGestureMatch = match;

        if (!match || !state.registeredGestures.length) {
            handleGestureActions(null, 0);
            if (el.gestureDetectionBadge) el.gestureDetectionBadge.className = "gesture-badge gesture-badge-idle";
            if (el.gestureBadgeText) {
                el.gestureBadgeText.textContent = state.registeredGestures.length
                    ? "No gesture registered for this model"
                    : "No gesture registered (capture a slot on the right)";
            }
            if (el.gestureConfidenceText) el.gestureConfidenceText.textContent = "0.0%";
            if (el.gestureConfidenceBar) el.gestureConfidenceBar.style.width = "0%";
            renderReasoning(null, event);
            clearHeatmap();
            return;
        }

        // Temporal smoothing over the server's blended scores.
        const window = Math.max(1, Math.round(state.gestureSmoothing));
        const smoothed = {};
        for (const s of match.scores) {
            if (!state.gestureHistory[s.name]) state.gestureHistory[s.name] = [];
            const hist = state.gestureHistory[s.name];
            hist.push(s.combined);
            while (hist.length > window) hist.shift();
            smoothed[s.name] = hist.reduce((a, b) => a + b, 0) / hist.length;
        }

        // Client-side decision with the UI's own threshold and margin, mirroring the server rule.
        const ranked = match.scores
            .map((s) => ({ name: s.name, is_neutral: s.is_neutral, score: smoothed[s.name] || 0 }))
            .sort((a, b) => b.score - a.score);
        const best = ranked[0];
        const second = ranked[1] ? ranked[1].score : 0;
        const margin = best ? best.score - second : 0;
        let recognized = null;
        let waitReason = "";
        if (!best) {
            waitReason = "no scores";
        } else if (best.is_neutral) {
            waitReason = `neutral pose '${best.name}'`;
        } else if (best.score < state.gestureThreshold) {
            waitReason = `${best.name} ${(best.score * 100).toFixed(0)}% < threshold ${(state.gestureThreshold * 100).toFixed(0)}%`;
        } else if (margin < state.gestureMargin) {
            waitReason = `${best.name}: lead ${(margin * 100).toFixed(1)}% < ${(state.gestureMargin * 100).toFixed(0)}%`;
        } else {
            recognized = best.name;
        }

        const bestNonNeutral = ranked.filter((r) => !r.is_neutral)[0];
        const displayScore = bestNonNeutral ? Math.max(0, Math.min(1, bestNonNeutral.score)) : 0;
        const scorePct = (displayScore * 100).toFixed(1);
        if (el.gestureConfidenceText) el.gestureConfidenceText.textContent = `${scorePct}%`;
        if (el.gestureConfidenceBar) el.gestureConfidenceBar.style.width = `${scorePct}%`;

        if (el.gestureDetectionBadge) {
            el.gestureDetectionBadge.className = recognized ? "gesture-badge gesture-badge-detected" : "gesture-badge gesture-badge-idle";
        }
        if (el.gestureBadgeText) {
            el.gestureBadgeText.textContent = recognized ? `Detected: ${recognized}` : `Waiting (${waitReason})`;
        }

        // Live meters on every slot.
        for (let i = 1; i <= GESTURE_SLOT_COUNT; i++) {
            const g = gestureForSlot(i);
            const scoreValEl = document.getElementById(`slot-score-val-${i}`);
            const scoreBarEl = document.getElementById(`slot-score-bar-${i}`);
            if (!scoreValEl || !scoreBarEl) continue;
            if (g && smoothed[g.name] !== undefined) {
                const v = Math.max(0, Math.min(1, smoothed[g.name]));
                scoreValEl.textContent = `${(v * 100).toFixed(1)}%`;
                scoreBarEl.style.width = `${v * 100}%`;
                const active = recognized && g.name === recognized;
                scoreBarEl.style.backgroundColor = active ? "#4ade80" : "#38bdf8";
                scoreValEl.style.color = active ? "#4ade80" : "#38bdf8";
            } else {
                scoreValEl.textContent = "--%";
                scoreBarEl.style.width = "0%";
            }
        }

        renderReasoning(match, event, { recognized, margin, waitReason });
        drawHeatmap(match.patch_diff, match.grid_size);
        handleGestureActions(recognized, displayScore);
    }

    function fmtScore(v) {
        return typeof v === "number" ? v.toFixed(3) : "--";
    }

    function renderReasoning(match, event, decision) {
        if (!el.gestureReasoningBody) return;

        if (el.gestureMethodBadge) {
            el.gestureMethodBadge.textContent = `method: ${match ? match.method : "--"}`;
        }
        if (el.reasonLatency) el.reasonLatency.textContent = event && typeof event.latency_ms === "number" ? `${event.latency_ms.toFixed(1)} ms` : "--";
        if (el.reasonFrame) el.reasonFrame.textContent = event ? String(event.frame_index) : "--";

        if (!match) {
            el.gestureReasoningBody.innerHTML = `<tr><td colspan="5" class="reasoning-empty">Waiting for the stream and at least one registered gesture.</td></tr>`;
            if (el.gestureReasoningDecision) {
                el.gestureReasoningDecision.textContent = "Decision: --";
                el.gestureReasoningDecision.classList.remove("detected");
            }
            if (el.reasonMargin) el.reasonMargin.textContent = "--";
            if (el.reasonThreshold) el.reasonThreshold.textContent = "--";
            if (el.reasonGrid) el.reasonGrid.textContent = "--";
            return;
        }

        const bestName = match.scores.length ? match.scores[0].name : null;
        const rows = match.scores.map((s) => {
            const cls = [s.name === bestName ? "best" : "", s.is_neutral ? "neutral" : ""].join(" ").trim();
            const contrast = s.contrastive === null || s.contrastive === undefined ? "--" : fmtScore(s.contrastive);
            const pct = Math.max(0, Math.min(100, s.combined * 100));
            return `<tr class="${cls}">
                <td>${escapeHtml(s.name)}${s.is_neutral ? " (neutre)" : ""}</td>
                <td class="num">${s.sample_count}</td>
                <td class="num">${fmtScore(s.raw_cosine)}</td>
                <td class="num">${contrast}</td>
                <td><div class="score-cell"><span class="num">${fmtScore(s.combined)}</span><div class="score-track"><div class="score-fill" style="width:${pct.toFixed(1)}%"></div></div></div></td>
            </tr>`;
        });
        el.gestureReasoningBody.innerHTML = rows.join("");

        if (el.gestureReasoningDecision) {
            const serverPart = `Server: ${escapeHtml(match.reason)}`;
            let clientPart = "";
            if (decision) {
                clientPart = decision.recognized
                    ? `UI (smoothed over ${Math.round(state.gestureSmoothing)} frames): ${escapeHtml(decision.recognized)} detected`
                    : `UI (smoothed over ${Math.round(state.gestureSmoothing)} frames): ${escapeHtml(decision.waitReason)}`;
            }
            el.gestureReasoningDecision.innerHTML = `${serverPart}<br>${clientPart}`;
            el.gestureReasoningDecision.classList.toggle("detected", !!(decision && decision.recognized));
        }
        if (el.reasonMargin) el.reasonMargin.textContent = `${fmtScore(match.margin)} (required ${fmtScore(match.margin_required)})`;
        if (el.reasonThreshold) el.reasonThreshold.textContent = fmtScore(match.threshold);
        if (el.reasonGrid) el.reasonGrid.textContent = match.grid_size ? `${match.grid_size}x${match.grid_size} patches` : "n/a (video model)";
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = String(str);
        return div.innerHTML;
    }

    function handleGestureActions(gesture, score) {
        const slotThemes = {
            1: { name: "Cyan", color: "#38bdf8", bg: "rgba(56, 189, 248, 0.15)", border: "rgba(56, 189, 248, 0.5)", freq: 523.25 },
            2: { name: "Amethyste", color: "#c084fc", bg: "rgba(192, 132, 252, 0.15)", border: "rgba(192, 132, 252, 0.5)", freq: 659.25 },
            3: { name: "Emeraude", color: "#4ade80", bg: "rgba(74, 222, 128, 0.15)", border: "rgba(74, 222, 128, 0.5)", freq: 783.99 },
        };

        let activeSlot = null;
        if (gesture) {
            for (let i = 1; i < GESTURE_NEUTRAL_SLOT; i++) {
                if (slotName(i) === gesture) {
                    activeSlot = i;
                    break;
                }
            }
        }

        for (let i = 1; i <= GESTURE_SLOT_COUNT; i++) {
            const card = document.getElementById(`slot-card-${i}`);
            if (card) card.classList.toggle("matched-active", activeSlot === i);
        }

        if (gesture && activeSlot) {
            const theme = slotThemes[activeSlot] || slotThemes[1];

            if (el.gestureThemeText) el.gestureThemeText.textContent = `${gesture}: ${theme.name} (slot ${activeSlot})`;
            if (el.gestureThemePill) {
                el.gestureThemePill.style.backgroundColor = theme.bg;
                el.gestureThemePill.style.borderColor = theme.border;
                el.gestureThemePill.style.color = theme.color;
                el.gestureThemePill.style.boxShadow = `0 0 14px ${theme.bg}`;
            }
            if (el.gestureVideoWrapper) {
                el.gestureVideoWrapper.style.boxShadow = `0 0 24px ${theme.bg}`;
                el.gestureVideoWrapper.style.borderColor = theme.border;
            }

            const now = Date.now();
            if (state.gestureAudioEnabled && now - state.lastAudioToneTime > 450) {
                state.lastAudioToneTime = now;
                playGestureTone(theme.freq);
            }

            if (state.activeGestureHold === gesture) {
                const elapsedSec = (now - state.holdStartTime) / 1000.0;
                const pct = Math.min(100, (elapsedSec / 1.0) * 100);
                if (el.gestureHoldTimerText) el.gestureHoldTimerText.textContent = `Hold: ${Math.min(1.0, elapsedSec).toFixed(1)}s / 1.0s`;
                if (el.gestureHoldBar) {
                    el.gestureHoldBar.style.width = `${pct}%`;
                    el.gestureHoldBar.style.backgroundColor = theme.color;
                }
                if (elapsedSec >= 1.0 && !state.holdTriggered) {
                    state.holdTriggered = true;
                    state.slotHoldCounts[activeSlot] = (state.slotHoldCounts[activeSlot] || 0) + 1;
                    const counterEl = document.getElementById(`slot-counter-${activeSlot}`);
                    if (counterEl) counterEl.textContent = state.slotHoldCounts[activeSlot];
                    if (state.gestureAudioEnabled) playGestureTone(theme.freq * 1.5);
                }
            } else {
                state.activeGestureHold = gesture;
                state.holdStartTime = now;
                state.holdTriggered = false;
                if (el.gestureHoldTimerText) el.gestureHoldTimerText.textContent = "Hold: 0.0s / 1.0s";
                if (el.gestureHoldBar) el.gestureHoldBar.style.width = "0%";
            }
        } else {
            if (el.gestureThemeText) el.gestureThemeText.textContent = "Neutral palette";
            if (el.gestureThemePill) {
                el.gestureThemePill.style.backgroundColor = "transparent";
                el.gestureThemePill.style.borderColor = "var(--border-color)";
                el.gestureThemePill.style.color = "var(--text-dim)";
                el.gestureThemePill.style.boxShadow = "none";
            }
            if (el.gestureVideoWrapper) {
                el.gestureVideoWrapper.style.boxShadow = "none";
                el.gestureVideoWrapper.style.borderColor = "var(--border-subtle)";
            }
            state.activeGestureHold = null;
            state.holdStartTime = null;
            state.holdTriggered = false;
            if (el.gestureHoldTimerText) el.gestureHoldTimerText.textContent = "Hold: 0.0s / 1.0s";
            if (el.gestureHoldBar) el.gestureHoldBar.style.width = "0%";
        }
    }

    function playGestureTone(freq) {
        if (!state.gestureAudioEnabled) return;
        try {
            if (!state.audioContext) {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (AudioCtx) {
                    state.audioContext = new AudioCtx();
                }
            }
            if (state.audioContext && state.audioContext.state === "suspended") {
                state.audioContext.resume();
            }
            if (!state.audioContext) return;

            const ctx = state.audioContext;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(freq, ctx.currentTime);

            gain.gain.setValueAtTime(0.001, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(ctx.currentTime);
            osc.stop(ctx.currentTime + 0.20);
        } catch (e) {
            console.warn("Audio tone playback failed:", e);
        }
    }

    // ------------------------------------------------------------------
    // Robot Twin
    //
    // A raw WebGL renderer (no library, works offline in the desktop window)
    // draws a 6 DOF arm from the telemetry pushed by /api/robot/ws. Manual
    // commands go back over the same socket. Mode C snapshots the canvas and
    // posts it as the observation when the controller asks for one.
    // ------------------------------------------------------------------

    const ROBOT_DOF = 6;
    const ROBOT_JOINT_NAMES = ["J1 base", "J2 shoulder", "J3 elbow", "J4 wrist pitch", "J5 wrist roll", "J6 wrist rotate"];

    const robot = {
        ws: null,
        wsRetry: null,
        telemetry: null,
        shown: [0, 0, 0, 0, 0, 0],
        shownGripper: 0.5,
        gl: null,
        program: null,
        cube: null,
        lines: null,
        orbit: { yaw: 0.8, pitch: 0.45, dist: 1.6, dragging: false, lastX: 0, lastY: 0 },
        raf: null,
        lastSend: 0,
        pendingSend: null,
        lastObserve: 0,
        sliderBusy: false
    };

    // --- Small matrix toolkit (column major, like WebGL expects) -----------

    function m4identity() {
        return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    }

    function m4multiply(a, b) {
        const o = new Float32Array(16);
        for (let c = 0; c < 4; c++) {
            for (let r = 0; r < 4; r++) {
                o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
            }
        }
        return o;
    }

    function m4translate(x, y, z) {
        const m = m4identity();
        m[12] = x; m[13] = y; m[14] = z;
        return m;
    }

    function m4scale(x, y, z) {
        const m = m4identity();
        m[0] = x; m[5] = y; m[10] = z;
        return m;
    }

    function m4rotateX(a) {
        const c = Math.cos(a), s = Math.sin(a);
        const m = m4identity();
        m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
        return m;
    }

    function m4rotateY(a) {
        const c = Math.cos(a), s = Math.sin(a);
        const m = m4identity();
        m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
        return m;
    }

    function m4rotateZ(a) {
        const c = Math.cos(a), s = Math.sin(a);
        const m = m4identity();
        m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
        return m;
    }

    function m4perspective(fovy, aspect, near, far) {
        const f = 1 / Math.tan(fovy / 2);
        const m = new Float32Array(16);
        m[0] = f / aspect; m[5] = f;
        m[10] = (far + near) / (near - far); m[11] = -1;
        m[14] = (2 * far * near) / (near - far);
        return m;
    }

    function m4lookAt(eye, target, up) {
        const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
        const zl = Math.hypot(zx, zy, zz) || 1;
        const z = [zx / zl, zy / zl, zz / zl];
        const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
        const xl = Math.hypot(x[0], x[1], x[2]) || 1;
        x[0] /= xl; x[1] /= xl; x[2] /= xl;
        const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
        const m = m4identity();
        m[0] = x[0]; m[4] = x[1]; m[8] = x[2];
        m[1] = y[0]; m[5] = y[1]; m[9] = y[2];
        m[2] = z[0]; m[6] = z[1]; m[10] = z[2];
        m[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
        m[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
        m[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
        return m;
    }

    function m4transformPoint(m, p) {
        const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
        return [
            (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w,
            (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w,
            (m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]) / w
        ];
    }

    // --- WebGL setup -------------------------------------------------------

    const ROBOT_VS = `
        attribute vec3 a_pos;
        attribute vec3 a_normal;
        uniform mat4 u_mvp;
        uniform mat4 u_model;
        varying vec3 v_normal;
        void main() {
            v_normal = mat3(u_model) * a_normal;
            gl_Position = u_mvp * vec4(a_pos, 1.0);
        }`;

    const ROBOT_FS = `
        precision mediump float;
        uniform vec4 u_color;
        uniform float u_lit;
        varying vec3 v_normal;
        void main() {
            vec3 n = normalize(v_normal);
            vec3 l = normalize(vec3(0.4, 1.0, 0.6));
            float d = max(dot(n, l), 0.0);
            float shade = mix(1.0, 0.35 + 0.65 * d, u_lit);
            gl_FragColor = vec4(u_color.rgb * shade, u_color.a);
        }`;

    function robotCompile(gl, type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            throw new Error(gl.getShaderInfoLog(sh));
        }
        return sh;
    }

    function robotCubeMesh(gl) {
        // Unit cube centred at the origin, 6 faces x 2 triangles, with normals.
        const faces = [
            [[0, 0, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            [[0, 0, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]],
            [[0, 1, 0], [-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]],
            [[0, -1, 0], [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],
            [[1, 0, 0], [1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]],
            [[-1, 0, 0], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]
        ];
        const data = [];
        for (const [n, a, b, c, d] of faces) {
            for (const v of [a, b, c, a, c, d]) {
                data.push(v[0] * 0.5, v[1] * 0.5, v[2] * 0.5, n[0], n[1], n[2]);
            }
        }
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
        return { buf, count: 36 };
    }

    function robotGridMesh(gl) {
        const data = [];
        const half = 0.6, step = 0.1;
        for (let i = -half; i <= half + 1e-6; i += step) {
            data.push(i, 0, -half, 0, 1, 0, i, 0, half, 0, 1, 0);
            data.push(-half, 0, i, 0, 1, 0, half, 0, i, 0, 1, 0);
        }
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
        return { buf, count: data.length / 6 };
    }

    function robotInitGl(canvas) {
        const gl = canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: true });
        if (!gl) return null;
        const program = gl.createProgram();
        gl.attachShader(program, robotCompile(gl, gl.VERTEX_SHADER, ROBOT_VS));
        gl.attachShader(program, robotCompile(gl, gl.FRAGMENT_SHADER, ROBOT_FS));
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program));
        }
        gl.useProgram(program);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        robot.gl = gl;
        robot.program = program;
        robot.cube = robotCubeMesh(gl);
        robot.lines = robotGridMesh(gl);
        robot.loc = {
            pos: gl.getAttribLocation(program, "a_pos"),
            normal: gl.getAttribLocation(program, "a_normal"),
            mvp: gl.getUniformLocation(program, "u_mvp"),
            model: gl.getUniformLocation(program, "u_model"),
            color: gl.getUniformLocation(program, "u_color"),
            lit: gl.getUniformLocation(program, "u_lit")
        };
        gl.enableVertexAttribArray(robot.loc.pos);
        gl.enableVertexAttribArray(robot.loc.normal);
        return gl;
    }

    function robotBind(mesh) {
        const gl = robot.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buf);
        gl.vertexAttribPointer(robot.loc.pos, 3, gl.FLOAT, false, 24, 0);
        gl.vertexAttribPointer(robot.loc.normal, 3, gl.FLOAT, false, 24, 12);
    }

    function robotDrawBox(viewProj, model, size, color, alpha) {
        const gl = robot.gl;
        const m = m4multiply(model, m4scale(size[0], size[1], size[2]));
        gl.uniformMatrix4fv(robot.loc.model, false, m);
        gl.uniformMatrix4fv(robot.loc.mvp, false, m4multiply(viewProj, m));
        gl.uniform4f(robot.loc.color, color[0], color[1], color[2], alpha === undefined ? 1 : alpha);
        gl.uniform1f(robot.loc.lit, 1);
        robotBind(robot.cube);
        gl.drawArrays(gl.TRIANGLES, 0, robot.cube.count);
    }

    function robotDrawLines(viewProj, mesh, color, alpha) {
        const gl = robot.gl;
        gl.uniformMatrix4fv(robot.loc.model, false, m4identity());
        gl.uniformMatrix4fv(robot.loc.mvp, false, viewProj);
        gl.uniform4f(robot.loc.color, color[0], color[1], color[2], alpha === undefined ? 1 : alpha);
        gl.uniform1f(robot.loc.lit, 0);
        robotBind(mesh);
        gl.drawArrays(gl.LINES, 0, mesh.count);
    }

    function robotLineMesh(points) {
        const gl = robot.gl;
        const data = [];
        for (const p of points) data.push(p[0], p[1], p[2], 0, 1, 0);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
        return { buf, count: points.length };
    }

    // Kinematic chain: base yaw (J1), shoulder pitch (J2), elbow pitch (J3),
    // wrist pitch (J4), wrist roll (J5), wrist rotate (J6), two jaw gripper.
    // Link lengths are metres in a small SO-100 like arm.
    const ROBOT_LINKS = { base: 0.06, shoulder: 0.24, elbow: 0.22, wristPitch: 0.06, wristRoll: 0.05, jaw: 0.05 };

    function robotDrawArm(viewProj, joints, gripper, color, alpha) {
        const L = ROBOT_LINKS;
        // Base plate and turret
        robotDrawBox(viewProj, m4translate(0, 0.01, 0), [0.16, 0.02, 0.16], [0.25, 0.28, 0.34], alpha);
        let T = m4multiply(m4translate(0, 0.02, 0), m4rotateY(joints[0]));
        robotDrawBox(viewProj, m4multiply(T, m4translate(0, L.base / 2, 0)), [0.09, L.base, 0.09], color, alpha);
        // Shoulder
        T = m4multiply(T, m4multiply(m4translate(0, L.base, 0), m4rotateZ(joints[1])));
        robotDrawBox(viewProj, m4multiply(T, m4translate(0, L.shoulder / 2, 0)), [0.05, L.shoulder, 0.07], color, alpha);
        // Elbow
        T = m4multiply(T, m4multiply(m4translate(0, L.shoulder, 0), m4rotateZ(joints[2])));
        robotDrawBox(viewProj, m4multiply(T, m4translate(0, L.elbow / 2, 0)), [0.045, L.elbow, 0.06], color, alpha);
        // Wrist pitch
        T = m4multiply(T, m4multiply(m4translate(0, L.elbow, 0), m4rotateZ(joints[3])));
        robotDrawBox(viewProj, m4multiply(T, m4translate(0, L.wristPitch / 2, 0)), [0.04, L.wristPitch, 0.05], color, alpha);
        // Wrist roll around the link axis, then rotate
        T = m4multiply(T, m4multiply(m4translate(0, L.wristPitch, 0), m4rotateX(joints[4])));
        T = m4multiply(T, m4rotateY(joints[5]));
        robotDrawBox(viewProj, m4multiply(T, m4translate(0, L.wristRoll / 2, 0)), [0.05, L.wristRoll, 0.035], [0.55, 0.58, 0.64], alpha);
        // Gripper: two jaws whose gap follows the opening
        const gap = 0.006 + 0.03 * gripper;
        const jawBase = m4multiply(T, m4translate(0, L.wristRoll, 0));
        robotDrawBox(viewProj, m4multiply(jawBase, m4translate(gap / 2 + 0.006, L.jaw / 2, 0)), [0.012, L.jaw, 0.03], [0.85, 0.86, 0.9], alpha);
        robotDrawBox(viewProj, m4multiply(jawBase, m4translate(-gap / 2 - 0.006, L.jaw / 2, 0)), [0.012, L.jaw, 0.03], [0.85, 0.86, 0.9], alpha);
        // Effector tip in world space
        return m4transformPoint(jawBase, [0, L.jaw, 0]);
    }

    function robotRender() {
        const gl = robot.gl;
        if (!gl) return;
        const canvas = gl.canvas;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.04, 0.05, 0.07, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const o = robot.orbit;
        const eye = [
            o.dist * Math.cos(o.pitch) * Math.sin(o.yaw),
            0.25 + o.dist * Math.sin(o.pitch),
            o.dist * Math.cos(o.pitch) * Math.cos(o.yaw)
        ];
        const view = m4lookAt(eye, [0, 0.25, 0], [0, 1, 0]);
        const proj = m4perspective(0.8, canvas.width / canvas.height, 0.05, 20);
        const viewProj = m4multiply(proj, view);

        robotDrawLines(viewProj, robot.lines, [0.22, 0.25, 0.32], 1);
        // Axes
        const axes = robotLineMesh([[0, 0.001, 0], [0.2, 0.001, 0], [0, 0.001, 0], [0, 0.2, 0], [0, 0.001, 0], [0, 0.001, 0.2]]);
        robotDrawLines(viewProj, { buf: axes.buf, count: 2 }, [0.9, 0.3, 0.3], 1);
        gl.uniform4f(robot.loc.color, 0.3, 0.9, 0.4, 1);
        gl.drawArrays(gl.LINES, 2, 2);
        gl.uniform4f(robot.loc.color, 0.35, 0.55, 1.0, 1);
        gl.drawArrays(gl.LINES, 4, 2);
        gl.deleteBuffer(axes.buf);

        // Actual arm
        const tip = robotDrawArm(viewProj, robot.shown, robot.shownGripper, [0.55, 0.6, 0.68], 1);
        // Pending ghost (safety gate)
        const t = robot.telemetry;
        if (t && t.pending) {
            robotDrawArm(viewProj, t.pending.joints, t.pending.gripper, [0.98, 0.7, 0.2], 0.35);
        }
        // Effector projection to the ground and marker
        const proj1 = robotLineMesh([tip, [tip[0], 0.001, tip[2]]]);
        robotDrawLines(viewProj, proj1, [0.3, 0.9, 0.5], 0.8);
        gl.deleteBuffer(proj1.buf);
        robotDrawBox(viewProj, m4translate(tip[0], 0.004, tip[2]), [0.02, 0.004, 0.02], [0.3, 0.9, 0.5], 1);
    }

    function robotAnimate() {
        robot.raf = null;
        if (state.activeSection !== "robot") return;
        const t = robot.telemetry;
        if (t) {
            // Ease toward the last telemetry so 30 Hz updates look continuous.
            for (let i = 0; i < ROBOT_DOF; i++) robot.shown[i] += (t.joints[i] - robot.shown[i]) * 0.35;
            robot.shownGripper += (t.gripper - robot.shownGripper) * 0.35;
        }
        robotRender();
        robot.raf = requestAnimationFrame(robotAnimate);
    }

    // --- WebSocket telemetry ----------------------------------------------

    function robotConnectWs() {
        if (robot.ws || state.activeSection !== "robot") return;
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const tokenParam = state.authToken && state.authToken !== "no_auth" ? `?token=${encodeURIComponent(state.authToken)}` : "";
        const ws = new WebSocket(`${proto}//${location.host}/api/robot/ws${tokenParam}`);
        robot.ws = ws;
        ws.onopen = () => robotSetWsChip(true);
        ws.onmessage = (e) => {
            try {
                robotApplyTelemetry(JSON.parse(e.data));
            } catch (err) {
                console.warn("robot telemetry parse error", err);
            }
        };
        ws.onclose = () => {
            robot.ws = null;
            robotSetWsChip(false);
            if (state.activeSection === "robot") {
                robot.wsRetry = setTimeout(robotConnectWs, 1500);
            }
        };
        ws.onerror = () => ws.close();
    }

    function robotDisconnectWs() {
        if (robot.wsRetry) {
            clearTimeout(robot.wsRetry);
            robot.wsRetry = null;
        }
        if (robot.ws) {
            robot.ws.onclose = null;
            robot.ws.close();
            robot.ws = null;
        }
        robotSetWsChip(false);
    }

    function robotSetWsChip(on) {
        const chip = document.getElementById("robot-ws-chip");
        const text = document.getElementById("robot-ws-text");
        if (chip) chip.classList.toggle("on", on);
        if (text) text.textContent = on ? "telemetry 30 Hz" : "telemetry off";
    }

    function robotSendCommand(joints, gripper, approved) {
        const payload = { joints, gripper, approved: !!approved };
        if (robot.ws && robot.ws.readyState === WebSocket.OPEN) {
            const now = performance.now();
            if (now - robot.lastSend < 33) {
                robot.pendingSend = payload;
                if (!robot.sendTimer) {
                    robot.sendTimer = setTimeout(() => {
                        robot.sendTimer = null;
                        if (robot.pendingSend) {
                            robot.ws.send(JSON.stringify(robot.pendingSend));
                            robot.pendingSend = null;
                            robot.lastSend = performance.now();
                        }
                    }, 33);
                }
                return;
            }
            robot.lastSend = now;
            robot.ws.send(JSON.stringify(payload));
        } else {
            apiFetch("/api/robot/joints", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
                .then(async (res) => {
                    if (!res.ok) {
                        const err = await res.json().catch(() => ({}));
                        notify(`Robot: ${err.error || res.status}`, "error");
                    }
                })
                .catch((e) => notify(`Robot: ${e.message}`, "error"));
        }
    }

    // --- Telemetry to UI ----------------------------------------------------

    function robotApplyTelemetry(t) {
        const prev = robot.telemetry;
        robot.telemetry = t;
        if (!prev) {
            robot.shown = t.joints.slice();
            robot.shownGripper = t.gripper;
            robotBuildSliders(t.limits);
        }
        const readout = document.getElementById("robot-readout");
        if (readout) {
            readout.textContent = t.joints.map((v, i) => `J${i + 1} ${v >= 0 ? " " : ""}${v.toFixed(2)}`).join("  ")
                + `  gripper ${Math.round(t.gripper * 100)}%` + (t.last_error ? `   error: ${t.last_error}` : "");
        }
        // Backend and connection
        const vb = document.getElementById("btn-robot-backend-virtual");
        const pb = document.getElementById("btn-robot-backend-physical");
        if (vb && pb) {
            vb.classList.toggle("active", t.backend === "virtual");
            pb.classList.toggle("active", t.backend === "physical");
        }
        const chip = document.getElementById("robot-conn-chip");
        const ctext = document.getElementById("robot-conn-text");
        if (chip) chip.classList.toggle("on", t.connected);
        if (ctext) ctext.textContent = t.connected ? `${t.backend} connected` : `${t.backend} disconnected`;
        // Mode and gate
        const modeSel = document.getElementById("select-robot-mode");
        if (modeSel && modeSel.value !== t.mode && document.activeElement !== modeSel) modeSel.value = t.mode;
        const gate = document.getElementById("toggle-robot-gate");
        if (gate) {
            gate.checked = t.safety_gate;
            gate.disabled = t.backend === "physical";
        }
        // E-stop
        const estopBanner = document.getElementById("robot-estop-banner");
        const resetBtn = document.getElementById("btn-robot-reset");
        if (estopBanner) estopBanner.style.display = t.estop ? "block" : "none";
        if (resetBtn) resetBtn.style.display = t.estop ? "inline-flex" : "none";
        // Pending (Mode B)
        const pendingBanner = document.getElementById("robot-pending-banner");
        if (pendingBanner) pendingBanner.style.display = t.pending ? "flex" : "none";
        // Sliders follow the targets unless the user is dragging
        if (!robot.sliderBusy) {
            for (let i = 0; i < ROBOT_DOF; i++) {
                const s = document.getElementById(`slider-robot-j${i + 1}`);
                const v = document.getElementById(`val-robot-j${i + 1}`);
                if (s) s.value = t.targets[i];
                if (v) v.textContent = `${(t.targets[i] * 180 / Math.PI).toFixed(1)} deg  (${t.targets[i].toFixed(3)} rad)`;
            }
            const g = document.getElementById("slider-robot-gripper");
            const gv = document.getElementById("val-robot-gripper");
            if (g) g.value = t.gripper_target;
            if (gv) gv.textContent = `${Math.round(t.gripper_target * 100)}%`;
        }
        const ramp = document.getElementById("robot-ramp-text");
        if (ramp) ramp.textContent = `max ${t.max_rad_per_s.toFixed(2)} rad/s`;
        const lg = document.getElementById("robot-last-gesture");
        if (lg) lg.textContent = `last gesture: ${t.last_gesture || "none"}`;
        // Goal (Mode C)
        const goal = t.goal || {};
        const et = document.getElementById("robot-energy-text");
        const eb = document.getElementById("robot-energy-bar");
        const best = document.getElementById("robot-energy-best");
        const rounds = document.getElementById("robot-goal-rounds");
        const step = document.getElementById("robot-goal-step");
        const phase = document.getElementById("robot-goal-phase");
        const num = (v) => (v !== null && v !== undefined ? v.toFixed(4) : "--");
        if (et) et.textContent = num(goal.current_energy);
        if (eb) eb.style.width = `${Math.round((goal.convergence || 0) * 100)}%`;
        if (best) best.textContent = num(goal.best_energy);
        if (rounds) rounds.textContent = String(goal.steps || 0);
        if (step) step.textContent = `${(goal.step_rad || 0).toFixed(3)} rad`;
        if (phase) phase.textContent = goal.phase || "idle";
        const pred = document.getElementById("robot-energy-pred");
        const policy = document.getElementById("robot-goal-policy");
        const trans = document.getElementById("robot-world-transitions");
        const fit = document.getElementById("robot-world-fit");
        const note = document.getElementById("robot-learn-note");
        if (pred) pred.textContent = num(goal.predicted_energy);
        if (policy) policy.textContent = goal.policy || "--";
        if (trans) trans.textContent = goal.world ? `${goal.world.transitions}${goal.world.ready ? "" : " (need 12)"}` : "0";
        if (fit) fit.textContent = goal.world && goal.world.fit_error !== null && goal.world.fit_error !== undefined ? `${(100 * (1 - Math.min(1, goal.world.fit_error))).toFixed(0)}%` : "--";
        if (note) {
            if (t.last_error && t.last_error.startsWith("Learning needs the camera")) note.textContent = t.last_error;
            else if ((t.mode === "exploring" || t.mode === "goal_seeking") && t.backend === "virtual") note.textContent = "Virtual arm: the camera does not see it, so the model only learns what changes in front of the camera. Use the physical arm for real learning.";
            else note.textContent = "";
        }
        if (!robot.raf && state.activeSection === "robot") robot.raf = requestAnimationFrame(robotAnimate);
    }

    function robotBuildSliders(limits) {
        const host = document.getElementById("robot-joint-sliders");
        if (!host || host.childElementCount) return;
        for (let i = 0; i < ROBOT_DOF; i++) {
            const row = document.createElement("div");
            row.className = "sensitivity-row";
            row.innerHTML = `
                <div class="sensitivity-label-row">
                    <label for="slider-robot-j${i + 1}">${ROBOT_JOINT_NAMES[i]}</label>
                    <span class="joint-value" id="val-robot-j${i + 1}">0.0 deg (0.000 rad)</span>
                </div>
                <input type="range" id="slider-robot-j${i + 1}" class="range-slider" min="${limits.min[i]}" max="${limits.max[i]}" step="0.005" value="0">`;
            host.appendChild(row);
            const slider = row.querySelector("input");
            slider.addEventListener("pointerdown", () => { robot.sliderBusy = true; });
            slider.addEventListener("pointerup", () => { robot.sliderBusy = false; });
            slider.addEventListener("input", robotSlidersChanged);
            slider.addEventListener("change", () => { robot.sliderBusy = false; });
        }
        const g = document.getElementById("slider-robot-gripper");
        if (g) {
            g.addEventListener("pointerdown", () => { robot.sliderBusy = true; });
            g.addEventListener("pointerup", () => { robot.sliderBusy = false; });
            g.addEventListener("input", robotSlidersChanged);
        }
    }

    function robotSlidersChanged() {
        const joints = [];
        for (let i = 0; i < ROBOT_DOF; i++) {
            const s = document.getElementById(`slider-robot-j${i + 1}`);
            const v = parseFloat(s ? s.value : "0");
            joints.push(v);
            const label = document.getElementById(`val-robot-j${i + 1}`);
            if (label) label.textContent = `${(v * 180 / Math.PI).toFixed(1)} deg  (${v.toFixed(3)} rad)`;
        }
        const g = parseFloat(document.getElementById("slider-robot-gripper").value);
        const gv = document.getElementById("val-robot-gripper");
        if (gv) gv.textContent = `${Math.round(g * 100)}%`;
        robotSendCommand(joints, g, false);
    }

    async function robotPost(path, body, okMsg) {
        try {
            const res = await apiFetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.error || `${res.status}`);
            if (okMsg) notify(okMsg, "success", 2500);
            if (data && data.joints) robotApplyTelemetry(data);
            return data;
        } catch (e) {
            notify(`Robot: ${e.message}`, "error");
            return null;
        }
    }

    function setupRobotTwin() {
        const canvas = document.getElementById("robot-canvas");
        if (!canvas) return;
        try {
            if (!robotInitGl(canvas)) {
                notify("WebGL is not available in this window; the robot twin cannot render.", "warning", 8000);
            }
        } catch (e) {
            notify(`WebGL init failed: ${e.message}`, "error");
        }
        // Orbit controls
        canvas.addEventListener("pointerdown", (e) => {
            robot.orbit.dragging = true;
            robot.orbit.lastX = e.clientX;
            robot.orbit.lastY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener("pointermove", (e) => {
            if (!robot.orbit.dragging) return;
            robot.orbit.yaw -= (e.clientX - robot.orbit.lastX) * 0.01;
            robot.orbit.pitch = Math.max(0.05, Math.min(1.4, robot.orbit.pitch + (e.clientY - robot.orbit.lastY) * 0.01));
            robot.orbit.lastX = e.clientX;
            robot.orbit.lastY = e.clientY;
        });
        const stopDrag = () => { robot.orbit.dragging = false; };
        canvas.addEventListener("pointerup", stopDrag);
        canvas.addEventListener("pointercancel", stopDrag);
        canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            robot.orbit.dist = Math.max(0.6, Math.min(4, robot.orbit.dist * (e.deltaY > 0 ? 1.1 : 0.9)));
        }, { passive: false });

        const bind = (id, fn) => {
            const n = document.getElementById(id);
            if (n) n.addEventListener("click", fn);
        };
        bind("btn-robot-backend-virtual", () => robotPost("/api/robot/target", { backend: "virtual" }, "Backend: WebGL simulator"));
        bind("btn-robot-backend-physical", async () => {
            const ok = await confirmDialog("Switch to the physical arm? Every command will be held by the safety gate until you approve it.", { title: "Physical backend", okLabel: "Switch", danger: false });
            if (ok) robotPost("/api/robot/target", { backend: "physical" }, "Backend: physical arm");
        });
        bind("btn-robot-estop", () => robotPost("/api/robot/e-stop", null, "Emergency stop engaged"));
        bind("btn-robot-reset", async () => {
            const ok = await confirmDialog("Reset the emergency stop? Confirm the arm and its surroundings were inspected.", { title: "Reset safety", okLabel: "Reset", danger: true });
            if (ok) robotPost("/api/robot/reset-safety", null, "Safety reset");
        });
        bind("btn-robot-approve", () => robotPost("/api/robot/approve", null, "Approved: executing on hardware"));
        bind("btn-robot-discard", () => robotPost("/api/robot/joints", { joints: robot.telemetry ? robot.telemetry.targets : [0, 0, 0, 0, 0, 0], gripper: robot.telemetry ? robot.telemetry.gripper_target : 0.5, approved: true }, "Pending command discarded"));
        bind("btn-robot-home", () => robotSendCommand([0, 0, 0, 0, 0, 0], 0.5, false));
        bind("btn-robot-goal", async () => {
            await robotPost("/api/robot/goal", {}, "Goal captured from the camera");
        });
        bind("btn-robot-world-clear", async () => {
            const ok = await confirmDialog("Forget every learned transition? The arm will have to explore again before it can plan.", { title: "Forget world model", okLabel: "Forget" });
            if (!ok) return;
            const res = await apiFetch("/api/robot/world-model", { method: "DELETE" });
            if (res.ok) notify("World model cleared", "info", 2500);
        });
        bind("btn-robot-goal-clear", async () => {
            const res = await apiFetch("/api/robot/goal", { method: "DELETE" });
            if (res.ok) robotApplyTelemetry(await res.json());
        });
        bind("btn-api-robot", () => {
            const t = robot.telemetry;
            showApiDialog("Command the arm", "Inference role. Joints in radians within the limits, gripper 0 to 1. With the physical backend the command waits for POST /api/robot/approve unless approved is true. Telemetry: GET /api/robot/status or the WebSocket /api/robot/ws.",
                { method: "POST", path: "/api/robot/joints", json: { joints: t ? t.targets.map((v) => +v.toFixed(3)) : [0, 0, 0, 0, 0, 0], gripper: t ? +t.gripper_target.toFixed(2) : 0.5, approved: false } });
        });
        const modeSel = document.getElementById("select-robot-mode");
        if (modeSel) modeSel.addEventListener("change", () => robotPost("/api/robot/mode", { mode: modeSel.value }, `Mode: ${modeSel.options[modeSel.selectedIndex].text}`));
        const gate = document.getElementById("toggle-robot-gate");
        if (gate) gate.addEventListener("change", () => robotPost("/api/robot/mode", { mode: modeSel ? modeSel.value : "manual", safety_gate: gate.checked }));
        bind("btn-robot-gesture-map-save", async () => {
            const ta = document.getElementById("robot-gesture-map");
            try {
                const map = JSON.parse(ta.value);
                const res = await apiFetch("/api/robot/gesture-map", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(map) });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.error || `${res.status}`);
                notify(`Gesture map saved (${data.entries} entries)`, "success");
            } catch (e) {
                notify(`Gesture map: ${e.message}`, "error");
            }
        });
    }

    async function robotEnterSection() {
        robotConnectWs();
        try {
            const res = await apiFetch("/api/robot/gesture-map");
            if (res.ok) {
                const ta = document.getElementById("robot-gesture-map");
                if (ta && !ta.value) ta.value = JSON.stringify(await res.json(), null, 2);
            }
        } catch (e) {
            console.warn("gesture map fetch failed", e);
        }
        if (!robot.raf) robot.raf = requestAnimationFrame(robotAnimate);
    }

    function robotLeaveSection() {
        robotDisconnectWs();
        if (robot.raf) {
            cancelAnimationFrame(robot.raf);
            robot.raf = null;
        }
    }

    // Kickoff
    document.addEventListener("DOMContentLoaded", init);
})();
