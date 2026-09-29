/* ================================================
   OralGuard AI — Production Application Logic
   ================================================ */

(function () {
  'use strict';

  // ── Configuration ──
  const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
    ? window.location.origin
    : 'http://localhost:8000';
  const API_PREFIX = '/api/v1';

  // ── State ──
  let currentLanguage = 'en';
  let currentScreen = 'splash';
  let cameraStream = null;
  let currentImageBlob = null;
  let currentImageDataUrl = null;
  let currentScreeningId = null;
  let questionnaireList = [];
  let currentQuestionIdx = 0;
  let questionnaireAnswers = {};
  let currentResultData = null;

  // ── Bilingual Dictionary ──
  const i18n = {
    en: {
      appName: 'OralGuard AI',
      subtitle: 'AI-Powered Oral Lesion Screening',
      scanLesion: 'Scan Your Oral Lesion',
      scanDesc: 'Take or upload a photo of your oral lesion for AI-powered screening and clinical risk assessment',
      startScan: 'Start Scan',
      uploadImage: 'Upload Image',
      featureAi: 'AI Screening',
      featureAiDesc: 'Deep learning analysis across 12+ oral lesion categories',
      featureReport: 'Detailed Report',
      featureReportDesc: 'Differential diagnosis matrix with composite risk score',
      featureReferral: 'Smart Referral',
      featureReferralDesc: 'Connect with nearby dental specialists and clinics',
      positionFrame: 'Position the oral lesion within the frame',
      analyzingTitle: 'Analyzing Lesion',
      analyzingDesc: 'AI is examining image features, margins, and textures...',
      stepQuality: 'Validating image quality & lighting',
      stepDetect: 'Detecting lesion boundaries & localization',
      stepFeatures: 'Extracting clinical visual biomarkers',
      stepClassify: 'Classifying lesion type & subtypes',
      stepGradcam: 'Generating Grad-CAM explainability heatmap',
      clinicalHistory: 'Clinical History',
      aiFinding: 'AI Initial Finding:',
      prev: 'Previous',
      skip: 'Skip',
      next: 'Next',
      submit: 'Submit & Get Results',
      diagnosticAssessment: 'Diagnostic Assessment',
      primaryFinding: 'Primary Assessment',
      confidence: 'Confidence',
      riskLevel: 'Risk Level',
      riskScore: 'Risk Score',
      lowRisk: 'LOW',
      medRisk: 'MEDIUM',
      highRisk: 'HIGH',
      urgRisk: 'URGENT',
      differentials: 'Differential Diagnosis Matrix',
      detectedFeatures: 'Detected Visual Biomarkers',
      recommendations: 'Clinical Recommendations',
      downloadPdf: 'Download PDF Report',
      shareWhatsapp: 'Share via WhatsApp',
      nearbyClinics: 'Nearby Clinics',
      newScreening: 'New Screening',
      historyTitle: 'Screening History',
      noHistory: 'No screening records found.',
      dashTitle: 'Health Surveillance Dashboard',
      totalScreenings: 'Total Screenings',
      todayScreenings: 'Today',
      malignancyRate: 'Malignancy Flag %',
      referralRate: 'Referral Rate',
      disclaimer: 'This is an AI-assisted screening tool and does not replace professional clinical examination. Histopathological examination (biopsy) is required for definitive diagnosis of malignancy.',
    },
    hi: {
      appName: 'ओरलगार्ड एआई',
      subtitle: 'एआई-संचालित मुख घाव स्क्रीनिंग प्रणाली',
      scanLesion: 'अपने मुंह के छाले / घाव की जांच करें',
      scanDesc: 'एआई विश्लेषण और जोखिम मूल्यांकन के लिए अपने मुंह के छाले या घाव की फोटो खींचें या अपलोड करें',
      startScan: 'जांच शुरू करें',
      uploadImage: 'फोटो अपलोड करें',
      featureAi: 'एआई स्क्रीनिंग',
      featureAiDesc: '12+ प्रकार के मौखिक घावों का डीप लर्निंग विश्लेषण',
      featureReport: 'विस्तृत रिपोर्ट',
      featureReportDesc: 'विभेदक निदान और समग्र जोखिम स्कोर प्राप्त करें',
      featureReferral: 'विशेषज्ञ परामर्श',
      featureReferralDesc: 'निकटतम दंत चिकित्सक या कैंसर विशेषज्ञ से संपर्क',
      positionFrame: 'घाव या छाले को फ्रेम के बीच में रखें',
      analyzingTitle: 'घाव का विश्लेषण जारी है',
      analyzingDesc: 'एआई छवि की गुणवत्ता, सीमाओं और बनावट की जांच कर रहा है...',
      stepQuality: 'छवि गुणवत्ता और प्रकाश की जांच',
      stepDetect: 'घाव की सीमाओं की पहचान',
      stepFeatures: 'दृश्य बायोमार्कर का निष्कर्षण',
      stepClassify: 'घाव के प्रकार का वर्गीकरण',
      stepGradcam: 'एआई हीटमैप (Grad-CAM) तैयार करना',
      clinicalHistory: 'चिकित्सीय इतिहास',
      aiFinding: 'एआई प्राथमिक अवलोकन:',
      prev: 'पिछला',
      skip: 'छोड़ें',
      next: 'अगला',
      submit: 'जमा करें और रिपोर्ट देखें',
      diagnosticAssessment: 'नैदानिक मूल्यांकन',
      primaryFinding: 'प्राथमिक निष्कर्ष',
      confidence: 'सटीकता विश्वास',
      riskLevel: 'जोखिम स्तर',
      riskScore: 'जोखिम स्कोर',
      lowRisk: 'निम्न (कम)',
      medRisk: 'मध्यम',
      highRisk: 'उच्च (गंभीर)',
      urgRisk: 'तत्काल परामर्श',
      differentials: 'संभावित विभेदक निदान',
      detectedFeatures: 'पहचाने गए दृश्य लक्षण',
      recommendations: 'चिकित्सीय सलाह व निर्देश',
      downloadPdf: 'पीडीएफ रिपोर्ट डाउनलोड करें',
      shareWhatsapp: 'व्हाट्सएप पर साझा करें',
      nearbyClinics: 'निकटतम अस्पताल',
      newScreening: 'नई जांच शुरू करें',
      historyTitle: 'पिछली जांचों का इतिहास',
      noHistory: 'कोई पिछला रिकॉर्ड नहीं मिला।',
      dashTitle: 'सार्वजनिक स्वास्थ्य डैशबोर्ड',
      totalScreenings: 'कुल स्क्रीनिंग',
      todayScreenings: 'आज की स्क्रीनिंग',
      malignancyRate: 'कैंसर संशय दर %',
      referralRate: 'रेफरल दर %',
      disclaimer: 'यह एक एआई-सहायता प्राप्त स्क्रीनिंग उपकरण है और पेशेवर दंत चिकित्सक की जांच का विकल्प नहीं है। संदिग्ध घाव की पुष्टि हेतु बायोप्सी आवश्यक है।',
    }
  };

  // ── DOM References ──
  const screens = {
    splash: document.getElementById('screen-splash'),
    home: document.getElementById('screen-home'),
    scan: document.getElementById('screen-scan'),
    analyzing: document.getElementById('screen-analyzing'),
    questionnaire: document.getElementById('screen-questionnaire'),
    results: document.getElementById('screen-results'),
    history: document.getElementById('screen-history'),
    dashboard: document.getElementById('screen-dashboard'),
  };

  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  const fileInput = document.getElementById('file-input');

  // Camera & Previews
  const cameraFeed = document.getElementById('camera-feed');
  const uploadPreview = document.getElementById('upload-preview');
  const scanCanvas = document.getElementById('scan-canvas');
  const analyzingImage = document.getElementById('analyzing-image');

  // Questionnaire Elements
  const qHeaderTitle = document.getElementById('q-header-title');
  const qProgressBadge = document.getElementById('q-progress-badge');
  const qProgressFill = document.getElementById('q-progress-fill');
  const qProgressPct = document.getElementById('q-progress-pct');
  const aiObsDesc = document.getElementById('ai-obs-desc');
  const qCategoryTag = document.getElementById('q-category-tag');
  const qQuestionText = document.getElementById('q-question-text');
  const qRationaleText = document.getElementById('q-rationale-text');
  const qOptionsContainer = document.getElementById('q-options-container');
  const btnQPrev = document.getElementById('btn-q-prev');
  const btnQSkip = document.getElementById('btn-q-skip');
  const btnQNext = document.getElementById('btn-q-next');
  const txtQPrev = document.getElementById('txt-q-prev');
  const txtQSkip = document.getElementById('txt-q-skip');
  const txtQNext = document.getElementById('txt-q-next');

  // Results Elements
  const resultScannedImg = document.getElementById('result-scanned-img');
  const resultHeatmapImg = document.getElementById('result-heatmap-img');
  const resultSegImg = document.getElementById('result-seg-img');
  const cardScannedImg = document.getElementById('card-scanned-img');
  const cardHeatmapImg = document.getElementById('card-heatmap-img');
  const cardSegImg = document.getElementById('card-seg-img');
  const riskBadge = document.getElementById('risk-badge');
  const riskLabel = document.getElementById('risk-label');
  const riskValue = document.getElementById('risk-value');
  const riskScoreEl = document.getElementById('risk-score');
  const diagnosisName = document.getElementById('diagnosis-name');
  const diagnosisSubtype = document.getElementById('diagnosis-subtype');
  const confidencePct = document.getElementById('confidence-pct');
  const confidenceFill = document.getElementById('confidence-fill');
  const featuresGrid = document.getElementById('features-grid');
  const diffList = document.getElementById('diff-list');
  const recommendationList = document.getElementById('recommendation-list');

  // Buttons
  const btnLangToggle = document.getElementById('btn-lang-toggle');
  const langCurrent = document.getElementById('lang-current');
  const btnMenu = document.getElementById('btn-menu');
  const btnStartScan = document.getElementById('btn-start-scan');
  const btnUpload = document.getElementById('btn-upload');
  const btnBackScan = document.getElementById('btn-back-scan');
  const btnCapture = document.getElementById('btn-capture');
  const btnGallery = document.getElementById('btn-gallery');
  const btnSwitchCam = document.getElementById('btn-switch-cam');
  const btnBackQ = document.getElementById('btn-back-q');
  const btnBackResults = document.getElementById('btn-back-results');
  const btnDownloadPdf = document.getElementById('btn-download-pdf');
  const btnShareDoctor = document.getElementById('btn-share-doctor');
  const btnNearbyClinic = document.getElementById('btn-nearby-clinic');
  const btnRescan = document.getElementById('btn-rescan');
  const btnHistory = document.getElementById('btn-history');
  const navScan = document.getElementById('nav-scan');
  const navHistory = document.getElementById('nav-history');
  const btnBackHistory = document.getElementById('btn-back-history');
  const btnRefreshHistory = document.getElementById('btn-refresh-history');
  const btnClearHistory = document.getElementById('btn-clear-history');
  const historyList = document.getElementById('history-list');
  const navDashboard = document.getElementById('nav-dashboard');
  const btnBackDashboard = document.getElementById('btn-back-dashboard');
  const btnExportCsv = document.getElementById('btn-export-csv');
  const mainBottomNav = document.getElementById('main-bottom-nav');

  // ── Navigation ──
  function showScreen(name) {
    Object.values(screens).forEach(s => s && s.classList.remove('active'));
    if (screens[name]) {
      screens[name].classList.add('active');
      currentScreen = name;
    }
    // Update bottom nav highlights and visibility (mobile)
    const hideOnScreens = ['splash', 'scan', 'analyzing', 'questionnaire'];
    if (mainBottomNav) {
      if (hideOnScreens.includes(name)) {
        mainBottomNav.classList.add('nav-hidden');
      } else {
        mainBottomNav.classList.remove('nav-hidden');
      }
    }
    document.querySelectorAll('.bottom-nav .nav-item').forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-screen') === name);
    });

    // Update desktop nav highlights
    document.querySelectorAll('.desktop-nav-item').forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-screen') === name);
    });
  }

  // Auto transition from splash
  setTimeout(() => showScreen('home'), 1800);

  // ── Language Toggle ──
  function setLanguage(lang) {
    currentLanguage = lang;
    if (langCurrent) langCurrent.textContent = lang.toUpperCase();
    const langCurrentDesk = document.getElementById('lang-current-desk');
    if (langCurrentDesk) langCurrentDesk.textContent = lang.toUpperCase();
    const t = i18n[lang];

    // Update static labels if visible
    if (qHeaderTitle) qHeaderTitle.textContent = t.clinicalHistory;
    if (txtQPrev) txtQPrev.textContent = t.prev;
    if (txtQSkip) txtQSkip.textContent = t.skip;
    if (txtQNext) txtQNext.textContent = currentQuestionIdx === questionnaireList.length - 1 ? t.submit : t.next;

    // Rerender question if currently on questionnaire
    if (currentScreen === 'questionnaire' && questionnaireList.length > 0) {
      renderQuestion(currentQuestionIdx);
    }
  }

  btnLangToggle?.addEventListener('click', () => {
    setLanguage(currentLanguage === 'en' ? 'hi' : 'en');
  });

  const btnLangToggleDesk = document.getElementById('btn-lang-toggle-desk');
  btnLangToggleDesk?.addEventListener('click', () => {
    setLanguage(currentLanguage === 'en' ? 'hi' : 'en');
  });

  // Desktop Header Navigation Links
  document.getElementById('desktop-brand-home')?.addEventListener('click', () => {
    stopCamera();
    showScreen('home');
  });
  document.getElementById('desk-nav-home')?.addEventListener('click', () => {
    stopCamera();
    showScreen('home');
  });
  document.getElementById('desk-nav-scan')?.addEventListener('click', () => {
    showScreen('scan');
    startCamera();
  });
  document.getElementById('desk-nav-history')?.addEventListener('click', () => {
    loadHistory();
  });
  document.getElementById('desk-nav-dashboard')?.addEventListener('click', () => {
    loadDashboard();
  });
  document.getElementById('btn-desk-new-scan')?.addEventListener('click', () => {
    showScreen('scan');
    startCamera();
  });

  // ── Sidebar Controls ──
  function openSidebar() {
    sidebar.classList.add('active');
    sidebarOverlay.classList.add('active');
  }

  function closeSidebar() {
    sidebar.classList.remove('active');
    sidebarOverlay.classList.remove('active');
  }

  btnMenu?.addEventListener('click', openSidebar);
  sidebarOverlay?.addEventListener('click', closeSidebar);

  // ── Camera Implementation ──
  let facingMode = 'environment';

  async function startCamera() {
    try {
      const constraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      };
      cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraFeed.srcObject = cameraStream;
      cameraFeed.classList.remove('hidden');
      uploadPreview.classList.add('hidden');
    } catch (err) {
      console.warn('Camera inaccessible, fallback to gallery/upload:', err.message);
      cameraFeed.classList.add('hidden');
      fileInput.click();
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
      cameraFeed.srcObject = null;
    }
  }

  btnSwitchCam?.addEventListener('click', () => {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    startCamera();
  });

  function captureFrame() {
    const canvas = scanCanvas;
    const video = cameraFeed;

    if (video.videoWidth === 0) {
      fileInput.click();
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      currentImageBlob = blob;
      currentImageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
      stopCamera();
      initiateScreening(blob);
    }, 'image/jpeg', 0.9);
  }

  // ── File Upload Handler ──
  function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) return;

    currentImageBlob = file;
    const reader = new FileReader();
    reader.onload = e => {
      currentImageDataUrl = e.target.result;
      uploadPreview.src = currentImageDataUrl;
      uploadPreview.classList.remove('hidden');
      cameraFeed.classList.add('hidden');
      stopCamera();

      setTimeout(() => initiateScreening(file), 400);
    };
    reader.readAsDataURL(file);
  }

  fileInput?.addEventListener('change', function () {
    if (this.files && this.files[0]) {
      handleFileSelect(this.files[0]);
    }
  });

  // ── Initiate Screening & Upload ──
  async function initiateScreening(imageFileOrBlob) {
    analyzingImage.src = currentImageDataUrl;
    showScreen('analyzing');

    // UI Steps progression
    const stepDetect = document.getElementById('step-detect');
    const stepFeatures = document.getElementById('step-features');
    const stepClassify = document.getElementById('step-classify');

    [stepDetect, stepFeatures, stepClassify].forEach(s => {
      s?.classList.remove('active', 'done');
      s?.querySelector('.step-check')?.classList.add('hidden');
    });
    stepDetect?.classList.add('active');

    const formData = new FormData();
    formData.append('file', imageFileOrBlob, 'oral_capture.jpg');

    try {
      const uploadResp = await fetch(`${API_BASE}${API_PREFIX}/screening/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResp.ok) {
        const errorData = await uploadResp.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || errorData.detail || 'Upload failed');
      }

      const uploadData = await uploadResp.json();
      currentScreeningId = uploadData.screening_id;

      // Animate steps
      setTimeout(() => {
        stepDetect?.classList.remove('active');
        stepDetect?.classList.add('done');
        stepDetect?.querySelector('.step-check')?.classList.remove('hidden');
        stepFeatures?.classList.add('active');
      }, 700);

      setTimeout(() => {
        stepFeatures?.classList.remove('active');
        stepFeatures?.classList.add('done');
        stepFeatures?.querySelector('.step-check')?.classList.remove('hidden');
        stepClassify?.classList.add('active');
      }, 1400);

      setTimeout(async () => {
        stepClassify?.classList.remove('active');
        stepClassify?.classList.add('done');
        stepClassify?.querySelector('.step-check')?.classList.remove('hidden');

        // Fetch dynamic questionnaire
        await loadQuestionnaire(currentScreeningId);
      }, 2000);

    } catch (err) {
      console.error('Screening Error:', err);
      alert(`Screening could not be completed: ${err.message}`);
      showScreen('home');
    }
  }

  // ── Load Questionnaire from API ──
  async function loadQuestionnaire(screeningId) {
    try {
      const resp = await fetch(`${API_BASE}${API_PREFIX}/screening/${screeningId}/questionnaire`);
      if (!resp.ok) throw new Error('Failed to load clinical questions');

      const data = await resp.json();
      questionnaireList = data.questions || [];
      currentQuestionIdx = 0;
      questionnaireAnswers = {};

      if (aiObsDesc && data.initial_assessment) {
        const diag = data.initial_assessment.primary_diagnosis.replace('_', ' ');
        const conf = Math.round((data.initial_assessment.confidence || 0.85) * 100);
        aiObsDesc.textContent = `Preliminary pattern points towards ${diag} (${conf}% visual confidence). Refining via history.`;
      }

      renderQuestion(0);
      showScreen('questionnaire');
    } catch (err) {
      console.warn('Questionnaire error, fetching direct results:', err);
      fetchResults(screeningId);
    }
  }

  // ── Render Question ──
  function renderQuestion(idx) {
    if (!questionnaireList || idx >= questionnaireList.length) return;

    const q = questionnaireList[idx];
    const t = i18n[currentLanguage];

    // Progress
    const total = questionnaireList.length;
    const pct = Math.round(((idx + 1) / total) * 100);
    qProgressBadge.textContent = `${idx + 1} / ${total}`;
    qProgressFill.style.width = `${pct}%`;
    qProgressPct.textContent = `${pct}%`;

    // Category & Texts
    qCategoryTag.textContent = q.category.replace('_', ' ').toUpperCase();
    qQuestionText.textContent = q.question;
    qRationaleText.textContent = q.clinical_rationale || 'Essential clinical discriminator.';

    // Nav button state
    btnQPrev.disabled = idx === 0;
    txtQPrev.textContent = t.prev;
    txtQSkip.textContent = t.skip;
    txtQNext.textContent = idx === total - 1 ? t.submit : t.next;

    // Render input controls
    qOptionsContainer.innerHTML = '';
    const currentVal = questionnaireAnswers[q.id];

    if (q.question_type === 'boolean') {
      const options = [
        { label: currentLanguage === 'hi' ? 'हाँ (Yes)' : 'Yes', value: true },
        { label: currentLanguage === 'hi' ? 'नहीं (No)' : 'No', value: false },
      ];
      options.forEach(opt => {
        const pill = document.createElement('div');
        pill.className = `q-option-pill ${currentVal === opt.value ? 'selected' : ''}`;
        pill.innerHTML = `
          <span>${opt.label}</span>
          <div class="q-option-check">${currentVal === opt.value ? '✓' : ''}</div>
        `;
        pill.onclick = () => {
          questionnaireAnswers[q.id] = opt.value;
          renderQuestion(idx);
          // Auto advance on single choice after brief delay
          setTimeout(() => advanceQuestion(), 250);
        };
        qOptionsContainer.appendChild(pill);
      });
    } else if (q.question_type === 'select' && q.options) {
      q.options.forEach(opt => {
        const isSelected = currentVal === opt.value;
        const pill = document.createElement('div');
        pill.className = `q-option-pill ${isSelected ? 'selected' : ''}`;
        pill.innerHTML = `
          <span>${opt.label}</span>
          <div class="q-option-check">${isSelected ? '✓' : ''}</div>
        `;
        pill.onclick = () => {
          questionnaireAnswers[q.id] = opt.value;
          renderQuestion(idx);
          setTimeout(() => advanceQuestion(), 250);
        };
        qOptionsContainer.appendChild(pill);
      });
    } else if (q.question_type === 'multiselect' && q.options) {
      // Multiselect — stored as an array; "none" deselects everything else
      const selectedArr = Array.isArray(currentVal) ? currentVal : [];
      q.options.forEach(opt => {
        const isSelected = selectedArr.includes(opt.value);
        const pill = document.createElement('div');
        pill.className = `q-option-pill multiselect-pill ${isSelected ? 'selected' : ''}`;
        pill.innerHTML = `
          <span>${opt.label}</span>
          <div class="q-option-check">${isSelected ? '✓' : ''}</div>
        `;
        pill.onclick = () => {
          let arr = Array.isArray(questionnaireAnswers[q.id]) ? [...questionnaireAnswers[q.id]] : [];
          if (opt.value === 'none') {
            // "None" clears all others and toggles itself
            arr = arr.includes('none') ? [] : ['none'];
          } else {
            // Remove 'none' when selecting a real option
            arr = arr.filter(v => v !== 'none');
            if (arr.includes(opt.value)) {
              arr = arr.filter(v => v !== opt.value);
            } else {
              arr.push(opt.value);
            }
            if (arr.length === 0) arr = ['none'];
          }
          questionnaireAnswers[q.id] = arr;
          renderQuestion(idx);
          // No auto-advance for multiselect — user clicks Next when done
        };
        qOptionsContainer.appendChild(pill);
      });
      // Helper hint
      const hint = document.createElement('p');
      hint.className = 'q-multiselect-hint';
      hint.textContent = currentLanguage === 'hi' ? 'एक से अधिक विकल्प चुन सकते हैं' : 'Select all that apply, then tap Next';
      qOptionsContainer.appendChild(hint);
    } else if (q.question_type === 'scale') {
      const scaleWrap = document.createElement('div');
      scaleWrap.className = 'q-scale-container';
      for (let s = 1; s <= 10; s++) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `q-scale-btn ${currentVal == s ? 'selected' : ''}`;
        btn.textContent = s;
        btn.onclick = () => {
          questionnaireAnswers[q.id] = s;
          renderQuestion(idx);
          setTimeout(() => advanceQuestion(), 250);
        };
        scaleWrap.appendChild(btn);
      }
      qOptionsContainer.appendChild(scaleWrap);
    } else if (q.question_type === 'number') {
      const input = document.createElement('input');
      input.type = 'number';
      input.className = 'q-input-number';
      input.placeholder = currentLanguage === 'hi' ? 'दिनों की संख्या (जैसे 5)' : 'Number of days (e.g. 5)';
      input.value = currentVal || '';
      input.oninput = e => {
        questionnaireAnswers[q.id] = parseInt(e.target.value) || 0;
      };
      qOptionsContainer.appendChild(input);
    }
  }

  function advanceQuestion() {
    if (currentQuestionIdx < questionnaireList.length - 1) {
      currentQuestionIdx++;
      renderQuestion(currentQuestionIdx);
    } else {
      submitQuestionnaire();
    }
  }

  btnQNext?.addEventListener('click', advanceQuestion);

  btnQPrev?.addEventListener('click', () => {
    if (currentQuestionIdx > 0) {
      currentQuestionIdx--;
      renderQuestion(currentQuestionIdx);
    }
  });

  btnQSkip?.addEventListener('click', () => {
    advanceQuestion();
  });

  btnBackQ?.addEventListener('click', () => {
    if (confirm('Return to home? Current screening will be canceled.')) {
      showScreen('home');
    }
  });

  // ── Submit Questionnaire & Fetch Final Results ──
  async function submitQuestionnaire() {
    showScreen('analyzing');
    const stepClassify = document.getElementById('step-classify');
    if (stepClassify) {
      stepClassify.classList.add('active');
      stepClassify.querySelector('span').textContent = 'Fusing image features with clinical history...';
    }

    try {
      const resp = await fetch(`${API_BASE}${API_PREFIX}/screening/${currentScreeningId}/questionnaire`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          screening_id: currentScreeningId,
          responses: questionnaireAnswers,
        }),
      });

      if (!resp.ok) {
        throw new Error('Failed to submit questionnaire responses');
      }

      await fetchResults(currentScreeningId);
    } catch (err) {
      console.error('Submission error:', err);
      // Fallback: try fetching results directly
      await fetchResults(currentScreeningId);
    }
  }

  // ── Fetch & Display Complete Results ──
  async function fetchResults(screeningId) {
    try {
      const resp = await fetch(`${API_BASE}${API_PREFIX}/screening/${screeningId}/results`);
      if (!resp.ok) throw new Error('Failed to fetch full diagnostic assessment');

      const data = await resp.json();
      currentResultData = data;
      displayResults(data);
      showScreen('results');
    } catch (err) {
      console.error('Fetch results error:', err);
      alert('Could not retrieve results. Please retry.');
      showScreen('home');
    }
  }

  // Helper: Format anatomical sites cleanly
  function formatSite(site) {
    if (!site) return 'Lateral Tongue (Left)';
    const siteMap = {
      'lateral_tongue_left': 'Lateral Tongue (Left)',
      'lateral_tongue_right': 'Lateral Tongue (Right)',
      'dorsal_tongue': 'Dorsal Tongue',
      'ventral_tongue': 'Ventral Tongue',
      'buccal_mucosa_left': 'Buccal Mucosa (Left)',
      'buccal_mucosa_right': 'Buccal Mucosa (Right)',
      'hard_palate': 'Hard Palate',
      'soft_palate': 'Soft Palate',
      'floor_of_mouth': 'Floor of Mouth',
      'labial_mucosa': 'Labial Mucosa',
      'gingiva_upper': 'Upper Gingiva',
      'gingiva_lower': 'Lower Gingiva',
      'retromolar_trigone': 'Retromolar Trigone'
    };
    if (siteMap[site.toLowerCase()]) return siteMap[site.toLowerCase()];
    return site.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  function displayResults(data) {
    const t = i18n[currentLanguage];

    // Images
    const origUrl = data.images?.original ? `${API_BASE}${data.images.original}` : currentImageDataUrl;
    const heatmapUrl = data.images?.gradcam ? `${API_BASE}${data.images.gradcam}` : currentImageDataUrl;
    const segUrl = data.images?.segmentation ? `${API_BASE}${data.images.segmentation}` : null;

    if (resultScannedImg) resultScannedImg.src = origUrl;
    if (resultHeatmapImg) resultHeatmapImg.src = heatmapUrl;
    if (resultSegImg && segUrl) resultSegImg.src = segUrl;

    // Risk Stratification
    const risk = (data.risk_level || 'low').toLowerCase();
    const score = data.risk_score !== undefined ? data.risk_score : 13;

    if (riskBadge) riskBadge.className = `risk-badge risk--${risk}`;
    if (riskScoreEl) riskScoreEl.textContent = `${score}`;

    const riskProgressFill = document.getElementById('risk-progress-fill');
    if (riskProgressFill) {
      riskProgressFill.style.width = `${Math.min(100, Math.max(8, score))}%`;
    }

    const tierLabel = document.getElementById('risk-tier-label');
    if (tierLabel) {
      tierLabel.textContent = risk === 'low' ? 'LOW RISK' :
                              risk === 'medium' ? 'MODERATE RISK' :
                              risk === 'high' ? 'HIGH RISK' : 'URGENT ATTENTION';
    }

    const badgeDesc = document.getElementById('risk-badge-desc');
    if (badgeDesc) {
      badgeDesc.textContent = risk === 'low' ? 'Benign Presentation • Favorable Prognosis' :
                              risk === 'medium' ? 'Mucosal Lesion Present • Monitoring Indicated' :
                              risk === 'high' ? 'Suspicious Epithelial Changes • Specialist Assessment' :
                              'High-Grade Atypia Suspected • Urgent Biopsy Required';
    }

    const tierPill = document.getElementById('risk-tier-pill');
    if (tierPill) {
      tierPill.textContent = risk === 'low' ? 'Grade 1' :
                             risk === 'medium' ? 'Grade 2' :
                             risk === 'high' ? 'Grade 3' : 'Grade 4';
    }

    // Primary Diagnosis
    const rawDiag = (data.primary_diagnosis || 'aphthous_ulcer').toLowerCase();
    let displayDiag = 'Recurrent Aphthous Ulcer';
    let icdCode = 'ICD-10: K12.0';

    if (rawDiag === 'other') {
      displayDiag = 'Other / Non-Specific Oral Lesion';
      icdCode = 'ICD-10: K13.7';
    } else if (rawDiag.includes('aphthous')) {
      displayDiag = 'Recurrent Aphthous Ulcer';
      icdCode = 'ICD-10: K12.0';
    } else if (rawDiag.includes('carcinoma') || rawDiag.includes('oscc')) {
      displayDiag = 'Oral Squamous Cell Carcinoma (Suspected)';
      icdCode = 'ICD-10: C06.9';
    } else if (rawDiag.includes('traumatic')) {
      displayDiag = 'Traumatic Mucosal Ulcer';
      icdCode = 'ICD-10: K12.04';
    } else if (rawDiag.includes('lichen')) {
      displayDiag = 'Oral Lichen Planus';
      icdCode = 'ICD-10: L43.9';
    } else if (rawDiag.includes('leukoplakia')) {
      displayDiag = 'Oral Leukoplakia';
      icdCode = 'ICD-10: K13.21';
    } else {
      displayDiag = rawDiag.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    if (diagnosisName) diagnosisName.textContent = displayDiag;

    const diagIcdCode = document.getElementById('diag-icd-code');
    if (diagIcdCode) diagIcdCode.textContent = icdCode;

    const subtype = data.aphthous_subtype || (data.classification_probabilities?.oscc > 0.4 ? 'Endophytic Type' : 'Minor Type');
    if (diagnosisSubtype) diagnosisSubtype.textContent = subtype;

    const confVal = Math.round((data.confidence || 0.55) * 100);
    const confidenceBadgeTxt = document.getElementById('confidence-badge-txt');
    if (confidenceBadgeTxt) confidenceBadgeTxt.textContent = `Confidence ${confVal}%`;
    if (confidencePct) confidencePct.textContent = `${confVal}%`;
    if (confidenceFill) confidenceFill.style.width = `${confVal}%`;

    // Medical Overview
    const overviewEl = document.getElementById('overview-text');
    if (overviewEl) {
      if (rawDiag.includes('carcinoma') || rawDiag.includes('oscc')) {
        overviewEl.textContent = 'Oral squamous cell carcinoma is the most frequent malignant neoplasm of the oral cavity. Clinical warning signs include chronic non-healing indurated ulcers with rolled margins, leukoplakia or erythroplakia. Urgent specialist histopathological biopsy is required for definitive diagnostic confirmation.';
      } else if (rawDiag.includes('traumatic')) {
        overviewEl.textContent = 'Traumatic ulcers are mechanical oral mucosal lesions typically caused by accidental cheek/tongue biting, jagged teeth, or foreign body abrasion. They present with acute localized tenderness and generally heal spontaneously within 7-10 days once the source of irritation is eliminated.';
      } else {
        overviewEl.textContent = 'Aphthous ulcers (canker sores) are the most common type of oral ulceration. They are typically round or oval, with a yellowish-gray pseudomembranous center and an erythematous halo. Minor aphthous ulcers are less than 1 cm in diameter and heal spontaneously within 7-14 days without scarring.';
      }
    }

    // Detected Visual Biomarkers
    if (featuresGrid && data.detected_features) {
      featuresGrid.innerHTML = '';
      const feats = data.detected_features;

      const siteFormatted = formatSite(feats.anatomical_location || 'lateral_tongue_left');
      const borderFormatted = feats.border_type === 'regular' ? 'Regular' : 'Irregular';
      const rednessFormatted = (feats.red_component || 'Severe').replace(/\b\w/g, l => l.toUpperCase());
      const whiteFormatted = (feats.white_component || 'Severe').replace(/\b\w/g, l => l.toUpperCase());
      const indurationFormatted = feats.induration ? 'Present' : 'No Induration';
      const ulcerFormatted = feats.ulceration ? 'Present' : 'Absent';

      const featItems = [
        { title: 'Ulceration', val: ulcerFormatted, type: feats.ulceration ? 'info' : 'good' },
        { title: 'Border', val: borderFormatted, type: feats.border_type === 'regular' ? 'good' : 'warn' },
        { title: 'Redness', val: rednessFormatted, type: feats.red_component === 'severe' ? 'warn' : 'info' },
        { title: 'White', val: whiteFormatted, type: 'info' },
        { title: 'Induration', val: indurationFormatted, type: feats.induration ? 'danger' : 'good' },
        { title: 'Site', val: siteFormatted, type: 'info' },
      ];

      featItems.forEach(f => {
        const item = document.createElement('div');
        item.className = `feature-tag feature-tag--${f.type}`;
        item.innerHTML = `
          <span class="feature-tag-dot"></span>
          <span class="feature-tag-title">${f.title}:</span>
          <span class="feature-tag-val">${f.val}</span>
        `;
        featuresGrid.appendChild(item);
      });
    }

    // Differential Diagnoses
    if (diffList && data.differential_diagnoses) {
      diffList.innerHTML = '';
      const icdMap = {
        'recurrent aphthous stomatitis': 'K12.0',
        'aphthous ulcer': 'K12.0',
        'traumatic ulcer': 'K12.04',
        'erythema multiforme': 'L51',
        'pemphigus vulgaris': 'L10.0',
        'behcet disease': 'M35.2',
        'oral squamous cell carcinoma': 'C06.9',
        'oscc': 'C06.9',
        'oral lichen planus': 'L43.9',
        'leukoplakia': 'K13.21',
        'erythroplakia': 'K13.29',
      };

      data.differential_diagnoses.slice(0, 5).forEach((diff, i) => {
        const prob = Math.round((diff.probability || 0.05) * 100);
        const name = diff.display_name || diff.condition || 'Oral Lesion';
        const key = name.toLowerCase();
        const icd = icdMap[key] || 'K13.7';
        const isSuspicious = key.includes('carcinoma') || key.includes('oscc') || key.includes('erythroplakia');

        const item = document.createElement('div');
        item.className = 'diff-item';
        item.innerHTML = `
          <div class="diff-rank">${i + 1}</div>
          <div class="diff-info">
            <div class="diff-name-row">
              <span class="diff-name">${name}</span>
              <span class="diff-icd">${icd}</span>
            </div>
            <div class="diff-bar-wrap">
              <div class="diff-bar ${isSuspicious ? 'diff-bar--danger' : ''}" style="width: ${prob}%"></div>
            </div>
          </div>
          <span class="diff-pct">${prob}%</span>
        `;
        diffList.appendChild(item);
      });
    }

    // Recommendations
    if (recommendationList && data.recommendations) {
      recommendationList.innerHTML = '';
      const recUrgencyBadge = document.getElementById('rec-urgency-badge');
      if (recUrgencyBadge) {
        recUrgencyBadge.textContent = risk === 'low' ? 'Routine Follow-up' :
                                     risk === 'medium' ? 'Semi-Urgent (1-2 Weeks)' :
                                     risk === 'high' ? 'Specialist Referral' : 'Immediate Evaluation';
      }

      data.recommendations.forEach(r => {
        const li = document.createElement('li');
        let iconSvg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>';
        if (r.toLowerCase().includes('2 weeks') || r.toLowerCase().includes('reassess')) {
          iconSvg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>';
        } else if (r.toLowerCase().includes('avoid') || r.toLowerCase().includes('spicy')) {
          iconSvg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>';
        } else if (r.toLowerCase().includes('symptomatic') || r.toLowerCase().includes('analgesic') || r.toLowerCase().includes('mouthwash')) {
          iconSvg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';
        }

        li.innerHTML = `
          <div class="rec-icon">${iconSvg}</div>
          <span>${r}</span>
        `;
        recommendationList.appendChild(li);
      });
    }
  }

  // ── Image Tabs Switching ──
  document.querySelectorAll('.img-tab-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.img-tab-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');

      const view = this.getAttribute('data-view');
      if (view === 'split') {
        cardScannedImg?.classList.remove('hidden');
        cardHeatmapImg?.classList.remove('hidden');
        cardSegImg?.classList.add('hidden');
      } else if (view === 'orig') {
        cardScannedImg?.classList.remove('hidden');
        cardHeatmapImg?.classList.add('hidden');
        cardSegImg?.classList.add('hidden');
      } else if (view === 'heatmap') {
        cardScannedImg?.classList.add('hidden');
        cardHeatmapImg?.classList.remove('hidden');
        cardSegImg?.classList.add('hidden');
      } else if (view === 'seg') {
        cardScannedImg?.classList.add('hidden');
        cardHeatmapImg?.classList.add('hidden');
        cardSegImg?.classList.remove('hidden');
      }
    });
  });

  // ── Download PDF Report ──
  btnDownloadPdf?.addEventListener('click', () => {
    if (!currentScreeningId) {
      alert('No active screening record to download.');
      return;
    }
    const reportUrl = `${API_BASE}${API_PREFIX}/screening/${currentScreeningId}/report`;
    window.open(reportUrl, '_blank');
  });

  // ── Share With Doctor (WhatsApp) ──
  btnShareDoctor?.addEventListener('click', () => {
    if (!currentResultData) return;
    const diag = currentResultData.primary_diagnosis?.replace('_', ' ') || 'Aphthous Ulcer';
    const risk = (currentResultData.risk_level || 'low').toUpperCase();
    const score = currentResultData.risk_score || 15;
    const date = new Date().toLocaleDateString();

    const text = encodeURIComponent(
      `🦷 *OralGuard AI — Screening Report*\n` +
      `Date: ${date}\n` +
      `Screening ID: ${currentScreeningId}\n` +
      `Preliminary Assessment: ${diag}\n` +
      `Risk Level: ${risk} (${score}/100)\n` +
      `Disclaimer: AI clinical screening assistance. Biopsy required for definitive malignancy diagnosis.`
    );
    window.open(`https://api.whatsapp.com/send?text=${text}`, '_blank');
  });

  // ── Nearby Clinics ──
  btnNearbyClinic?.addEventListener('click', () => {
    window.open('https://www.google.com/maps/search/dental+clinic+near+me', '_blank');
  });

  // ── New Screening ──
  btnRescan?.addEventListener('click', () => {
    currentScreeningId = null;
    currentImageBlob = null;
    currentImageDataUrl = null;
    questionnaireAnswers = {};
    showScreen('home');
  });

  // ── History Screen ──
  async function loadHistory() {
    showScreen('history');
    if (!historyList) return;

    historyList.innerHTML = '<div style="text-align:center; padding: 24px; color: var(--gray-400);">Loading history...</div>';

    try {
      const resp = await fetch(`${API_BASE}${API_PREFIX}/screening/history/list`);
      if (!resp.ok) throw new Error('Failed to load history');

      const data = await resp.json();
      const items = data.screenings || [];

      if (items.length === 0) {
        historyList.innerHTML = `
          <div class="history-empty">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            <p>${i18n[currentLanguage].noHistory}</p>
          </div>
        `;
        return;
      }

      historyList.innerHTML = '';
      items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'history-item';
        card.setAttribute('data-id', item.id);
        const dateStr = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent';
        const risk = (item.risk_level || 'low').toLowerCase();

        card.innerHTML = `
          <img class="history-thumb" src="${item.image_url ? API_BASE + item.image_url : 'https://placehold.co/100x100/F5F3FF/7C3AED?text=Oral'}" alt="Lesion"/>
          <div class="history-details">
            <div class="history-diag">${item.primary_diagnosis_display || item.primary_diagnosis}</div>
            <div class="history-date">${dateStr} • Confidence: ${Math.round((item.confidence || 0.85)*100)}%</div>
          </div>
          <div class="history-item-actions">
            <span class="history-badge risk--${risk}">${risk.toUpperCase()} (${item.risk_score || 15})</span>
            <button class="history-delete-btn" type="button" title="Delete record" aria-label="Delete screening record">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 6h18"/>
                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
                <line x1="10" y1="11" x2="10" y2="17"/>
                <line x1="14" y1="11" x2="14" y2="17"/>
              </svg>
            </button>
          </div>
        `;

        card.onclick = () => {
          currentScreeningId = item.id;
          fetchResults(item.id);
        };

        const deleteBtn = card.querySelector('.history-delete-btn');
        if (deleteBtn) {
          deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent opening report
            const confirmed = confirm('Are you sure you want to delete this screening record? This cannot be undone.');
            if (!confirmed) return;

            deleteBtn.disabled = true;
            deleteBtn.innerHTML = '<span class="history-spinner"></span>';

            try {
              let res = await fetch(`${API_BASE}${API_PREFIX}/screening/${item.id}`, {
                method: 'DELETE',
              });
              if (!res.ok) {
                res = await fetch(`${API_BASE}${API_PREFIX}/screening/history/${item.id}`, {
                  method: 'DELETE',
                });
              }
              if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || 'Failed to delete record');
              }

              // Animate out card smoothly
              card.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
              card.style.opacity = '0';
              card.style.transform = 'translateX(25px) scale(0.96)';

              setTimeout(() => {
                card.remove();
                if (!historyList.querySelector('.history-item')) {
                  historyList.innerHTML = `
                    <div class="history-empty">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                      </svg>
                      <p>${i18n[currentLanguage].noHistory}</p>
                    </div>
                  `;
                }
              }, 300);
            } catch (err) {
              deleteBtn.disabled = false;
              deleteBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M3 6h18"/>
                  <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
                  <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
                  <line x1="10" y1="11" x2="10" y2="17"/>
                  <line x1="14" y1="11" x2="14" y2="17"/>
                </svg>
              `;
              alert(`Could not delete record: ${err.message}`);
            }
          });
        }

        historyList.appendChild(card);
      });
    } catch (err) {
      historyList.innerHTML = `<div style="color: #EF4444; padding: 20px;">Could not load history: ${err.message}</div>`;
    }
  }

  btnHistory?.addEventListener('click', loadHistory);
  navHistory?.addEventListener('click', loadHistory);
  btnBackHistory?.addEventListener('click', () => showScreen('home'));
  btnRefreshHistory?.addEventListener('click', loadHistory);

  btnClearHistory?.addEventListener('click', async () => {
    const hasItems = historyList?.querySelector('.history-item');
    if (!hasItems) {
      alert('No screening records found to clear.');
      return;
    }

    if (!confirm('Are you sure you want to delete ALL screening history records? This cannot be undone.')) {
      return;
    }

    try {
      const resp = await fetch(`${API_BASE}${API_PREFIX}/screening/history/clear/all`, {
        method: 'DELETE',
      });
      if (!resp.ok) throw new Error('Failed to clear history');

      historyList.innerHTML = `
        <div class="history-empty">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          <p>${i18n[currentLanguage].noHistory}</p>
        </div>
      `;
    } catch (err) {
      alert(`Could not clear history: ${err.message}`);
    }
  });

  // ── Dashboard Screen ──
  async function loadDashboard() {
    showScreen('dashboard');

    try {
      const resp = await fetch(`${API_BASE}${API_PREFIX}/dashboard/stats`);
      if (!resp.ok) throw new Error('Failed to load dashboard metrics');

      const stats = await resp.json();

      document.getElementById('kpi-total').textContent = stats.total_screenings || 0;
      document.getElementById('kpi-today').textContent = stats.screenings_today || 0;
      document.getElementById('kpi-pos-rate').textContent = `${(stats.positive_rate || 0).toFixed(1)}%`;
      document.getElementById('kpi-ref-rate').textContent = `${(stats.referral_rate || 0).toFixed(1)}%`;

      // Risk bars
      const rd = stats.risk_distribution || {};
      const total = stats.total_screenings || 1;

      const setBar = (id, count, fillId) => {
        const el = document.getElementById(id);
        const fill = document.getElementById(fillId);
        if (el) el.textContent = count || 0;
        if (fill) fill.style.width = `${Math.min(100, Math.round(((count || 0) / total) * 100))}%`;
      };

      setBar('r-count-low', rd.low, 'r-fill-low');
      setBar('r-count-med', rd.medium, 'r-fill-med');
      setBar('r-count-high', rd.high, 'r-fill-high');
      setBar('r-count-urg', rd.urgent, 'r-fill-urg');

      // Top conditions
      const listEl = document.getElementById('dash-conditions-list');
      if (listEl && stats.top_conditions) {
        listEl.innerHTML = '';
        stats.top_conditions.forEach(c => {
          const row = document.createElement('div');
          row.className = 'dash-cond-row';
          row.innerHTML = `
            <span class="dash-cond-name">${(c.condition || '').replace('_', ' ').toUpperCase()}</span>
            <span class="dash-cond-stats">${c.count} cases (${c.percentage}%)</span>
          `;
          listEl.appendChild(row);
        });
      }
    } catch (err) {
      console.warn('Dashboard stats error:', err);
    }
  }

  navDashboard?.addEventListener('click', loadDashboard);
  btnBackDashboard?.addEventListener('click', () => showScreen('home'));
  btnExportCsv?.addEventListener('click', () => {
    window.open(`${API_BASE}${API_PREFIX}/dashboard/export`, '_blank');
  });

  // ── Global Buttons ──
  btnStartScan?.addEventListener('click', () => {
    showScreen('scan');
    startCamera();
  });
  
  navScan?.addEventListener('click', () => {
    showScreen('scan');
    startCamera();
  });

  btnUpload?.addEventListener('click', () => {
    fileInput.removeAttribute('capture');
    fileInput.click();
  });

  btnGallery?.addEventListener('click', () => {
    fileInput.removeAttribute('capture');
    fileInput.click();
  });

  btnCapture?.addEventListener('click', captureFrame);

  btnBackScan?.addEventListener('click', () => {
    stopCamera();
    showScreen('home');
  });

  btnBackResults?.addEventListener('click', () => {
    showScreen('home');
  });

  document.querySelectorAll('.nav-item[data-screen="home"]').forEach(btn => {
    btn.addEventListener('click', () => {
      stopCamera();
      showScreen('home');
    });
  });

})();
