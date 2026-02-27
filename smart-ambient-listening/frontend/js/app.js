/**
 * SMART Ambient Listening Application
 *
 * Flow:
 * 1. User clicks "Start Recording" → Deploy Modal app + Warmup (model loads)
 * 2. User speaks...
 * 3. User clicks "Stop Recording" → Transcribe (fast, container already warm)
 */
document.addEventListener('DOMContentLoaded', async () => {
    // State
    let smartClient = null;
    let openemrApi = null;
    let patient = null;
    let recorder = null;
    let transcriptionHistory = [];
    let modalDeployed = false;
    let warmupInterval = null;
    let deployPromise = null;
    let userSettings = null;
    let currentSoapNote = null;       // Store the raw SOAP note text for saving
    let currentSoapResult = null;     // Store the full result object

    // Configuration - CRITICAL: Must point to transcription service
    const config = window.SMART_CONFIG || {
        // If running on same server with Nginx reverse proxy:
        transcriptionServiceUrl: window.location.origin + '/api/transcribe',
        summarizationServiceUrl: window.location.origin + '/api/summarize'
    };

    console.log('🔧 Service URLs:', config);

    /**
     * Get the SMART access token for backend authentication
     * @returns {Promise<string>} The access token
     */
    async function getAccessToken() {
        if (!smartClient) {
            throw new Error('SMART client not initialized');
        }

        const state = smartClient.state;
        if (state && state.tokenResponse && state.tokenResponse.access_token) {
            return state.tokenResponse.access_token;
        }

        throw new Error('No access token available');
    }

    /**
     * Send a warmup ping to keep the container alive.
     */
    async function sendWarmupPing() {
        console.log('Sending warmup ping to keep container alive...');
        try {
            const token = await getAccessToken();

            const response = await fetch(`${config.transcriptionServiceUrl}/warmup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: '{}'
            });

            if (!response.ok) {
                console.warn('Warmup ping failed, container may scale down');
            } else {
                console.log('Warmup ping successful');
            }
        } catch (error) {
            console.error('Failed to send warmup ping:', error);
        }
    }

    /**
     * Parse error from response properly
     */
    async function parseError(response) {
        let errorMessage = `Server error: ${response.status} ${response.statusText}`;

        try {
            const contentType = response.headers.get('content-type');

            if (contentType && contentType.includes('application/json')) {
                const errorData = await response.json();

                // Handle FastAPI error format
                if (errorData.detail) {
                    if (typeof errorData.detail === 'string') {
                        errorMessage = errorData.detail;
                    } else if (Array.isArray(errorData.detail)) {
                        errorMessage = errorData.detail
                            .map(e => `${e.loc ? e.loc.join('.') + ': ' : ''}${e.msg || JSON.stringify(e)}`)
                            .join('; ');
                    } else if (typeof errorData.detail === 'object') {
                        errorMessage = JSON.stringify(errorData.detail);
                    }
                } else if (errorData.message) {
                    errorMessage = errorData.message;
                } else {
                    errorMessage = JSON.stringify(errorData);
                }
            } else {
                const errorText = await response.text();
                if (errorText && errorText.trim()) {
                    errorMessage = errorText;
                }
            }
        } catch (e) {
            console.error('Error parsing error response:', e);
        }

        return errorMessage;
    }

    /**
     * Deploy Modal app and warm up the container.
     * BYOK version - credentials retrieved server-side from OpenEMR
     */
    async function deployAndWarmup() {
        console.log('🚀 Deploying Modal app and warming up container...');
        console.log('   URL:', `${config.transcriptionServiceUrl}/deploy-and-warmup`);

        try {
            const token = await getAccessToken();
            console.log('   Token acquired:', token ? '✅' : '❌');

            const response = await fetch(`${config.transcriptionServiceUrl}/deploy-and-warmup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: '{}'
            });

            console.log('   Response status:', response.status);

            if (!response.ok) {
                const errorMessage = await parseError(response);
                console.error('   Deploy failed:', errorMessage);
                throw new Error(errorMessage);
            }

            const result = await response.json();
            console.log('✅ Modal app deployed and warmed up:', result);
            modalDeployed = true;
            return true;
        } catch (error) {
            console.error('❌ Failed to deploy/warmup:', error);
            showError(error.message || 'Failed to deploy transcription service. Check console for details.');
            return false;
        }
    }

    // DOM Elements
    const elements = {
        patientBanner: document.getElementById('patient-banner'),
        patientName: document.getElementById('patient-name'),
        patientDob: document.getElementById('patient-dob'),
        patientMrn: document.getElementById('patient-mrn'),
        connectionStatus: document.getElementById('connection-status'),
        statusText: document.getElementById('status-text'),
        btnAmbient: document.getElementById('btn-ambient'),
        recordingStatus: document.getElementById('recording-status'),
        transcriptionLoading: document.getElementById('transcription-loading'),
        transcriptionResults: document.getElementById('transcription-results'),
        transcriptionHistory: document.getElementById('transcription-history'),
        historyList: document.getElementById('history-list'),
        actionsSection: document.getElementById('actions-section'),
        btnCopy: document.getElementById('btn-copy'),
        btnClear: document.getElementById('btn-clear'),
        btnSummarize: document.getElementById('btn-summarize'),
        soapSection: document.getElementById('soap-section'),
        soapLoading: document.getElementById('soap-loading'),
        soapResults: document.getElementById('soap-results'),
        errorModal: document.getElementById('error-modal'),
        errorMessage: document.getElementById('error-message'),
        btnCloseError: document.getElementById('btn-close-error'),
        permissionModal: document.getElementById('permission-modal'),
        btnRequestPermission: document.getElementById('btn-request-permission'),
        visualizer: document.getElementById('visualizer'),
        // Encounter selection modal elements
        encounterModal: document.getElementById('encounter-modal'),
        encounterList: document.getElementById('encounter-list'),
        encounterLoading: document.getElementById('encounter-loading'),
        btnCloseEncounter: document.getElementById('btn-close-encounter')
    };

    /**
     * Initialize the application
     */
    async function init() {
        if (!AmbientRecorder.isSupported()) {
            showError('Your browser does not support audio recording.');
            return;
        }

        try {
            smartClient = await FHIR.oauth2.ready();
            await connectToEHR();
        } catch (error) {
            console.log('No SMART context available:', error.message);
            updateConnectionStatus(false, 'Not connected to EHR. Launch from OpenEMR patient dashboard.');
            elements.btnAmbient.disabled = false;
        }

        initRecorder();
        setupEventListeners();
    }

    /**
     * Connect to EHR and load patient data
     */
    async function connectToEHR() {
        try {
            updateConnectionStatus(true, 'Connected to OpenEMR');
            patient = await smartClient.patient.read();
            displayPatientBanner(patient);
            elements.btnAmbient.disabled = false;

            openemrApi = new OpenEMRApi(smartClient);
            await loadCustomUserSettings();
        } catch (error) {
            console.error('EHR connection error:', error);
            updateConnectionStatus(false, 'Failed to load patient data');
        }
    }

    /**
     * Load custom user settings from OpenEMR
     */
    async function loadCustomUserSettings() {
        if (!openemrApi) {
            console.log('OpenEMR API not initialized, skipping custom settings');
            return;
        }

        try {
            const result = await openemrApi.getCustomUserSettings();
            userSettings = {};

            if (result.data && Array.isArray(result.data)) {
                result.data.forEach(setting => {
                    userSettings[setting.field_id] = setting.field_value;
                });
            }

            console.log('Loaded custom user settings:', userSettings);

            const hasGroq = userSettings['1'] || userSettings['GROQ API Token'];
            const hasModal = userSettings['2'] || userSettings['MODAL API Token'];
            const hasModalSecret = userSettings['3'] || userSettings['MODAL API secret'];

            if (!hasGroq || !hasModal || !hasModalSecret) {
                console.warn('⚠️ API keys not fully configured!');
                console.warn('   Groq:', hasGroq ? '✅' : '❌');
                console.warn('   Modal Token:', hasModal ? '✅' : '❌');
                console.warn('   Modal Secret:', hasModalSecret ? '✅' : '❌');
            }

            applyUserSettings();
        } catch (error) {
            console.warn('Could not load custom user settings:', error.message);
            console.log('The api:oemr scope may need to be granted, or no USR layout fields are configured.');
        }
    }

    /**
     * Apply user settings to the application
     */
    function applyUserSettings() {
        if (!userSettings) {
            console.log('No user settings to apply');
            return;
        }
        console.log('User settings loaded and ready to apply:', userSettings);
    }

    /**
     * Save a custom user setting
     */
    async function saveUserSetting(fieldId, value) {
        if (!openemrApi) {
            console.warn('OpenEMR API not initialized, cannot save setting');
            return false;
        }

        try {
            await openemrApi.updateCustomUserSetting(fieldId, value);
            userSettings[fieldId] = value;
            console.log(`Saved user setting: ${fieldId} = ${value}`);
            return true;
        } catch (error) {
            console.error(`Failed to save user setting ${fieldId}:`, error);
            return false;
        }
    }

    /**
     * Get a custom user setting value
     */
    function getUserSetting(fieldId, defaultValue = null) {
        return userSettings?.[fieldId] ?? defaultValue;
    }

    /**
     * Display patient information banner
     */
    function displayPatientBanner(patient) {
        const name = getPatientName(patient);
        const dob = patient.birthDate || '';
        const mrn = getPatientMRN(patient);

        elements.patientName.textContent = name;
        elements.patientDob.textContent = dob ? `DOB: ${dob}` : '';
        elements.patientMrn.textContent = mrn ? `MRN: ${mrn}` : '';
        elements.patientBanner.classList.remove('hidden');
    }

    /**
     * Get patient display name from FHIR Patient resource
     */
    function getPatientName(patient) {
        const name = patient.name?.[0];
        if (!name) return 'Unknown';
        const given = name.given?.join(' ') || '';
        const family = name.family || '';
        return `${given} ${family}`.trim();
    }

    /**
     * Get patient MRN from FHIR Patient resource
     */
    function getPatientMRN(patient) {
        if (!patient.identifier) return null;
        const mrnId = patient.identifier.find(
            id => id.type?.coding?.some(c => c.code === 'MR')
        );
        return mrnId?.value || patient.identifier[0]?.value || null;
    }

    /**
     * Update connection status indicator
     */
    function updateConnectionStatus(connected, message) {
        const indicator = elements.connectionStatus.querySelector('.status-indicator');
        indicator.classList.toggle('connected', connected);
        indicator.classList.toggle('disconnected', !connected);
        elements.statusText.textContent = message;
    }

    /**
     * Initialize audio recorder
     */
    function initRecorder() {
        recorder = new AmbientRecorder({
            visualizer: elements.visualizer,
            onStart: handleRecordingStart,
            onStop: handleRecordingStop,
            onError: handleRecordingError
        });
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        elements.btnAmbient.addEventListener('click', toggleRecording);
        elements.btnCopy.addEventListener('click', copyTranscription);
        elements.btnClear.addEventListener('click', clearSession);
        elements.btnSummarize.addEventListener('click', generateSOAPNote);
        elements.btnCloseError.addEventListener('click', () => {
            elements.errorModal.classList.add('hidden');
        });

        elements.btnRequestPermission.addEventListener('click', async () => {
            elements.permissionModal.classList.add('hidden');
            const result = await recorder.requestPermission();
            if (result.granted) {
                toggleRecording();
            } else {
                showError('Microphone permission denied.');
            }
        });

        // Encounter modal close button
        if (elements.btnCloseEncounter) {
            elements.btnCloseEncounter.addEventListener('click', () => {
                elements.encounterModal.classList.add('hidden');
            });
        }
    }

    /**
     * Toggle recording state
     */
    async function toggleRecording() {
        if (recorder.isRecording) {
            recorder.stop();
        } else {
            try {
                const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
                if (permissionStatus.state === 'denied') {
                    showError('Microphone access is blocked. Enable it in browser settings.');
                    return;
                }
                if (permissionStatus.state === 'prompt') {
                    elements.permissionModal.classList.remove('hidden');
                    return;
                }
            } catch (e) {
                // permissions API not supported
            }
            await recorder.start();
        }
    }

    /**
     * Handle recording start
     */
    function handleRecordingStart() {
        console.log('handleRecordingStart called');

        elements.btnAmbient.classList.add('recording');
        elements.btnAmbient.querySelector('.btn-text').textContent = 'Stop Listening';
        elements.recordingStatus.classList.remove('hidden');
        elements.transcriptionResults.innerHTML = `
            <p class="placeholder-text">
                🚀 Setting up transcription service (first time may take 1-2 minutes)...
            </p>
        `;

        deployPromise = deployAndWarmup();

        deployPromise.then((transSuccess) => {
            if (transSuccess) {
                elements.transcriptionResults.innerHTML = `
                    <p class="placeholder-text">
                        ✅ Transcription ready. Recording...
                    </p>
                `;
            } else {
                elements.transcriptionResults.innerHTML = `
                    <p class="placeholder-text">
                        ⚠️ Transcription setup incomplete. Please try again.
                    </p>
                `;
            }

            warmupInterval = setInterval(sendWarmupPing, 60000);
            console.log('Warmup interval started');
        }).catch(error => {
            console.error('Deployment error:', error);
            elements.transcriptionResults.innerHTML = `
                <p class="placeholder-text">
                    ⚠️ Service setup in progress... Recording will be transcribed when ready.
                </p>
            `;
            warmupInterval = setInterval(sendWarmupPing, 60000);
        });
    }

    /**
     * Handle recording stop
     */
    async function handleRecordingStop(blob) {
        console.log('handleRecordingStop called');

        if (warmupInterval) {
            console.log('Clearing warmup interval...');
            clearInterval(warmupInterval);
            warmupInterval = null;
            console.log('Warmup interval cleared');
        }

        elements.btnAmbient.classList.remove('recording');
        elements.btnAmbient.querySelector('.btn-text').textContent = 'Start Ambient Listening';
        elements.recordingStatus.classList.add('hidden');
        elements.transcriptionLoading.classList.remove('hidden');

        try {
            if (deployPromise) {
                elements.transcriptionResults.innerHTML = `
                    <p class="placeholder-text">
                        ⏳ Finishing transcription service setup... Please wait.
                    </p>
                `;
                const deploySuccess = await deployPromise;
                if (!deploySuccess) {
                    throw new Error('Transcription service failed to initialize. Please try again.');
                }
                elements.transcriptionResults.innerHTML = `
                    <p class="placeholder-text">
                        📝 Transcribing your recording...
                    </p>
                `;
            }

            const result = await transcribeAudio(blob);
            displayTranscription(result);
            addToHistory(result);
            elements.actionsSection.classList.remove('hidden');
        } catch (error) {
            console.error('Transcription error:', error);
            showError(`Transcription failed: ${error.message}`);
        } finally {
            elements.transcriptionLoading.classList.add('hidden');
            deployPromise = null;
        }
    }

    /**
     * Handle recording error
     */
    function handleRecordingError(error) {
        console.log('handleRecordingError called');

        if (warmupInterval) {
            console.log('Clearing warmup interval due to error...');
            clearInterval(warmupInterval);
            warmupInterval = null;
        }

        showError(`Recording error: ${error.message}`);
        elements.btnAmbient.classList.remove('recording');
        elements.btnAmbient.querySelector('.btn-text').textContent = 'Start Ambient Listening';
        elements.recordingStatus.classList.add('hidden');
    }

    /**
     * Send audio to transcription service
     */
    async function transcribeAudio(blob) {
        console.log('🎤 Transcribing audio...');

        try {
            const token = await getAccessToken();

            const formData = new FormData();
            formData.append('audio', blob, 'recording.webm');

            if (patient) {
                formData.append('patient_id', patient.id);
            }

            console.log('   URL:', `${config.transcriptionServiceUrl}/transcribe`);
            console.log('   Token:', token ? '✅' : '❌');

            const response = await fetch(`${config.transcriptionServiceUrl}/transcribe`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });

            console.log('   Response status:', response.status);

            if (!response.ok) {
                const errorMessage = await parseError(response);
                throw new Error(errorMessage);
            }

            return response.json();
        } catch (error) {
            console.error('❌ Transcription failed:', error);
            throw error;
        }
    }

    /**
     * Display transcription result
     */
    function displayTranscription(result) {
        const text = result.text || 'No transcription available';
        const duration = result.duration ? `${result.duration.toFixed(1)}s` : '';

        elements.transcriptionResults.innerHTML = `
            <div class="transcription-text">${escapeHtml(text)}</div>
            ${duration ? `<div class="transcription-meta"><span>Duration: ${duration}</span></div>` : ''}
        `;
    }

    /**
     * Add transcription to history
     */
    function addToHistory(result) {
        transcriptionHistory.push({
            text: result.text,
            timestamp: new Date().toISOString(),
            duration: result.duration
        });
        updateHistoryDisplay();
    }

    /**
     * Update history display
     */
    function updateHistoryDisplay() {
        if (transcriptionHistory.length === 0) {
            elements.transcriptionHistory.classList.add('hidden');
            return;
        }

        elements.transcriptionHistory.classList.remove('hidden');
        elements.historyList.innerHTML = transcriptionHistory.map(item => `
            <div class="history-item">
                <div class="history-time">${formatTime(item.timestamp)}</div>
                <div class="history-text">${escapeHtml(item.text)}</div>
            </div>
        `).join('');
    }

    /**
     * Copy transcription to clipboard
     */
    async function copyTranscription() {
        const text = transcriptionHistory.map(h => h.text).join('\n\n');
        try {
            await navigator.clipboard.writeText(text);
            elements.btnCopy.textContent = 'Copied!';
            setTimeout(() => elements.btnCopy.textContent = 'Copy Transcription', 2000);
        } catch (error) {
            showError('Failed to copy to clipboard');
        }
    }

    /**
     * Get patient EHR data for summarization context
     */
    async function getPatientEHRData() {
        if (!smartClient || !patient) {
            return '';
        }

        try {
            const conditions = await smartClient.request(`Condition?patient=${patient.id}`).catch(() => null);
            const medications = await smartClient.request(`MedicationStatement?patient=${patient.id}`).catch(() => null);

            const ehrData = {
                conditions: conditions?.entry?.map(e => e.resource.code?.text || 'Unknown') || [],
                medications: medications?.entry?.map(e => e.resource.medicationCodeableConcept?.text || 'Unknown') || []
            };

            return JSON.stringify(ehrData);
        } catch (error) {
            console.warn('Failed to fetch EHR data:', error);
            return '';
        }
    }

    /**
     * Generate SOAP note from transcription
     */
    async function generateSOAPNote() {
        const transcript = transcriptionHistory.map(h => h.text).join('\n\n');

        if (!transcript.trim()) {
            showError('No transcription available to summarize');
            return;
        }

        elements.soapSection.classList.remove('hidden');
        elements.soapLoading.classList.remove('hidden');
        elements.soapResults.innerHTML = '';

        try {
            console.log('🔄 Generating SOAP note...');

            const token = await getAccessToken();
            const ehrData = await getPatientEHRData();
            const patientName = patient ? getPatientName(patient) : 'Patient';

            console.log('   URL:', `${config.summarizationServiceUrl}/summarize`);
            console.log('   Token:', token ? '✅' : '❌');

            const response = await fetch(`${config.summarizationServiceUrl}/summarize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    transcript_text: transcript,
                    openemr_text: ehrData,
                    patient_name: patientName
                })
            });

            console.log('   Response status:', response.status);

            if (!response.ok) {
                const errorMessage = await parseError(response);
                throw new Error(errorMessage);
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Summarization failed');
            }

            console.log('✅ SOAP note generated successfully');

            // Store the SOAP note for saving
            currentSoapNote = result.soap_note || '';
            currentSoapResult = result;

            displaySOAPNote(result);

        } catch (error) {
            console.error('SOAP generation error:', error);
            showError('Failed to generate SOAP note: ' + error.message);
            elements.soapSection.classList.add('hidden');
        } finally {
            elements.soapLoading.classList.add('hidden');
        }
    }

    /**
     * Display SOAP note with Save button
     */
    function displaySOAPNote(result) {
        elements.soapResults.innerHTML = `
            <div class="soap-note">
                <pre style="white-space: pre-wrap; font-family: inherit;">${escapeHtml(result.soap_note || 'No summary available')}</pre>
                <div class="soap-actions">
                    <button class="btn btn-primary" id="btn-save-soap" onclick="window.saveSOAPNote()">
                        💾 Save SOAP Note
                    </button>
                    <button class="btn btn-secondary" onclick="window.copySOAPNote()">
                        📋 Copy SOAP Note
                    </button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // SOAP Note Parsing — splits raw text into S/O/A/P sections
    // =========================================================================

    /**
     * Parse a SOAP note string into its four sections.
     * Handles various header formats: "Subjective:", "SUBJECTIVE:", "S:", etc.
     *
     * @param {string} soapText - Full SOAP note text
     * @returns {{ subjective: string, objective: string, assessment: string, plan: string }}
     */
    function parseSOAPSections(soapText) {
        const sections = {
            subjective: '',
            objective: '',
            assessment: '',
            plan: ''
        };

        if (!soapText || !soapText.trim()) {
            return sections;
        }

        // Regex to match common SOAP section headers
        // Matches: "Subjective", "SUBJECTIVE", "S:", "Subjective:", etc.
        const sectionRegex = /(?:^|\n)\s*(?:#{0,3}\s*)?(Subjective|Objective|Assessment|Plan|S|O|A|P)\s*[:\-—]?\s*\n?/gi;

        const markers = [];
        let match;

        while ((match = sectionRegex.exec(soapText)) !== null) {
            const label = match[1].toLowerCase();
            let sectionKey;

            switch (label) {
                case 'subjective': case 's': sectionKey = 'subjective'; break;
                case 'objective':  case 'o': sectionKey = 'objective';  break;
                case 'assessment': case 'a': sectionKey = 'assessment'; break;
                case 'plan':       case 'p': sectionKey = 'plan';       break;
                default: continue;
            }

            markers.push({
                key: sectionKey,
                // Content starts right after the header match
                startIndex: match.index + match[0].length
            });
        }

        if (markers.length === 0) {
            // No recognizable headers — put everything in subjective as fallback
            sections.subjective = soapText.trim();
            return sections;
        }

        // Extract text between consecutive markers
        for (let i = 0; i < markers.length; i++) {
            const start = markers[i].startIndex;
            const end = i + 1 < markers.length
                ? soapText.lastIndexOf('\n', markers[i + 1].startIndex - 1) || markers[i + 1].startIndex
                : soapText.length;

            // Use the regex match start of next marker as the boundary
            const endIndex = i + 1 < markers.length
                ? soapText.indexOf('\n', markers[i + 1].startIndex - markers[i + 1].startIndex) !== -1
                    ? markers[i + 1].startIndex - (soapText.substring(0, markers[i + 1].startIndex).match(/\n\s*(?:#{0,3}\s*)?(?:Subjective|Objective|Assessment|Plan|S|O|A|P)\s*[:\-—]?\s*\n?$/i)?.[0]?.length || 0)
                    : markers[i + 1].startIndex
                : soapText.length;

            sections[markers[i].key] = soapText.substring(start, endIndex).trim();
        }

        // Simpler, more reliable extraction
        // Re-do with a cleaner approach
        const cleanSections = { subjective: '', objective: '', assessment: '', plan: '' };
        const headerPattern = /(?:^|\n)\s*(?:#{0,3}\s*)?(Subjective|Objective|Assessment|Plan|S|O|A|P)\s*[:\-—]?\s*\n/gi;
        const splits = soapText.split(headerPattern);

        // splits will alternate: [textBefore, headerLabel, textAfter, headerLabel, textAfter, ...]
        for (let i = 1; i < splits.length; i += 2) {
            const label = splits[i].toLowerCase();
            const content = (splits[i + 1] || '').trim();
            let key;

            switch (label) {
                case 'subjective': case 's': key = 'subjective'; break;
                case 'objective':  case 'o': key = 'objective';  break;
                case 'assessment': case 'a': key = 'assessment'; break;
                case 'plan':       case 'p': key = 'plan';       break;
                default: continue;
            }

            // If the same section appears multiple times, append
            if (cleanSections[key]) {
                cleanSections[key] += '\n' + content;
            } else {
                cleanSections[key] = content;
            }
        }

        // If the cleaner approach found content, use it; otherwise fall back
        const hasCleanContent = Object.values(cleanSections).some(v => v.length > 0);
        return hasCleanContent ? cleanSections : sections;
    }

    // =========================================================================
    // Save SOAP Note to OpenEMR Encounter
    // =========================================================================

    /**
     * Main entry point for saving the SOAP note.
     * Determines the encounter and saves.
     */
    window.saveSOAPNote = async function () {
        if (!currentSoapNote) {
            showError('No SOAP note available to save.');
            return;
        }

        if (!openemrApi || !patient) {
            showError('Not connected to OpenEMR. Please launch the app from a patient dashboard.');
            return;
        }

        // Update button to show loading
        const saveBtn = document.getElementById('btn-save-soap');
        if (saveBtn) {
            saveBtn.disabled = true;
            saveBtn.textContent = '⏳ Loading encounters...';
        }

        try {
            // Step 1: Check for encounter in SMART launch context
            const contextEncounterId = await openemrApi.getEncounterFromContext();

            if (contextEncounterId) {
                console.log('📋 Found encounter in SMART context:', contextEncounterId);
                await doSaveSoapNote(patient.id, contextEncounterId);
                return;
            }

            // Step 2: Fetch patient encounters and let user pick
            console.log('📋 No encounter in context, fetching patient encounters...');
            const encounters = await openemrApi.getPatientEncounters(patient.id);

            if (!encounters || encounters.length === 0) {
                showError('No encounters found for this patient. Please create an encounter in OpenEMR first.');
                resetSaveButton();
                return;
            }

            // Show encounter selection modal
            showEncounterSelectionModal(encounters);

        } catch (error) {
            console.error('Error preparing to save SOAP note:', error);
            showError('Failed to load encounters: ' + error.message);
            resetSaveButton();
        }
    };

    /**
     * Show a modal for the user to select which encounter to save to.
     */
    function showEncounterSelectionModal(encounters) {
        resetSaveButton();

        if (!elements.encounterModal) {
            console.error('Encounter modal element not found in DOM');
            showError('UI error: encounter selection modal not found.');
            return;
        }

        // Build encounter list HTML
        const listHtml = encounters.map(enc => {
            const id = enc.uuid || enc.id || enc.eid;
            const date = enc.date ? new Date(enc.date).toLocaleDateString() : 'Unknown date';
            const reason = enc.reason || enc.pc_catname || enc.encounter_reason || '';
            const facility = enc.facility || enc.facility_name || '';
            const eid = enc.eid || enc.id || '';

            return `
                <div class="encounter-item" data-encounter-uuid="${escapeHtml(String(id))}">
                    <div class="encounter-info">
                        <strong>${escapeHtml(date)}</strong>
                        ${eid ? `<span class="encounter-eid">#${escapeHtml(String(eid))}</span>` : ''}
                        ${reason ? `<div class="encounter-reason">${escapeHtml(reason)}</div>` : ''}
                        ${facility ? `<div class="encounter-facility">${escapeHtml(facility)}</div>` : ''}
                    </div>
                    <button class="btn btn-primary btn-small btn-select-encounter"
                            onclick="window.selectEncounterAndSave('${escapeHtml(String(id))}')">
                        Select
                    </button>
                </div>
            `;
        }).join('');

        elements.encounterList.innerHTML = listHtml;
        if (elements.encounterLoading) {
            elements.encounterLoading.classList.add('hidden');
        }
        elements.encounterModal.classList.remove('hidden');
    }

    /**
     * Called when user selects an encounter from the modal.
     */
    window.selectEncounterAndSave = async function (encounterUuid) {
        // Close the modal
        elements.encounterModal.classList.add('hidden');

        await doSaveSoapNote(patient.id, encounterUuid);
    };

    /**
     * Perform the actual save of the SOAP note to OpenEMR.
     */
    async function doSaveSoapNote(patientId, encounterUuid) {
        const saveBtn = document.getElementById('btn-save-soap');
        if (saveBtn) {
            saveBtn.disabled = true;
            saveBtn.textContent = '⏳ Saving...';
        }

        try {
            // Parse the SOAP note into sections
            const soapSections = parseSOAPSections(currentSoapNote);

            console.log('💾 Saving SOAP note to encounter:', encounterUuid);
            console.log('   Parsed sections:', {
                subjective: soapSections.subjective.substring(0, 50) + '...',
                objective: soapSections.objective.substring(0, 50) + '...',
                assessment: soapSections.assessment.substring(0, 50) + '...',
                plan: soapSections.plan.substring(0, 50) + '...'
            });

            const result = await openemrApi.saveSoapNote(patientId, encounterUuid, soapSections);

            console.log('✅ SOAP note saved successfully:', result);

            // Update button to show success
            if (saveBtn) {
                saveBtn.disabled = false;
                saveBtn.textContent = '✅ Saved to Encounter!';
                saveBtn.classList.remove('btn-primary');
                saveBtn.classList.add('btn-success');

                setTimeout(() => {
                    saveBtn.textContent = '💾 Save SOAP Note';
                    saveBtn.classList.remove('btn-success');
                    saveBtn.classList.add('btn-primary');
                }, 3000);
            }

        } catch (error) {
            console.error('❌ Failed to save SOAP note:', error);
            showError('Failed to save SOAP note: ' + error.message);
            resetSaveButton();
        }
    }

    /**
     * Reset the save button to its default state
     */
    function resetSaveButton() {
        const saveBtn = document.getElementById('btn-save-soap');
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = '💾 Save SOAP Note';
        }
    }

    /**
     * Copy SOAP note to clipboard (kept as secondary action)
     */
    window.copySOAPNote = async function () {
        const soapText = currentSoapNote || elements.soapResults.innerText;
        try {
            await navigator.clipboard.writeText(soapText);
            alert('SOAP note copied to clipboard!');
        } catch (error) {
            showError('Failed to copy SOAP note');
        }
    };

    /**
     * Clear session
     */
    function clearSession() {
        transcriptionHistory = [];
        currentSoapNote = null;
        currentSoapResult = null;

        elements.transcriptionResults.innerHTML = `
            <p class="placeholder-text">
                Click "Start Ambient Listening" to begin capturing audio.
                The transcription will appear here.
            </p>
        `;
        elements.transcriptionHistory.classList.add('hidden');
        elements.historyList.innerHTML = '';
        elements.actionsSection.classList.add('hidden');
        elements.soapSection.classList.add('hidden');
        elements.soapResults.innerHTML = '';
    }

    /**
     * Show error modal
     */
    function showError(message) {
        console.error('💥 Error shown to user:', message);
        elements.errorMessage.textContent = message;
        elements.errorModal.classList.remove('hidden');
    }

    /**
     * Escape HTML
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format timestamp
     */
    function formatTime(isoString) {
        return new Date(isoString).toLocaleTimeString();
    }

    // Initialize
    init();
});