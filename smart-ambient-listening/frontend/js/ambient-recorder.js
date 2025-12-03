/**
 * Ambient Audio Recorder
 *
 * Uses MediaStream Recording API to capture audio from the browser.
 * Includes audio visualization and chunking support.
 */

class AmbientRecorder {
    constructor(options = {}) {
        this.options = {
            mimeType: this._getSupportedMimeType(),
            audioBitsPerSecond: 128000,
            timeslice: 5000,
            visualizer: null,
            onDataAvailable: null,
            onStart: null,
            onStop: null,
            onError: null,
            ...options
        };

        this.mediaRecorder = null;
        this.audioStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.chunks = [];
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
        this.visualizerInterval = null;
    }

    _getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4',
            'audio/wav'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        return '';
    }

    static isSupported() {
        return !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
    }

    async requestPermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            this.audioStream = stream;
            return { granted: true, stream };
        } catch (error) {
            return {
                granted: false,
                error: error.name === 'NotAllowedError' ? 'Permission denied' : error.message
            };
        }
    }

    _initAnalyzer() {
        if (!this.audioStream || !this.options.visualizer) return;

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;

        const source = this.audioContext.createMediaStreamSource(this.audioStream);
        source.connect(this.analyser);
        this._startVisualization();
    }

    _startVisualization() {
        const canvas = this.options.visualizer;
        if (!canvas || !this.analyser) return;

        const ctx = canvas.getContext('2d');
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            if (!this.isRecording) return;
            this.visualizerInterval = requestAnimationFrame(draw);
            this.analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / bufferLength) * 2.5;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * canvas.height;
                const hue = 200 - (dataArray[i] / 255) * 60;
                ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        };
        draw();
    }

    async start() {
        if (this.isRecording) return;

        if (!this.audioStream) {
            const result = await this.requestPermission();
            if (!result.granted) {
                if (this.options.onError) this.options.onError(new Error(result.error));
                return;
            }
        }

        const recorderOptions = {
            mimeType: this.options.mimeType,
            audioBitsPerSecond: this.options.audioBitsPerSecond
        };

        try {
            this.mediaRecorder = new MediaRecorder(this.audioStream, recorderOptions);
        } catch (error) {
            this.mediaRecorder = new MediaRecorder(this.audioStream);
        }

        this.chunks = [];

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                this.chunks.push(event.data);
                if (this.options.onDataAvailable) {
                    this.options.onDataAvailable(event.data, this.chunks.length);
                }
            }
        };

        this.mediaRecorder.onstart = () => {
            this.isRecording = true;
            this.startTime = Date.now();
            this._startTimer();
            this._initAnalyzer();
            if (this.options.onStart) this.options.onStart();
        };

        this.mediaRecorder.onstop = () => {
            this.isRecording = false;
            this._stopTimer();
            this._stopVisualization();
            const blob = new Blob(this.chunks, { type: this.options.mimeType || 'audio/webm' });
            if (this.options.onStop) this.options.onStop(blob, this.chunks);
        };

        this.mediaRecorder.onerror = (event) => {
            this.isRecording = false;
            if (this.options.onError) this.options.onError(event.error);
        };

        this.mediaRecorder.start(this.options.timeslice);
    }

    stop() {
        if (!this.isRecording || !this.mediaRecorder) return;
        this.mediaRecorder.stop();
    }

    getDuration() {
        if (!this.startTime) return 0;
        return Date.now() - this.startTime;
    }

    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    _startTimer() {
        const timerElement = document.getElementById('recording-timer');
        if (!timerElement) return;
        this.timerInterval = setInterval(() => {
            timerElement.textContent = this.formatDuration(this.getDuration());
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _stopVisualization() {
        if (this.visualizerInterval) {
            cancelAnimationFrame(this.visualizerInterval);
            this.visualizerInterval = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }

    dispose() {
        this.stop();
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
        this._stopVisualization();
        this._stopTimer();
        this.mediaRecorder = null;
        this.chunks = [];
    }

    getBlob() {
        if (this.chunks.length === 0) return null;
        return new Blob(this.chunks, { type: this.options.mimeType || 'audio/webm' });
    }
}

window.AmbientRecorder = AmbientRecorder;