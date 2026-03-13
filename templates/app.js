// ============================================
// SignBridge - Ultimate Integration Script
// ============================================

// --- Configuration ---
import { createClient } from '@supabase/supabase-js'

// REPLACE with your Supabase info
const supabaseUrl = 'YOUR_PROJECT_URL'
const supabaseKey = 'YOUR_ANON_KEY'
const supabase = createClient(supabaseUrl, supabaseKey)
// -------------------------
// Supabase Database Functions
// -------------------------
async function getSigns() {
    const { data, error } = await supabase
        .from('signs')
        .select('*');
    if (error) console.error('Supabase fetch error:', error);
    else console.log('Signs:', data);
    return data;
}

async function saveSign(name, file_url) {
    const { data, error } = await supabase
        .from('signs')
        .insert([{ name, file_url }]);
    if (error) console.error('Supabase save error:', error);
    else console.log('Saved:', data);
}
const API_ENDPOINT = '/api/predict';
const DETECTION_INTERVAL = 33; // Match ~30 FPS of training data (was 500)
const CONFIDENCE_THRESHOLD = 0.20; // Lowered from 0.40 since 69 classes reduces avg confidence
const COOLDOWN_TIME = 1500; // 1.5 seconds debounce

// --- State ---
let isDetecting = false;
let currentSentence = [];
let lastSign = "";
let lastSignTime = 0;
let videoStream = null;
// --- Emergency Signs ---
const EMERGENCY_SIGNS = ['HELP', 'SICK', 'WATER', 'BATHROOM'];

// ============================================
// 1. Browser Compatibility Check
// ============================================
function checkBrowserCompatibility() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showBrowserError();
        return false;
    }
    return true;
}

function showBrowserError() {
    const main = document.querySelector('.left-col');
    main.innerHTML = `
        <div style="background: red; color: white; padding: 20px; text-align: center; border-radius: 10px;">
            <h2>Incompatible Browser</h2>
            <p>Please upgrade to Chrome/Edge to use AI features.</p>
        </div>
    `;
}

// ============================================
// 2. Webcam Setup
// ============================================
async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });
        const video = document.getElementById('webcam');
        video.srcObject = videoStream;
        video.onloadedmetadata = () => {
            video.play();
            isDetecting = true;
            detectionLoop();
        };
        updateStatus('detected', '👁️ Detecting...');
    } catch (err) {
        console.error('Camera error:', err);
        updateStatus('not-detected', '❌ Camera denied');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    isDetecting = false;
}

// ============================================
// 3. Frame Capture & API
// ============================================
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

async function detectionLoop() {
    if (!isDetecting) return;

    const video = document.getElementById('webcam');
    if (video.readyState !== 4) {
        setTimeout(detectionLoop, DETECTION_INTERVAL);
        return;
    }

    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Send to API
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            handlePrediction(data);
        } catch (err) {
            console.error('API Error:', err);
        } finally {
            // Wait for response before sending next frame to prevent queueing
            setTimeout(detectionLoop, DETECTION_INTERVAL);
        }
    }, 'image/jpeg', 0.7);
}

// ============================================
// 4. Prediction Handler (Core Logic)
// ============================================
function handlePrediction(data) {
    // Handle "No Hand Detection"
    if (!data.detected || !data.prediction || data.confidence < CONFIDENCE_THRESHOLD * 100) {
        updateStatus('not-detected', '👁️ No hand detected');
        document.getElementById('mainSign').textContent = '-';
        document.getElementById('mainConf').textContent = 'Confidence: 0%';
        document.getElementById('statusBadge').className = 'status-badge not-detected';
        return;
    }

    const sign = data.prediction;
    const confidence = data.confidence / 100;

    // Update UI
    updateStatus('detected', `✅ ${sign}`);
    document.getElementById('mainSign').textContent = sign;
    document.getElementById('mainConf').textContent = `Confidence: ${Math.round(data.confidence)}%`;
    document.getElementById('statusBadge').className = 'status-badge detected';

    // Show top 3 predictions
    if (data.predictions && data.predictions.length > 0) {
        const top3List = document.getElementById('top3List');
        top3List.innerHTML = data.predictions.map((s, i) =>
            `<li>${i + 1}. ${s} (${Math.round(data.confidences[i])}%)</li>`
        ).join('');
    }

    // Emergency Mode Check
    if (EMERGENCY_SIGNS.includes(sign)) {
        triggerEmergencyMode(sign);
    }

    // Sentence Builder with Debounce
    const now = Date.now();
    if (sign !== lastSign && (now - lastSignTime) > COOLDOWN_TIME) {
        addToSentence(sign);
        lastSign = sign;
        lastSignTime = now;
    }
}

function updateStatus(type, text) {
    const badge = document.getElementById('statusBadge');
    badge.textContent = text;
    badge.className = `status-badge ${type}`;
}

// ============================================
// 5. Sentence Builder
// ============================================
function addToSentence(sign) {
    currentSentence.push(sign);
    renderSentence();
    speakText(sign); // Speak each word as detected
}

function renderSentence() {
    const text = currentSentence.join(' ') || '(make signs to build sentence)';
    document.getElementById('sentenceText').textContent = text;
}

function speakSentence() {
    const text = currentSentence.join(' ');
    speakText(text);
}

function clearSentence() {
    currentSentence = [];
    lastSign = '';
    lastSignTime = 0;
    renderSentence();
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        window.speechSynthesis.speak(utterance);
    }
}

// ============================================
// 6. Emergency Mode
// ============================================
function triggerEmergencyMode(sign) {
    document.body.classList.add('emergency-flash');

    const msg = `Emergency! This person needs ${sign}!`;
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(msg);
        utterance.rate = 0.7;
        utterance.pitch = 1.2;
        window.speechSynthesis.speak(utterance);
    }

    // Auto-stop flash after 5 seconds
    setTimeout(() => {
        document.body.classList.remove('emergency-flash');
    }, 5000);
}

// ============================================
// 7. Two-Way Communication (Speech-to-Text)
// ============================================
let recognition = null;

function initSpeechRecognition() {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            addHearingMessage(transcript);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };
    }
}

function startVoiceInput() {
    if (recognition) {
        recognition.start();
    } else {
        alert('Speech recognition not supported in this browser');
    }
}

function addHearingMessage(text) {
    const chatBox = document.getElementById('conversationHistory') || createConversationBox();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message hearing';
    msgDiv.innerHTML = `<strong>Hearing Person:</strong> ${text}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function createConversationBox() {
    const container = document.querySelector('.right-col');
    const chatBox = document.createElement('div');
    chatBox.id = 'conversationHistory';
    chatBox.className = 'conversation-box';
    chatBox.style.cssText = 'max-height: 200px; overflow-y: auto; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px;';
    container.insertBefore(chatBox, container.firstChild);
    return chatBox;
}

// ============================================
// 8. Practice Mode
// ============================================
let practiceTargetSign = '';
let practiceMode = false;

function startPracticeMode(targetSign) {
    practiceMode = true;
    practiceTargetSign = targetSign;
    document.getElementById('practiceFeedback').textContent = `Practice: Show "${targetSign}"`;
}

function checkPracticeSign(predictedSign) {
    if (!practiceMode) return;

    const feedbackEl = document.getElementById('practiceFeedback');
    if (predictedSign === practiceTargetSign) {
        feedbackEl.textContent = '✅ Correct!';
        feedbackEl.style.color = '#00ff88';
    } else {
        feedbackEl.textContent = '❌ Try again...';
        feedbackEl.style.color = 'orange';
    }
}

// ============================================
// 9. Sample Collection
// ============================================
let isRecording = false;
let recordingSign = '';
let frameCount = 0;
const MAX_FRAMES = 45;

async function startRecording(signName) {
    isRecording = true;
    recordingSign = signName;
    frameCount = 0;
    document.getElementById('recordingUI').style.display = 'block';
    document.getElementById('recordingSign').textContent = signName;
    recordFrame();
}

function recordFrame() {
    if (!isRecording) return;

    const video = document.getElementById('webcam');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('frame', blob);

        try {
            await fetch('/collect_frame', {
                method: 'POST',
                body: formData
            });

            frameCount++;
            document.getElementById('frameNum').textContent = frameCount;
            document.getElementById('progressFill').style.width = `${(frameCount / MAX_FRAMES) * 100}%`;

            if (frameCount >= MAX_FRAMES) {
                stopRecording();
            } else {
                setTimeout(recordFrame, 33); // ~30fps
            }
        } catch (err) {
            console.error('Recording error:', err);
        }
    }, 'image/jpeg', 0.7);
}

function stopRecording() {
    isRecording = false;
    document.getElementById('recordingUI').style.display = 'none';

    // Save recorded sign in Supabase
    // recordingSign = the name of the sign you just recorded
    // You can replace 'URL_TO_VIDEO_OR_IMAGE' with your actual file URL if you have one
    saveSign(recordingSign, 'URL_TO_VIDEO_OR_IMAGE');

    // Reload grid (can still show locally)
    loadSigns();
}

// ============================================
// 10. Load Signs Grid
// ============================================
async function loadSigns() {
    try {
        const res = await fetch('/samples');
        const samples = await res.json();

        const grid = document.getElementById('signsGrid');
        grid.innerHTML = '';

        for (const [sign, count] of Object.entries(samples)) {
            const card = document.createElement('div');
            card.className = 'sign-card';
            card.innerHTML = `
                <span class="sign-name">${sign}</span>
                <span class="sign-count">${count}</span>
            `;
            card.onclick = () => startRecording(sign);
            grid.appendChild(card);
        }

        document.getElementById('signCount').textContent = Object.keys(samples).length;
    } catch (err) {
        console.error('Load signs error:', err);
    }
}

// ============================================
// 11. Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    if (!checkBrowserCompatibility()) return;

    // Check for existing video element
    const video = document.getElementById('webcam');
    if (video) {
        startCamera();
    }

    // Load signs
    loadSigns();

    // Init speech recognition
    initSpeechRecognition();
    getSigns().then(signs => {
    // signs is an array of all rows from your Supabase table
    console.log('Loaded signs from Supabase:', signs);

    // Example: update your signsGrid with DB signs
    const grid = document.getElementById('signsGrid');
    if(grid) {
        grid.innerHTML = '';
        signs.forEach(sign => {
            const card = document.createElement('div');
            card.className = 'sign-card';
            card.innerHTML = `
                <span class="sign-name">${sign.name}</span>
                <span class="sign-count">Saved</span>
            `;
            card.onclick = () => startRecording(sign.name);
            grid.appendChild(card);
        });
    }
});

    // Expose functions globally for onclick handlers
    window.speakSentence = speakSentence;
    window.clearSentence = clearSentence;
    window.startVoiceInput = startVoiceInput;
    window.startPracticeMode = startPracticeMode;
});
