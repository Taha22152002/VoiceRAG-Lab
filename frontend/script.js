// =============================
// CONFIGURATION
// =============================
// Backend server URL (ensure this matches the running backend port)
const BACKEND_URL = 'http://127.0.0.1:5200';
const ENDPOINTS = {
    ingest: `${BACKEND_URL}/ingest/all`,
    chat: `${BACKEND_URL}/chat`
};
const WS_URL = 'ws://127.0.0.1:5201';

// =============================
// DOM ELEMENTS
// =============================
const systemPromptEl = document.getElementById('system-prompt');
const fileUploadEl = document.getElementById('file-upload');
const fileCountEl = document.getElementById('file-count');
const statusBoxEl = document.getElementById('status-box');
const startBtnEl = document.getElementById('start-btn');
const userMessageEl = document.getElementById('user-message');
const sendBtnEl = document.getElementById('send-btn');
const chatbotEl = document.getElementById('chatbot');
const apiEndpointEl = document.getElementById('api-endpoint');

// =============================
// AUDIO CACHE FOR TTS PLAYBACK
// =============================
let audioCache = new Map(); // messageId -> { url, blob }
let currentAudio = { audio: null, id: null };
// Message stacking state
let messageQueue = [];
let isProcessing = false;
let currentRequestController = null; // AbortController for in-flight chat request
let stackingTimer = null; // timer to allow quick successive messages to stack
const STACKING_WINDOW_MS = 800; // user messages within this window will be combined

const textModal = document.getElementById('text-modal');
const textListEl = document.getElementById('text-entry-list');
const linkModal = document.getElementById('link-modal');
const linkListEl = document.getElementById('link-entry-list');
const textEntryCountEl = document.getElementById('text-entry-count');
const linkEntryCountEl = document.getElementById('link-entry-count');
const noTextEntriesEl = document.getElementById('no-text-entries');
const noLinkEntriesEl = document.getElementById('no-link-entries');

// =============================
// STATE
// =============================
let chatHistory = [];
let isChatActive = false;
let activeMode = 'files';
let textEntries = [];
let linkEntries = [];
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let chatSocket = null;
let isListening = false;
let vadAudioContext = null;
let vadProcessor = null;
let vadStream = null;
let selectedMimeTypeForRecorder = null;
let isMuted = false;
let responseAccum = new Map(); // messageId -> accumulated text

// =============================
// INITIAL SETUP
// =============================
apiEndpointEl.value = `RAG Backend at ${BACKEND_URL}`;

document.addEventListener('DOMContentLoaded', () => {
    validateInputs();
});

// =============================
// UTILITY FUNCTIONS
// =============================
function updateStatus(message, type = 'info') {
    statusBoxEl.innerHTML = message;
    statusBoxEl.className = `w-full p-3 rounded-lg text-sm font-semibold status-${type}`;
}

function markdownToHtml(markdownText) {
    let htmlText = markdownText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                               .replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    const lines = htmlText.split('\n');
    let output = [];
    let inList = false;

    lines.forEach(line => {
        const trimmed = line.trim();
        if (/^[\*\-\+] /.test(trimmed)) {
            if (!inList) { output.push('<ul>'); inList = true; }
            output.push(`<li>${trimmed.substring(2).trim()}</li>`);
        } else {
            if (inList) { output.push('</ul>'); inList = false; }
            output.push(line);
        }
    });
    if (inList) output.push('</ul>');
    return output.join('\n').replace(/\n\s*\n/g, '<br><br>').trim();
}

function appendMessage(role, text, sources = [], messageId = null) {
    const isUser = role === 'user';
    const isSystem = role === 'system';
    
    const bubbleClass = isUser ? 'user-bubble' : (isSystem ? 'system-bubble' : 'bot-bubble');
    const justifyClass = isUser ? 'justify-end' : 'justify-start';
    const roundedClass = isUser ? 'rounded-br-sm' : 'rounded-tl-sm';
    const labelText = isUser ? 'User' : (isSystem ? 'System' : 'Bot');
    const labelColor = isUser ? 'text-[#384959]' : 'text-[#88BBF2]';

    const wrapper = document.createElement('div');
    wrapper.className = `flex ${justifyClass}`;

    let sourceHtml = '';
    if (!isUser && !isSystem && sources.length) {
        const sourceList = sources.map((s, i) =>
            `<li><a href="${s.uri}" target="_blank" class="underline text-[#6A89A7] hover:text-[#88BBF2]">${s.title || `Source ${i + 1}`}</a></li>`
        ).join('');
        sourceHtml = `
            <div class="mt-2 pt-2 border-t border-t-[#4A637C] text-xs">
                <strong class="text-[#6A89A7] block mb-1">Sources/Grounding:</strong>
                <ul class="list-none p-0 m-0 text-xs space-y-1">${sourceList}</ul>
            </div>`;
    }

    const msgIdAttr = messageId ? `data-message-id="${messageId}"` : '';
    // status span shows processing without blocking UI
    const statusSpan = (!isSystem && messageId) ? `<span class="text-xs text-[#6A89A7]" id="msg-status-${messageId}"></span>` : '';
    const audioControlsHtml = (!isUser && !isSystem && messageId) ? `
        <div class="mt-2 flex items-center space-x-2">
            <button class="px-2 py-1 text-xs bg-[#384959] text-white rounded" onclick="playMessageAudio('${messageId}')">🔁 Replay</button>
            <span class="text-xs text-[#6A89A7]" id="audio-status-${messageId}">No audio yet</span>
        </div>
    ` : '';

    wrapper.innerHTML = `
        <div class="${bubbleClass} p-3 rounded-xl ${roundedClass} max-w-xs md:max-w-md shadow-lg" ${msgIdAttr}>
            <span class="font-bold text-xs ${labelColor}">${labelText}</span><br>
            <div class="message-content">${markdownToHtml(text)}</div>
            ${sourceHtml}
            ${statusSpan}
            ${audioControlsHtml}
        </div>
    `;

    chatbotEl.appendChild(wrapper);
    chatbotEl.scrollTop = chatbotEl.scrollHeight;
}

// =============================
// TTS HELPERS
// =============================
function generateMessageId() {
    return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function generateAndStoreTTS(text, messageId) {
    try {
        // Request TTS from backend
        const res = await fetch(`${BACKEND_URL}/voice/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Unknown error' }));
            updateMessageWithAudio(messageId, null, `TTS error: ${err.error || 'Unknown error'}`);
            return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        // Cache and update UI
        const prev = audioCache.get(messageId);
        if (prev && prev.url) URL.revokeObjectURL(prev.url);
        audioCache.set(messageId, { url, blob });
        updateMessageWithAudio(messageId, url, 'Ready');
    } catch (e) {
        console.error('[TTS] Generation error:', e);
        updateMessageWithAudio(messageId, null, 'TTS failed');
    }
}

function updateMessageWithAudio(messageId, audioUrl, statusText) {
    const statusEl = document.getElementById(`audio-status-${messageId}`);
    if (statusEl) statusEl.textContent = statusText || (audioUrl ? 'Ready' : 'No audio');
}

async function playMessageAudio(messageId) {
    let entry = audioCache.get(messageId);
    if (!entry || !entry.url) {
        updateMessageWithAudio(messageId, null, 'Generating…');
        const bubble = document.querySelector(`[data-message-id="${messageId}"] .message-content`);
        const text = bubble ? bubble.textContent : '';
        await generateAndStoreTTS(text || '', messageId);
        entry = audioCache.get(messageId);
        if (!entry || !entry.url) {
            updateMessageWithAudio(messageId, null, 'TTS failed');
            return;
        }
    }
    if (currentAudio.audio) {
        try { currentAudio.audio.pause(); } catch {}
    }
    const audio = new Audio(entry.url);
    currentAudio = { audio, id: messageId };
    audio.onended = () => {
        updateMessageWithAudio(messageId, entry.url, 'Ready');
        currentAudio = { audio: null, id: null };
    };
    audio.play().then(() => {
        updateMessageWithAudio(messageId, entry.url, 'Playing');
    }).catch(err => {
        console.error('[TTS] Playback error:', err);
        updateMessageWithAudio(messageId, entry.url, 'Playback error');
    });
}

function pauseMessageAudio(messageId) {
    if (currentAudio.id === messageId && currentAudio.audio) {
        try { currentAudio.audio.pause(); } catch {}
        updateMessageWithAudio(messageId, audioCache.get(messageId)?.url, 'Paused');
    }
}

// =============================
// VALIDATION & UI STATE
// =============================
function changeIngestionMode(mode) {
    activeMode = mode;
    ['files','text','links'].forEach(m => {
        const el = document.getElementById(`ingest-mode-${m}`);
        if(el) el.classList.toggle('hidden', m !== mode);
    });
    validateInputs();
}

function validateInputs() {
    const selectedModeEl = document.querySelector('input[name="ingestion-mode"]:checked');
    activeMode = selectedModeEl ? selectedModeEl.value : 'files';

    const promptValue = systemPromptEl?.value.trim() || '';
    const filesPresent = fileUploadEl?.files?.length > 0;
    const textPresent = textEntries.length > 0;
    const linksPresent = linkEntries.length > 0;
    const isDataPresent = filesPresent || textPresent || linksPresent;

    fileCountEl.textContent = `${fileUploadEl?.files?.length || 0} files selected`;
    textEntryCountEl.textContent = `${textEntries.length} text entries added`;
    linkEntryCountEl.textContent = `${linkEntries.length} link entries added`;

    let modeSummary = [];
    if (filesPresent) modeSummary.push(`Files (${fileUploadEl.files.length})`);
    if (textPresent) modeSummary.push(`Text (${textEntries.length})`);
    if (linksPresent) modeSummary.push(`Links (${linkEntries.length})`);
    modeSummary = modeSummary.join(' & ');

    // Flexible validation for two modes:
    // Basic LLM mode: prompt only
    // Enhanced RAG mode: prompt + any data sources
    const hasPrompt = promptValue !== '';
    const canStartBasicLLM = hasPrompt;
    const canStartRAG = hasPrompt && isDataPresent;

    startBtnEl.disabled = !canStartBasicLLM; // allow start with prompt only
    sendBtnEl.disabled = !isChatActive || !canStartBasicLLM;

    if (!hasPrompt) {
        updateStatus("⚠️ Please provide a **System Prompt** to define the chatbot's role.", 'error');
    } else if (!isChatActive && canStartRAG) {
        updateStatus(`✅ Configured — RAG enabled with: <strong>${modeSummary}</strong>`, 'success');
    } else if (!isChatActive && canStartBasicLLM && !isDataPresent) {
        updateStatus("✅ Configured — Basic LLM mode (no data sources).", 'success');
    }
}

// =============================
// MODAL & ENTRY MANAGEMENT
// =============================
function displayEntries(mode) {
    if (mode === 'text') {
        textListEl.innerHTML = '';
        if (textEntries.length === 0) {
            noTextEntriesEl.classList.remove('hidden');
            return;
        }
        noTextEntriesEl.classList.add('hidden');

        textEntries.forEach((entry, index) => {
            const row = textListEl.insertRow();
            row.innerHTML = `
                <td class="w-1/4">${entry.name}</td>
                <td class="w-3/4">${entry.value.substring(0, 80)}${entry.value.length > 80 ? '...' : ''}</td>
                <td class="text-right w-16">
                    <button onclick="removeEntry('text', ${index})" class="text-red-400 hover:text-red-600 font-bold text-lg leading-none">&times;</button>
                </td>
            `;
        });
    } else if (mode === 'links') {
         linkListEl.innerHTML = '';
         if (linkEntries.length === 0) {
            noLinkEntriesEl.classList.remove('hidden');
            return;
        }
        noLinkEntriesEl.classList.add('hidden');

        linkEntries.forEach((entry, index) => {
            const row = linkListEl.insertRow();
            row.innerHTML = `
                <td class="w-1/4">${entry.name}</td>
                <td class="text-xs text-[#88BBF2] break-all w-3/4">${entry.link}</td>
                <td class="text-right w-16">
                    <button onclick="removeEntry('links', ${index})" class="text-red-400 hover:text-red-600 font-bold text-lg leading-none">&times;</button>
                </td>
            `;
        });
    }
}

function removeEntry(mode, index) {
    if (mode === 'text') textEntries.splice(index,1);
    else if (mode === 'links') linkEntries.splice(index,1);
    displayEntries(mode);
    validateInputs();
}

function openTextModal() { displayEntries('text'); textModal?.classList.remove('hidden'); document.getElementById('text-name-input')?.focus(); }
function closeTextModal() { textModal?.classList.add('hidden'); validateInputs(); }
function openLinkModal() { displayEntries('links'); linkModal?.classList.remove('hidden'); document.getElementById('link-name-input')?.focus(); }
function closeLinkModal() { linkModal?.classList.add('hidden'); validateInputs(); }

function addTextEntry() {
    const nameInput = document.getElementById('text-name-input');
    const valueInput = document.getElementById('text-value-input');
    const errorEl = document.getElementById('text-modal-error');

    const name = nameInput?.value.trim();
    const value = valueInput?.value.trim();
    if (!name || !value) { 
        errorEl.textContent = "Name and Value cannot be empty."; 
        return; 
    }

    // Duplicate name check
    if (textEntries.some(e => e.name === name)) { 
        errorEl.textContent = `Name "${name}" already exists.`; 
        return; 
    }

    // Duplicate value check
    if (textEntries.some(e => e.value === value)) {
        errorEl.textContent = "This text is already added.";
        return;
    }

    textEntries.push({name, value});
    nameInput.value = ''; 
    valueInput.value = ''; 
    errorEl.textContent='';
    displayEntries('text'); 
    nameInput.focus();
}

function addLinkEntry() {
    const nameInput = document.getElementById('link-name-input');
    const linkInput = document.getElementById('link-value-input');
    const errorEl = document.getElementById('link-modal-error');

    const name = nameInput?.value.trim();
    const link = linkInput?.value.trim();
    if (!name || !link) { errorEl.textContent = "Name and Link cannot be empty."; return; }

    if (linkEntries.some(e=>e.name===name)) { errorEl.textContent=`Name "${name}" already exists.`; return; }
    try { new URL(link); } catch {_=> {errorEl.textContent="Invalid URL format."; return;} }
    if (linkEntries.some(e=>e.link===link)) { errorEl.textContent="This exact link is already added."; return; }

    linkEntries.push({name,link});
    nameInput.value=''; linkInput.value=''; errorEl.textContent='';
    displayEntries('links'); nameInput.focus();
}

// =============================
// CHAT LOGIC
// =============================
async function startChat() {
    chatHistory = [];
    chatbotEl.innerHTML = '';
    isChatActive = false;
    // keep UI interactive while starting
    // sendBtnEl.disabled = true;
    // startBtnEl.disabled = true;

    const hasPrompt = systemPromptEl?.value.trim() !== '';
    if (!hasPrompt) {
        updateStatus('❌ System Prompt is required.', 'error');
        validateInputs();
        return;
    }

    const formData = new FormData();
    const files = fileUploadEl?.files || [];
    const totalCount = files.length + textEntries.length + linkEntries.length;

    // If data sources exist, ingest for Enhanced RAG; otherwise skip ingestion and start Basic LLM
    if (totalCount > 0) {
        [...files].forEach(file=>formData.append('files',file));
        formData.append('data', JSON.stringify({textEntries,linkEntries}));

        updateStatus(`⏳ Ingesting ${totalCount} entries...`, 'loading');

        try {
            const res = await fetch(ENDPOINTS.ingest,{method:'POST',body:formData});
            const data = await res.json();
            if(res.ok) updateStatus(`✅ RAG Ready. ${data.message || ''}`, 'success');
            else updateStatus(`❌ Ingestion failed: ${data.error || 'Unknown error.'}`, 'error');
        } catch(e) {
            console.error(e);
            updateStatus(`❌ Network error connecting to RAG server at ${BACKEND_URL}`, 'error');
        }

        appendMessage('system', `Chat started. RAG Pipeline active with ${totalCount} entries.`);
    } else {
        updateStatus('✅ Basic LLM mode enabled. No data sources ingested.', 'success');
        appendMessage('system', 'Chat started. Basic LLM mode (no data sources).');
    }

    isChatActive = true;
    sendBtnEl.disabled = false; startBtnEl.disabled = false;
    userMessageEl?.focus();

    try {
        await connectChatSocket();
    } catch (e) {
        console.error('WS connect error', e);
    }
}

async function sendMessage() {
    const message = userMessageEl?.value.trim();
    if(!message || !isChatActive) return;

    // push to queue and show user bubble
    userMessageEl.value='';
    appendMessage('user', message);
    messageQueue.push({ text: message, ts: Date.now() });

    // start stacking window; reset if more messages come quickly
    if (stackingTimer) clearTimeout(stackingTimer);
    stackingTimer = setTimeout(() => {
        processMessageQueue();
    }, STACKING_WINDOW_MS);
}

// Combine queued messages and send one request
async function processMessageQueue() {
    if (isProcessing || messageQueue.length === 0) return;
    isProcessing = true;
    // Non-blocking: keep input and send enabled for natural typing

    const combined = messageQueue.map(m => m.text).join(' ');
    messageQueue = [];

    // Add to chat history as a single user message
    chatHistory.push({ role: 'user', text: combined });

    // Correlate response without rendering a pending bubble
    const pendingId = generateMessageId();

    try {
        if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {
            const historyPayload = chatHistory.map(m=>({role:m.role==='user'?'user':'model', text:m.text}));
            chatSocket.send(JSON.stringify({ type: 'user_message', text: combined, messageId: pendingId, history: historyPayload }));
        } else {
            const payload = {
                userMessage: combined,
                systemPrompt: systemPromptEl?.value.trim(),
                history: chatHistory.map(m=>({role:m.role==='user'?'user':'model', text:m.text}))
            };
            currentRequestController = new AbortController();
            const { signal } = currentRequestController;
            const res = await fetch(ENDPOINTS.chat,{ method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload), signal });
            const data = await res.json();
            if(res.ok){
                const botResp = data.response || 'No response received';
                const bubbleContent = document.querySelector(`[data-message-id="${pendingId}"] .message-content`);
                if (bubbleContent) {
                    bubbleContent.innerHTML = markdownToHtml(botResp);
                } else {
                    appendMessage('model', botResp, data.sources || [], pendingId);
                }
                chatHistory.push({role:'model',text:botResp});
                try { playMessageAudio(pendingId); } catch {}
            } else {
                const bubbleContent = document.querySelector(`[data-message-id="${pendingId}"] .message-content`);
                const msgText = `❌ LLM Error: ${data?.error || 'Unknown error'}`;
                if (bubbleContent) {
                    bubbleContent.textContent = msgText;
                } else {
                    appendMessage('model', msgText, [], pendingId);
                }
            }
        }
    } catch(e) {
        console.error(e);
        const bubbleContent = document.querySelector(`[data-message-id="${pendingId}"] .message-content`);
        const msgText = '❌ Network error. Could not reach chat server.';
        if (bubbleContent) {
            bubbleContent.textContent = msgText;
        } else {
            appendMessage('model', msgText, [], pendingId);
        }
    } finally {
        isProcessing = false;
        currentRequestController = null;
        userMessageEl.focus();
        if (messageQueue.length > 0) {
            if (stackingTimer) clearTimeout(stackingTimer);
            stackingTimer = setTimeout(() => processMessageQueue(), STACKING_WINDOW_MS);
        }
    }
}

function clearChat() {
    chatHistory=[]; chatbotEl.innerHTML=''; isChatActive=false;
    updateStatus('Chat cleared. Configure settings and start RAG ingestion.', 'error');
    // Cleanup audio cache
    try {
        if (currentAudio.audio) currentAudio.audio.pause();
    } catch {}
    audioCache.forEach(({url}) => { try { URL.revokeObjectURL(url); } catch {} });
    audioCache.clear();
    currentAudio = { audio: null, id: null };
    validateInputs();
}

// =============================
// VOICE RECORDING FUNCTIONS
// =============================
async function toggleVoiceRecording() {
    const voiceBtn = document.getElementById('voice-btn');
    const recordingIndicator = document.getElementById('recording-indicator');
    if (!isListening) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const preferredTypes = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/ogg',
                'audio/wav'
            ];
            selectedMimeTypeForRecorder = null;
            for (const t of preferredTypes) {
                try {
                    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) {
                        selectedMimeTypeForRecorder = t;
                        break;
                    }
                } catch {}
            }
            mediaRecorder = selectedMimeTypeForRecorder ? new MediaRecorder(stream, { mimeType: selectedMimeTypeForRecorder }) : new MediaRecorder(stream);
            audioChunks = [];
            mediaRecorder.ondataavailable = e => { audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                const effectiveType = mediaRecorder.mimeType || (selectedMimeTypeForRecorder || 'audio/webm');
                const blob = new Blob(audioChunks, { type: effectiveType });
                const ext = effectiveType.includes('webm') ? 'webm' : effectiveType.includes('ogg') ? 'ogg' : effectiveType.includes('wav') ? 'wav' : 'webm';
                const filename = `voice_message.${ext}`;
                sendVoiceMessage(blob, filename);
                audioChunks = [];
            };
            vadStream = stream;
            vadAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = vadAudioContext.createMediaStreamSource(stream);
            const hp = vadAudioContext.createBiquadFilter();
            hp.type = 'highpass';
            hp.frequency.value = 180;
            vadProcessor = vadAudioContext.createScriptProcessor(2048, 1, 1);
            let lastVoiceTs = 0;
            const SILENCE_MS = 800;
            const THRESHOLD = 0.02;
            vadProcessor.onaudioprocess = function(e){
                if (isMuted) return;
                const buf = e.inputBuffer.getChannelData(0);
                let sum = 0;
                for (let i = 0; i < buf.length; i++) { const s = buf[i]; sum += s*s; }
                const rms = Math.sqrt(sum / buf.length);
                const now = Date.now();
                if (rms > THRESHOLD) {
                    lastVoiceTs = now;
                    if (!isRecording) { try { mediaRecorder.start(); isRecording = true; updateStatus('🎙️ Recording…', 'loading'); } catch {} }
                } else {
                    if (isRecording && now - lastVoiceTs > SILENCE_MS) { try { mediaRecorder.stop(); isRecording = false; updateStatus('🔄 Transcribing…', 'loading'); } catch {} }
                }
            };
            source.connect(hp);
            hp.connect(vadProcessor);
            vadProcessor.connect(vadAudioContext.destination);
            isListening = true;
            voiceBtn.style.backgroundColor = '#ff4444';
            voiceBtn.textContent = '🔴';
            recordingIndicator.classList.remove('hidden');
        } catch (error) {
            updateStatus('❌ Microphone access denied. Please allow microphone access.', 'error');
        }
    } else {
        try {
            if (isRecording) { try { mediaRecorder.stop(); } catch {} }
            if (vadProcessor) { try { vadProcessor.disconnect(); } catch {} }
            if (vadAudioContext) { try { vadAudioContext.close(); } catch {} }
            if (vadStream) { try { vadStream.getTracks().forEach(t=>t.stop()); } catch {} }
        } catch {}
        isListening = false;
        isRecording = false;
        voiceBtn.style.backgroundColor = '';
        voiceBtn.textContent = '🎤';
        recordingIndicator.classList.add('hidden');
        updateStatus('Call ended.', 'success');
    }
}

function toggleMute() {
    const muteBtn = document.getElementById('mute-btn');
    isMuted = !isMuted;
    if (isMuted) {
        muteBtn.textContent = '🔊';
        muteBtn.style.backgroundColor = '#ff4444';
        if (isRecording) { try { mediaRecorder.stop(); isRecording = false; } catch {} }
        updateStatus('Mic muted.', 'error');
    } else {
        muteBtn.textContent = '🔇';
        muteBtn.style.backgroundColor = '';
        updateStatus('Mic unmuted.', 'success');
    }
}

async function sendVoiceMessage(audioBlob, filename = 'voice_message.webm') {
    const formData = new FormData();
    // Use provided filename to match blob type; backend reads mimetype too
    formData.append('audio', audioBlob, filename);
    
    updateStatus('🔄 Converting speech to text...', 'loading');
    
    try {
        const response = await fetch(`${BACKEND_URL}/voice/stt`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.text) {
            // Auto-send the transcribed text to the model
            userMessageEl.value = data.text;
            updateStatus('✅ Voice transcribed — sending to model…', 'success');
            await sendMessage();
        } else {
            const detail = data && (data.details || data.error);
            updateStatus(`❌ Voice transcription failed: ${detail || 'Unknown error'}`, 'error');
            console.error('[Voice] STT error response:', data);
        }
    } catch (error) {
        console.error('Error sending voice message:', error);
        updateStatus('❌ Network error during voice processing', 'error');
    }
}

// =============================
// INITIAL UI VALIDATION
// =============================
validateInputs();
function connectChatSocket() {
    return new Promise((resolve, reject) => {
        try {
            chatSocket = new WebSocket(WS_URL);
            chatSocket.onopen = () => {
                chatSocket.send(JSON.stringify({ type: 'session_start', systemPrompt: systemPromptEl?.value.trim() }));
                resolve();
            };
            chatSocket.onmessage = (evt) => {
                try {
                    const msg = JSON.parse(evt.data);
                    if (msg.type === 'model_response_chunk') {
                        const existing = responseAccum.get(msg.messageId) || '';
                        const next = existing + (msg.delta || '');
                        responseAccum.set(msg.messageId, next);
                        const bubbleContent = document.querySelector(`[data-message-id="${msg.messageId}"] .message-content`);
                        if (bubbleContent) {
                            bubbleContent.innerHTML = markdownToHtml(next);
                        } else {
                            appendMessage('model', next, [], msg.messageId);
                        }
                    } else if (msg.type === 'model_response_done') {
                        const bubbleContent = document.querySelector(`[data-message-id="${msg.messageId}"] .message-content`);
                        if (bubbleContent) {
                            bubbleContent.innerHTML = markdownToHtml(msg.response || '');
                        } else {
                            appendMessage('model', msg.response || '', msg.sources || [], msg.messageId);
                        }
                        chatHistory.push({ role: 'model', text: msg.response || '' });
                        try { playMessageAudio(msg.messageId); } catch {}
                        responseAccum.delete(msg.messageId);
                    } else if (msg.type === 'error') {
                        const bubbleContent = document.querySelector(`[data-message-id="${msg.messageId}"] .message-content`);
                        const msgText = `❌ LLM Error: ${msg.message || 'Unknown error'}`;
                        if (bubbleContent) bubbleContent.textContent = msgText; else appendMessage('model', msgText, [], msg.messageId);
                    }
                } catch (e) {
                    console.error('WS message parse error', e);
                }
            };
            chatSocket.onerror = (e) => {
                console.error('WS error', e);
            };
            chatSocket.onclose = () => {
                chatSocket = null;
            };
        } catch (e) {
            reject(e);
        }
    });
}
