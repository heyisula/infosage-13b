/* InfoSage AI - Interaction Logic */
window.onerror = function (msg, url, line, col, error) {
    alert("JS Error: " + msg + "\nLine: " + line);
    return false;
};

// State
let currentChatId = null;
let messages = [];
let isGenerating = false;
let statusPollTimer = null;
let currentModelStatus = 'stopped';

// DOM Elements
const els = {
    messages: document.getElementById('messages'),
    input: document.getElementById('input'),
    sendBtn: document.getElementById('send-btn'),
    modelBtn: document.getElementById('model-btn'),
    statusRing: document.getElementById('status-ring'),
    statusText: document.getElementById('status-text'),
    vramBar: document.getElementById('vram-bar'),
    vramInfo: document.getElementById('vram-info'),
    modelName: document.getElementById('model-name'),
    gpuName: document.getElementById('gpu-name'),
    chatList: document.getElementById('chat-list'),
    chatTitle: document.getElementById('chat-title'),
    welcome: document.getElementById('welcome'),
    sidebar: document.getElementById('sidebar'),
    themeIcon: document.getElementById('theme-icon')
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    checkModelStatus();
    loadChatList();
    els.input.focus();

    // Auto-resize input
    els.input.addEventListener('input', () => {
        const hasText = els.input.value.trim().length > 0;
        els.sendBtn.disabled = !hasText;
        autoResize(els.input);
    });

    statusPollTimer = setInterval(checkModelStatus, 3000);
    if (window.lucide) lucide.createIcons();

    // Event Listener
    els.modelBtn.addEventListener('click', toggleModel);

    // Ambient System
    document.addEventListener('mousemove', (e) => {
        const x = e.clientX;
        const y = e.clientY;

        // Update CSS variables for spotlight
        document.body.style.setProperty('--mouse-x', `${x}px`);
        document.body.style.setProperty('--mouse-y', `${y}px`);

        // Parallax effect for orbs
        const orbs = document.querySelectorAll('.orb');
        orbs.forEach((orb, index) => {
            const speed = (index + 1) * 20;
            const xOffset = (window.innerWidth / 2 - x) / speed;
            const yOffset = (window.innerHeight / 2 - y) / speed;
            orb.style.transform = `translate(${xOffset}px, ${yOffset}px)`;
        });
    });
});

// Theme System
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    if (theme === 'dark') {
        els.themeIcon.setAttribute('data-lucide', 'moon');
    } else {
        els.themeIcon.setAttribute('data-lucide', 'sun');
    }
    if (window.lucide) lucide.createIcons();
}

// Model Logic
async function checkModelStatus() {
    try {
        const res = await fetch('/api/model/status');
        const data = await res.json();
        updateModelUI(data);
    } catch {
        updateModelUI({ status: 'stopped' });
    }
}

function updateModelUI(data) {
    const status = (data.status || 'stopped').toLowerCase(); // Normalize
    currentModelStatus = status;

    // Status Indicator
    els.statusRing.className = `pulse-ring ${status}`;
    els.statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);

    // VRAM Bar
    if (data.vram_free_mb && data.vram_total_mb) {
        const used = data.vram_total_mb - data.vram_free_mb;
        const pct = (used / data.vram_total_mb) * 100;

        els.vramBar.style.width = `${pct}%`;
        els.vramInfo.textContent = `${(used / 1024).toFixed(1)} / ${(data.vram_total_mb / 1024).toFixed(0)} GB`;
    }

    if (data.gpu_name && els.gpuName) {
        els.gpuName.textContent = data.gpu_name;
    }
    if (data.model_name && els.modelName) {
        els.modelName.textContent = data.model_name;
    }

    // Button State
    if (status === 'stopped') {
        els.modelBtn.className = 'btn-primary';
        els.modelBtn.disabled = false;
        els.modelBtn.innerHTML = '<i data-lucide="power"></i><span>Start Engine</span>';
    } else if (status === 'loading') {
        els.modelBtn.className = 'btn-primary loading';
        els.modelBtn.disabled = true;
        els.modelBtn.innerHTML = '<i data-lucide="loader-2"></i><span>Ignition...</span>';
    } else if (status === 'ready') {
        els.modelBtn.className = 'btn-primary stop';
        els.modelBtn.disabled = false;
        els.modelBtn.innerHTML = '<i data-lucide="square"></i><span>Stop Engine</span>';
    }
    if (window.lucide) lucide.createIcons();
}

async function toggleModel() {
    if (currentModelStatus === 'stopped') {
        // Optimistic UI Update
        updateModelUI({ status: 'loading' });
        try {
            const res = await fetch('/api/model/start', { method: 'POST' });

            if (!res.ok) {
                const txt = await res.text();
                throw new Error("Server error: " + res.status);
            }
        } catch (e) {
            console.error("Failed to start:", e);
            updateModelUI({ status: 'stopped' });
        }
    } else if (currentModelStatus === 'ready') {
        await fetch('/api/model/stop', { method: 'POST' });
    }

    // Poll shortly after
    setTimeout(checkModelStatus, 1000);
}

// Chat Logic
async function sendMessage() {
    const text = els.input.value.trim();
    if (!text || isGenerating) return;

    if (els.welcome) els.welcome.style.display = 'none';
    if (!currentChatId) {
        currentChatId = generateId();
        messages = [];
    }

    // UI Updates
    renderMessage('user', text);
    messages.push({ role: 'user', content: text });

    els.input.value = '';
    els.input.style.height = 'auto';
    els.sendBtn.disabled = true;
    isGenerating = true;

    // Show Typing
    const typingID = showTyping();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await res.json();
        removeTyping(typingID);

        if (data.error) {
            renderMessage('bot', ` Error: ${data.error}`);
        } else {
            renderMessage('bot', data.response, data.source);
            messages.push({ role: 'bot', content: data.response, source: data.source });
        }

        // Updating title
        if (messages.length === 2) {
            const title = text.slice(0, 40) + (text.length > 40 ? '...' : '');
            els.chatTitle.textContent = title;
        }

        saveChat();
        loadChatList();

    } catch (e) {
        removeTyping(typingID);
        renderMessage('bot', ` Connection Failure: ${e.message}`);
    }

    isGenerating = false;
    els.input.focus();
}

function renderMessage(role, content, source) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const initial = role === 'user' ? 'You' : 'AI';
    const icon = role === 'user' ? 'user' : 'bot';

    let html = `
        <div class="msg-avatar">
            <i data-lucide="${icon}"></i>
        </div>
        <div class="msg-content">
            <div class="msg-bubble">${escapeHtml(content)}</div>
    `;

    if (source && source !== 'none') {
        html += `<div class="source-pills">`;
        source.split(' + ').forEach(s => {
            const type = s.toLowerCase().includes('local') ? 'local' : 'live';
            html += `<span class="source-pill ${type}">${s}</span>`;
        });
        html += `</div>`;
    }

    html += `</div>`;

    div.innerHTML = html;
    els.messages.appendChild(div);
    if (window.lucide) lucide.createIcons();
    els.messages.scrollTo({ top: els.messages.scrollHeight, behavior: 'smooth' });
}

function showTyping() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message bot typing';
    div.id = id;
    div.innerHTML = `
        <div class="msg-avatar"><i data-lucide="bot"></i></div>
        <div class="msg-content">
            <div class="msg-bubble">Thinking...</div>
        </div>
    `;
    els.messages.appendChild(div);
    if (window.lucide) lucide.createIcons();
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Chat History Sync
async function saveChat() {
    if (!currentChatId) return;
    await fetch('/api/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            id: currentChatId,
            title: els.chatTitle.textContent,
            messages
        })
    });
}

async function loadChatList() {
    const res = await fetch('/api/history');
    const chats = await res.json();

    els.chatList.innerHTML = chats.map(c => `
        <div class="history-item ${c.id === currentChatId ? 'active' : ''}" 
             onclick="loadChat('${c.id}')">
            <div class="history-title">${escapeHtml(c.title)}</div>
        </div>
    `).join('');
}

async function loadChat(id) {
    const res = await fetch(`/api/history/${id}`);
    const data = await res.json();

    currentChatId = data.id;
    messages = data.messages || [];
    els.chatTitle.textContent = data.title;

    // Clear & Re-render
    els.messages.innerHTML = '';
    if (els.welcome) els.welcome.style.display = messages.length ? 'none' : 'block';

    messages.forEach(m => renderMessage(m.role, m.content, m.source));
    loadChatList();
}

function newChat() {
    currentChatId = null;
    messages = [];
    els.messages.innerHTML = '';
    els.messages.appendChild(els.welcome);
    els.welcome.style.display = 'block';
    els.chatTitle.textContent = 'New Conversation';
    loadChatList();
}

async function deleteCurrentChat() {
    if (!currentChatId || !confirm('Delete this chat?')) return;
    await fetch(`/api/history/${currentChatId}`, { method: 'DELETE' });
    newChat();
}

// Utilities
function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function generateId() {
    return Math.random().toString(36).substr(2, 9);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
}

function toggleSidebar() {
    els.sidebar.classList.toggle('open');
}
