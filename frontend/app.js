/**
 * ClearPath Support Chatbot - Frontend Logic
 *
 * Features:
 *  - Regular and streaming query modes
 *  - Conversation memory (maintains conversation_id)
 *  - Debug panel with metadata display
 *  - Evaluator warning labels
 *  - Auto-resizing textarea
 *  - Markdown-like rendering for bot responses
 */

const API_BASE = window.location.origin;
let conversationId = null;
let isProcessing = false;

// --- DOM Elements ---
const messagesEl = document.getElementById('messages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const streamToggle = document.getElementById('streamToggle');
const debugPanel = document.getElementById('debugPanel');
const debugContent = document.getElementById('debugContent');
const toggleDebugBtn = document.getElementById('toggleDebug');
const closeDebugBtn = document.getElementById('closeDebug');
const newChatBtn = document.getElementById('newChat');

// --- Event Listeners ---
sendBtn.addEventListener('click', handleSend);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

queryInput.addEventListener('input', autoResize);

toggleDebugBtn.addEventListener('click', () => {
    debugPanel.classList.toggle('hidden');
});

closeDebugBtn.addEventListener('click', () => {
    debugPanel.classList.add('hidden');
});

newChatBtn.addEventListener('click', () => {
    conversationId = null;
    messagesEl.innerHTML = '';
    debugContent.innerHTML = '<div class="debug-empty"><p>Send a message to see debug information.</p></div>';
    addBotMessage(
        '<p>New conversation started! How can I help you with ClearPath today?</p>'
    );
});

// --- Core Functions ---

async function handleSend() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    queryInput.value = '';
    autoResize();

    // Add user message
    addUserMessage(query);

    // Add typing indicator
    const typingEl = addTypingIndicator();

    try {
        if (streamToggle.checked) {
            await handleStreamQuery(query, typingEl);
        } else {
            await handleRegularQuery(query, typingEl);
        }
    } catch (error) {
        console.error('Query error:', error);
        removeTypingIndicator(typingEl);
        addBotMessage('<p>Sorry, something went wrong. Please try again.</p>', null);
    }

    isProcessing = false;
    sendBtn.disabled = false;
    queryInput.focus();
}

async function handleRegularQuery(query, typingEl) {
    const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: query,
            conversation_id: conversationId,
        }),
    });

    removeTypingIndicator(typingEl);

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    conversationId = data.conversation_id;

    const html = renderMarkdown(data.answer);
    addBotMessage(html, data.metadata, data.warning);
    updateDebugPanel(data);
}

async function handleStreamQuery(query, typingEl) {
    const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: query,
            conversation_id: conversationId,
        }),
    });

    if (!response.ok) {
        removeTypingIndicator(typingEl);
        throw new Error(`HTTP ${response.status}`);
    }

    removeTypingIndicator(typingEl);

    // Create the bot message container for streaming
    const messageEl = createBotMessageElement();
    const bubbleEl = messageEl.querySelector('.message-bubble');
    let fullText = '';
    
    // Add streaming cursor indicator
    bubbleEl.innerHTML = '<span class="streaming-cursor">▊</span>';

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const jsonStr = line.slice(6);

            try {
                const event = JSON.parse(jsonStr);

                if (event.type === 'token') {
                    fullText += event.content;
                    bubbleEl.innerHTML = renderMarkdown(fullText) + '<span class="streaming-cursor">▊</span>';
                    scrollToBottom();
                } else if (event.type === 'done') {
                    // Remove cursor when done
                    bubbleEl.innerHTML = renderMarkdown(fullText);
                    conversationId = event.conversation_id;

                    // Add metadata badges and warning
                    if (event.metadata) {
                        addMetaBadges(messageEl, event.metadata);
                    }
                    if (event.warning) {
                        addWarningLabel(messageEl, event.warning);
                    }

                    // Update debug panel
                    updateDebugPanel({
                        metadata: event.metadata,
                        sources: event.sources,
                        conversation_id: event.conversation_id,
                        warning: event.warning,
                    });
                }
            } catch (e) {
                console.warn('Failed to parse SSE event:', e);
            }
        }
    }
}

// --- Message UI Functions ---

function addUserMessage(text) {
    const el = document.createElement('div');
    el.className = 'message user-message fade-in';
    el.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">${escapeHtml(text)}</div>
        </div>
    `;
    messagesEl.appendChild(el);
    scrollToBottom();
}

function addBotMessage(html, metadata = null, warning = null) {
    const el = createBotMessageElement();
    const bubbleEl = el.querySelector('.message-bubble');
    bubbleEl.innerHTML = html;

    if (metadata) {
        addMetaBadges(el, metadata);
    }

    if (warning) {
        addWarningLabel(el, warning);
    }

    scrollToBottom();
    return el;
}

function createBotMessageElement() {
    const el = document.createElement('div');
    el.className = 'message bot-message fade-in';
    el.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                <path d="M12 6v6l4 2"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble"></div>
        </div>
    `;
    messagesEl.appendChild(el);
    return el;
}

function addMetaBadges(messageEl, metadata) {
    const contentEl = messageEl.querySelector('.message-content');
    const metaEl = document.createElement('div');
    metaEl.className = 'message-meta';

    const classification = metadata.classification || 'unknown';
    const model = metadata.model_used || 'unknown';
    const modelShort = model.includes('8b') ? '8B' : model.includes('70b') ? '70B' : model;
    const tokens = metadata.tokens || {};
    const totalTokens = (tokens.input || 0) + (tokens.output || 0);
    const latency = metadata.latency_ms || 0;
    const flags = metadata.evaluator_flags || [];

    metaEl.innerHTML = `
        <span class="meta-badge model">${modelShort} (${classification})</span>
        <span class="meta-badge tokens">${totalTokens} tokens</span>
        <span class="meta-badge">${latency}ms</span>
        ${flags.length > 0
            ? `<span class="meta-badge flagged">${flags.join(', ')}</span>`
            : '<span class="meta-badge" style="color: var(--success);">clean</span>'
        }
    `;

    contentEl.appendChild(metaEl);
}

function addWarningLabel(messageEl, warning) {
    const contentEl = messageEl.querySelector('.message-content');
    const warningEl = document.createElement('div');
    warningEl.className = 'warning-label';
    warningEl.textContent = warning;
    contentEl.appendChild(warningEl);
}

function addTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message bot-message fade-in';
    el.id = 'typing-indicator';
    el.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                <path d="M12 6v6l4 2"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
}

function removeTypingIndicator(el) {
    if (el && el.parentNode) {
        el.parentNode.removeChild(el);
    }
}

// --- Debug Panel ---

function updateDebugPanel(data) {
    const metadata = data.metadata || {};
    const sources = data.sources || [];
    const tokens = metadata.tokens || {};
    const flags = metadata.evaluator_flags || [];

    const classification = metadata.classification || 'unknown';
    const classClass = classification === 'simple' ? 'simple' : 'complex';

    let html = `
        <div class="debug-card">
            <div class="debug-card-header">Routing & Model</div>
            <div class="debug-row">
                <span class="label">Classification</span>
                <span class="value ${classClass}">${classification.toUpperCase()}</span>
            </div>
            <div class="debug-row">
                <span class="label">Model</span>
                <span class="value">${metadata.model_used || 'N/A'}</span>
            </div>
            ${metadata.rationale ? `
            <div class="debug-row">
                <span class="label">Rationale</span>
                <span class="value" style="font-size: 0.82rem; line-height: 1.4;">${metadata.rationale}</span>
            </div>
            ` : ''}
            <div class="debug-row">
                <span class="label">Latency</span>
                <span class="value">${metadata.latency_ms || 0}ms</span>
            </div>
        </div>

        <div class="debug-card">
            <div class="debug-card-header">Token Usage</div>
            <div class="debug-row">
                <span class="label">Input</span>
                <span class="value">${tokens.input || 0}</span>
            </div>
            <div class="debug-row">
                <span class="label">Output</span>
                <span class="value">${tokens.output || 0}</span>
            </div>
            <div class="debug-row">
                <span class="label">Total</span>
                <span class="value">${(tokens.input || 0) + (tokens.output || 0)}</span>
            </div>
        </div>

        <div class="debug-card">
            <div class="debug-card-header">Evaluator</div>
            <div class="debug-row">
                <span class="label">Chunks Retrieved</span>
                <span class="value">${metadata.chunks_retrieved || 0}</span>
            </div>
            <div class="debug-flags">
                ${flags.length > 0
            ? flags.map(f => `<span class="debug-flag warning">${f}</span>`).join('')
            : '<span class="debug-flag ok">No flags</span>'
        }
            </div>
        </div>
    `;

    if (sources.length > 0) {
        html += `
            <div class="debug-card">
                <div class="debug-card-header">Sources (${sources.length})</div>
                <div class="debug-sources">
                    ${sources.map(s => `
                        <div class="debug-source-item">
                            ${s.document}${s.page ? ` (p${s.page})` : ''}
                            <span class="score">${s.relevance_score ? (s.relevance_score * 100).toFixed(0) + '%' : ''}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    if (data.conversation_id) {
        html += `
            <div class="debug-card">
                <div class="debug-card-header">Session</div>
                <div class="debug-row">
                    <span class="label">Conversation ID</span>
                    <span class="value" style="font-size: 0.65rem;">${data.conversation_id}</span>
                </div>
            </div>
        `;
    }

    debugContent.innerHTML = html;
}

// --- Utilities ---

function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Inline code
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Bullet points
    html = html.replace(/^[-•]\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Numbered lists
    html = html.replace(/^\d+\.\s+(.*)$/gm, '<li>$1</li>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n+/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = `<p>${html}</p>`;

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

function autoResize() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
}

// Global function for suggestion chips
window.askQuestion = function (question) {
    queryInput.value = question;
    handleSend();
};
