const messages = document.querySelector('#messages');
const input = document.querySelector('#input');
const sendButton = document.querySelector('#send');
const sourceMode = document.querySelector('#sourceMode');
const imageInput = document.querySelector('#image');
const imagePreview = document.querySelector('#imagePreview');
const previewImage = document.querySelector('#previewImage');
const previewName = document.querySelector('#previewName');
const uploadStatus = document.querySelector('#uploadStatus');
const requestStatus = document.querySelector('#requestStatus');
const healthBadge = document.querySelector('#healthBadge');
const clearButton = document.querySelector('#clear');

const historyButton = document.querySelector('#history');
const historyOverlay = document.querySelector('#historyOverlay');
const closeHistoryButton = document.querySelector('#closeHistory');
const historySearch = document.querySelector('#historySearch');
const historyList = document.querySelector('#historyList');

const knowledgeButton = document.querySelector('#knowledge');
const knowledgeOverlay = document.querySelector('#knowledgeOverlay');
const closeKnowledgeButton = document.querySelector('#closeKnowledge');
const refreshKnowledgeButton = document.querySelector('#refreshKnowledge');
const knowledgeFile = document.querySelector('#knowledgeFile');
const knowledgeStatus = document.querySelector('#knowledgeStatus');
const knowledgeList = document.querySelector('#knowledgeList');
const knowledgeDocumentCount = document.querySelector('#knowledgeDocumentCount');
const knowledgeChunkCount = document.querySelector('#knowledgeChunkCount');

const THREAD_KEY = 'ai-private-chef-thread-id';
const statusMessages = {
  auto: '正在分析需求…',
  knowledge: '正在检索私人知识库…',
  web: '正在准备网络搜索。',
};
let threadId = localStorage.getItem(THREAD_KEY) || crypto.randomUUID();
let imageUrl = null;
let previewObjectUrl = null;
let sending = false;
let uploading = false;
let conversationItems = [];
localStorage.setItem(THREAD_KEY, threadId);

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function renderMarkdown(markdown) {
  const lines = String(markdown || '').replace(/\r/g, '').split('\n');
  const output = [];
  let listType = null;

  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      continue;
    }
    if (line.startsWith('### ')) {
      closeList(); output.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`); continue;
    }
    if (line.startsWith('## ')) {
      closeList(); output.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`); continue;
    }
    const unordered = line.match(/^[-*]\s+(.+)/);
    const ordered = line.match(/^\d+[.)]\s+(.+)/);
    if (unordered) {
      if (listType !== 'ul') { closeList(); listType = 'ul'; output.push('<ul>'); }
      output.push(`<li>${inlineMarkdown(unordered[1])}</li>`); continue;
    }
    if (ordered) {
      if (listType !== 'ol') { closeList(); listType = 'ol'; output.push('<ol>'); }
      output.push(`<li>${inlineMarkdown(ordered[1])}</li>`); continue;
    }
    closeList();
    output.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeList();
  return output.join('');
}

function hideEmptyState() {
  const state = messages.querySelector('.empty-state');
  if (state) state.remove();
}

function showEmptyState(title, description, icon = '🍳') {
  messages.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">${escapeHtml(icon)}</div>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(description)}</p>
    </div>`;
}

function addMessage(role, text = '', options = {}) {
  hideEmptyState();
  const row = document.createElement('article');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '你' : '厨';

  const wrap = document.createElement('div');
  wrap.className = 'message-wrap';
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? '你' : 'AI 私厨';
  const bubble = document.createElement('div');
  bubble.className = `message-bubble${options.typing ? ' typing' : ''}`;
  if (role === 'assistant' && !options.plain) bubble.innerHTML = renderMarkdown(text);
  else bubble.textContent = text;

  wrap.append(label, bubble);
  row.append(avatar, wrap);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return {row, wrap, bubble};
}

function renderSources(messageWrap, sources) {
  if (!sources.length) return;
  let section = messageWrap.querySelector('.message-sources');
  if (!section) {
    section = document.createElement('div');
    section.className = 'message-sources';
    section.innerHTML = '<strong>检索来源</strong><div class="source-list"></div>';
    messageWrap.appendChild(section);
  }
  const list = section.querySelector('.source-list');
  list.innerHTML = '';
  for (const source of sources) {
    const link = document.createElement('a');
    link.className = `source-card ${source.source_type === 'knowledge' ? 'local' : 'web'}`;
    link.href = source.url || '#';
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    const type = source.source_type === 'knowledge' ? '知识库' : '网络';
    const detail = source.source_type === 'knowledge' && source.page
      ? `第 ${source.page} 页${source.score != null ? ` · 相关度 ${Math.round(source.score * 100)}%` : ''}`
      : '外部网页';
    link.innerHTML = `<span>${type}</span><b>${escapeHtml(source.title || '参考资料')}</b><small>${escapeHtml(detail)}</small>`;
    list.appendChild(link);
  }
}

function setRequestStatus(text = '', isError = false) {
  requestStatus.textContent = text;
  requestStatus.classList.toggle('error', isError);
}

function resetImage() {
  imageUrl = null;
  imageInput.value = '';
  imagePreview.classList.add('hidden');
  uploadStatus.textContent = '';
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = null;
  previewImage.removeAttribute('src');
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health', {cache: 'no-store'});
    const data = await response.json();
    const ready = response.ok && data.services?.qwen;
    healthBadge.className = `health ${ready ? 'ok' : 'error'}`;
    const ragText = data.services?.rag ? `RAG ${data.knowledge?.documents || 0} 份` : 'RAG 未配置';
    healthBadge.innerHTML = `<span></span>${ready ? `${escapeHtml(data.model)} · ${ragText}` : '配置不完整'}`;
  } catch (_) {
    healthBadge.className = 'health error';
    healthBadge.innerHTML = '<span></span>服务未连接';
  }
}

async function loadHistory() {
  const requestedThreadId = threadId;
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(requestedThreadId)}`, {cache: 'no-store'});
    if (!response.ok) return;
    const data = await response.json();
    if (requestedThreadId !== threadId) return;
    if (!data.messages?.length) {
      showEmptyState('新会话已经准备好', '告诉我你的食材和需求，或询问私人知识库中的内容。');
      return;
    }
    messages.innerHTML = '';
    for (const item of data.messages) addMessage(item.role, item.content);
    setRequestStatus(`已恢复 ${data.messages.length} 条历史消息`);
  } catch (_) {
    setRequestStatus('历史记录暂时无法加载', true);
  }
}

function formatConversationTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString('zh-CN', {
    month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function renderConversations() {
  const keyword = historySearch.value.trim().toLowerCase();
  const filtered = conversationItems.filter((item) => item.title.toLowerCase().includes(keyword));
  if (!filtered.length) {
    historyList.innerHTML = `<div class="knowledge-empty">${keyword ? '没有找到匹配的对话。' : '还没有可恢复的对话。'}</div>`;
    return;
  }
  historyList.innerHTML = filtered.map((item) => `
    <button class="history-item${item.thread_id === threadId ? ' active' : ''}" type="button" data-thread-id="${escapeHtml(item.thread_id)}">
      <strong>${escapeHtml(item.title)}</strong>
      <span>${escapeHtml(formatConversationTime(item.updated_at))} · ${item.message_count} 条消息</span>
    </button>
  `).join('');
}

async function loadConversations() {
  historyList.innerHTML = '<div class="knowledge-empty">正在读取历史对话…</div>';
  try {
    const response = await fetch('/api/conversations', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '历史对话加载失败');
    conversationItems = data.conversations || [];
    renderConversations();
  } catch (error) {
    historyList.innerHTML = `<div class="knowledge-empty error">${escapeHtml(error.message)}</div>`;
  }
}

function syncDrawerState() {
  const drawerOpen = !historyOverlay.classList.contains('hidden')
    || !knowledgeOverlay.classList.contains('hidden');
  document.body.classList.toggle('drawer-open', drawerOpen);
}

function openHistory() {
  historyOverlay.classList.remove('hidden');
  historyOverlay.setAttribute('aria-hidden', 'false');
  syncDrawerState();
  historySearch.value = '';
  loadConversations();
  historySearch.focus();
}

function closeHistory() {
  historyOverlay.classList.add('hidden');
  historyOverlay.setAttribute('aria-hidden', 'true');
  syncDrawerState();
}

async function requestConversationTitle(targetThreadId) {
  try {
    await fetch(`/api/conversations/${encodeURIComponent(targetThreadId)}/title`, {method: 'POST'});
  } catch (_) {
    // The history list can always fall back to the first user message.
  }
}

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
});

document.querySelector('#new').addEventListener('click', () => {
  if (sending || uploading) {
    setRequestStatus('请等待当前操作完成后再新建会话', true);
    return;
  }
  threadId = crypto.randomUUID();
  localStorage.setItem(THREAD_KEY, threadId);
  showEmptyState('新会话已经准备好', '告诉我你的食材和需求，或询问私人知识库中的内容。');
  input.value = '';
  resetImage();
  setRequestStatus('已创建新会话');
  input.focus();
});

clearButton.addEventListener('click', async () => {
  if (!window.confirm('确定清空当前会话记录吗？此操作不能撤销。')) return;
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(threadId)}`, {method: 'DELETE'});
    if (!response.ok) throw new Error('清空失败');
    threadId = crypto.randomUUID();
    localStorage.setItem(THREAD_KEY, threadId);
    showEmptyState('记录已清空', '可以开始一段新的美食对话。', '✓');
    resetImage();
    setRequestStatus('当前会话历史已删除');
  } catch (error) {
    setRequestStatus(error.message, true);
  }
});

document.querySelector('#removeImage').addEventListener('click', resetImage);

imageInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    resetImage();
    setRequestStatus('图片不能超过 8 MB', true);
    return;
  }

  resetImage();
  previewObjectUrl = URL.createObjectURL(file);
  previewImage.src = previewObjectUrl;
  previewName.textContent = file.name;
  imagePreview.classList.remove('hidden');
  uploadStatus.textContent = '正在安全上传…';
  uploading = true;
  sendButton.disabled = true;
  setRequestStatus('正在上传食材图片…');

  try {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/api/upload', {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '上传失败');
    imageUrl = data.image_url;
    uploadStatus.textContent = `上传成功 · ${data.storage.toUpperCase()}`;
    setRequestStatus('图片已就绪，可以直接发送或补充要求');
  } catch (error) {
    imageUrl = null;
    uploadStatus.textContent = '上传失败';
    setRequestStatus(error.message, true);
  } finally {
    uploading = false;
    sendButton.disabled = false;
  }
});

async function sendMessage() {
  if (sending || uploading) return;
  let text = input.value.trim();
  if (!text && !imageUrl) {
    setRequestStatus('请先输入需求，或者上传一张食材图片', true);
    input.focus();
    return;
  }
  if (!text && imageUrl) {
    text = '请识别图片中的食材，并推荐两道适合的菜，说明做法、难度、营养评价和参考来源。';
  }

  const requestImageUrl = imageUrl;
  const selectedSourceMode = sourceMode.value;
  const requestThreadId = threadId;
  sending = true;
  sendButton.disabled = true;
  sendButton.querySelector('span').textContent = '生成中';
  input.value = '';
  addMessage('user', requestImageUrl ? `📷 ${text}` : text, {plain: true});
  const answerMessage = addMessage('assistant', '正在连接 AI 私厨…', {typing: true, plain: true});
  setRequestStatus(statusMessages[selectedSourceMode]);

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: text,
        image_url: requestImageUrl,
        thread_id: requestThreadId,
        source_mode: selectedSourceMode,
      }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `请求失败（${response.status}）`);
    }
    if (!response.body) throw new Error('浏览器没有收到流式响应');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const sourceMap = new Map();
    let buffer = '';
    let fullText = '';

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const events = buffer.split('\n\n');
      buffer = events.pop();
      for (const event of events) {
        if (!event.startsWith('data: ')) continue;
        const data = JSON.parse(event.slice(6));
        if (data.type === 'status') setRequestStatus(data.message);
        if (data.type === 'sources') {
          for (const source of data.sources || []) {
            const key = `${source.source_type}|${source.url}|${source.document_id}|${source.page}`;
            sourceMap.set(key, source);
          }
          renderSources(answerMessage.wrap, [...sourceMap.values()]);
        }
        if (data.type === 'text') {
          fullText += data.text;
          answerMessage.bubble.classList.remove('typing');
          answerMessage.bubble.innerHTML = renderMarkdown(fullText);
          messages.scrollTop = messages.scrollHeight;
        }
        if (data.type === 'error') throw new Error(data.message);
        if (data.type === 'done') setRequestStatus('回答已完成，可以继续追问');
      }
    }
    if (!fullText) throw new Error('回答已结束，但没有收到文本内容');
    resetImage();
    requestConversationTitle(requestThreadId);
  } catch (error) {
    answerMessage.bubble.classList.remove('typing');
    answerMessage.bubble.textContent = `发送失败：${error.message}`;
    setRequestStatus(error.message, true);
  } finally {
    sending = false;
    sendButton.disabled = false;
    sendButton.querySelector('span').textContent = '发送';
  }
}

sendButton.addEventListener('click', sendMessage);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function setKnowledgeStatus(text = '', isError = false) {
  knowledgeStatus.textContent = text;
  knowledgeStatus.classList.toggle('error', isError);
}

function renderKnowledgeDocuments(documents) {
  if (!documents.length) {
    knowledgeList.innerHTML = '<div class="knowledge-empty">知识库还是空的，先上传一份菜谱或营养资料吧。</div>';
    return;
  }
  knowledgeList.innerHTML = documents.map((document) => `
    <article class="knowledge-item">
      <div class="file-icon">${document.file_type.toUpperCase()}</div>
      <div class="file-info">
        <a href="/api/knowledge/documents/${encodeURIComponent(document.id)}/download" target="_blank" rel="noopener noreferrer">${escapeHtml(document.filename)}</a>
        <span>${formatBytes(document.size_bytes)} · ${document.chunk_count} 个片段 · ${escapeHtml(new Date(document.created_at).toLocaleString())}</span>
      </div>
      <button type="button" class="delete-document" data-document-id="${escapeHtml(document.id)}" data-filename="${escapeHtml(document.filename)}">删除</button>
    </article>
  `).join('');
}

async function loadKnowledge() {
  knowledgeList.innerHTML = '<div class="knowledge-empty">正在读取知识库…</div>';
  try {
    const response = await fetch('/api/knowledge/documents', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '知识库加载失败');
    knowledgeDocumentCount.textContent = data.documents?.length || 0;
    knowledgeChunkCount.textContent = data.chunks || 0;
    renderKnowledgeDocuments(data.documents || []);
  } catch (error) {
    knowledgeList.innerHTML = '<div class="knowledge-empty error">知识库暂时无法加载。</div>';
    setKnowledgeStatus(error.message, true);
  }
}

function openKnowledge() {
  knowledgeOverlay.classList.remove('hidden');
  knowledgeOverlay.setAttribute('aria-hidden', 'false');
  syncDrawerState();
  loadKnowledge();
}

function closeKnowledge() {
  knowledgeOverlay.classList.add('hidden');
  knowledgeOverlay.setAttribute('aria-hidden', 'true');
  syncDrawerState();
}

historyButton.addEventListener('click', openHistory);
closeHistoryButton.addEventListener('click', closeHistory);
historySearch.addEventListener('input', renderConversations);
historyOverlay.addEventListener('click', (event) => {
  if (event.target === historyOverlay) closeHistory();
});
historyList.addEventListener('click', async (event) => {
  const button = event.target.closest('.history-item');
  if (!button) return;
  if (sending || uploading) {
    setRequestStatus('请等待当前操作完成后再切换对话', true);
    closeHistory();
    return;
  }
  threadId = button.dataset.threadId;
  localStorage.setItem(THREAD_KEY, threadId);
  closeHistory();
  resetImage();
  await loadHistory();
  input.focus();
});

knowledgeButton.addEventListener('click', openKnowledge);
closeKnowledgeButton.addEventListener('click', closeKnowledge);
refreshKnowledgeButton.addEventListener('click', loadKnowledge);
knowledgeOverlay.addEventListener('click', (event) => {
  if (event.target === knowledgeOverlay) closeKnowledge();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !historyOverlay.classList.contains('hidden')) closeHistory();
  if (event.key === 'Escape' && !knowledgeOverlay.classList.contains('hidden')) closeKnowledge();
});

knowledgeFile.addEventListener('change', async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    if (file.size > 10 * 1024 * 1024) {
      setKnowledgeStatus(`${file.name} 超过 10 MB，已跳过。`, true);
      continue;
    }
    setKnowledgeStatus(`正在解析并向量化 ${file.name}（${index + 1}/${files.length}）…`);
    const form = new FormData();
    form.append('file', file);
    try {
      const response = await fetch('/api/knowledge/documents', {method: 'POST', body: form});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || '上传失败');
      const duplicateText = data.duplicate ? '（文件已存在，未重复索引）' : `，生成 ${data.chunk_count} 个知识片段`;
      setKnowledgeStatus(`${data.filename} 已就绪${duplicateText}`);
    } catch (error) {
      setKnowledgeStatus(`${file.name}：${error.message}`, true);
    }
  }
  knowledgeFile.value = '';
  await loadKnowledge();
  await checkHealth();
});

knowledgeList.addEventListener('click', async (event) => {
  const button = event.target.closest('.delete-document');
  if (!button) return;
  const filename = button.dataset.filename;
  if (!window.confirm(`确定删除“${filename}”及其全部向量吗？`)) return;
  button.disabled = true;
  try {
    const response = await fetch(`/api/knowledge/documents/${encodeURIComponent(button.dataset.documentId)}`, {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '删除失败');
    setKnowledgeStatus(`${filename} 已从知识库删除。`);
    await loadKnowledge();
    await checkHealth();
  } catch (error) {
    setKnowledgeStatus(error.message, true);
    button.disabled = false;
  }
});

checkHealth();
loadHistory();
