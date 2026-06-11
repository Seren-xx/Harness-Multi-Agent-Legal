import streamlit as st
import sqlite3
import json
import os
import uuid
import config
from step05_multi_agent_brain import legal_brain
from step04_test_retrieval_search_bm25_rerank import get_vector_store, execute_legal_search
from step01_word_to_md import parse_document, clean_text
from memory_manager import memory_manager

SQLITE_DB_PATH = os.path.join(config.DATA_DIR, "chat_history.db")


def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id TEXT PRIMARY KEY, conversation_id TEXT, user_query TEXT, answer TEXT, feedback INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')
    conn.commit()
    
    # 检查并迁移旧表结构
    try:
        c.execute("SELECT conversation_id FROM chat_history LIMIT 1")
    except sqlite3.OperationalError:
        # 旧表没有conversation_id列，需要迁移
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history_new
                     (id TEXT PRIMARY KEY, conversation_id TEXT, user_query TEXT, answer TEXT, feedback INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')
        c.execute("INSERT INTO chat_history_new (id, user_query, answer, feedback, timestamp) SELECT id, user_query, answer, feedback, timestamp FROM chat_history")
        c.execute("DROP TABLE chat_history")
        c.execute("ALTER TABLE chat_history_new RENAME TO chat_history")
        conn.commit()
    
    conn.close()


def create_conversation(title="新对话"):
    """创建新对话"""
    conv_id = str(uuid.uuid4())
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (conv_id, title))
    conn.commit()
    conn.close()
    return conv_id


def load_conversations():
    """加载所有对话列表"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows


def delete_conversation(conv_id):
    """删除整个对话及其历史记录"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE conversation_id = ?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()


def load_history(conversation_id):
    """加载指定对话的历史记录"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, user_query, answer, feedback, timestamp FROM chat_history WHERE conversation_id = ? ORDER BY timestamp ASC", (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def save_chat(conversation_id, query, answer, feedback=None):
    """保存聊天记录到指定对话"""
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (id, conversation_id, user_query, answer, feedback) VALUES (?, ?, ?, ?, ?)",
              (chat_id, conversation_id, query, answer, feedback))
    c.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return chat_id


def process_uploaded_file(file_path):
    """处理上传的文件"""
    text = parse_document(file_path)
    if text:
        name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(config.PROCESSED_DATA_DIR, f"{name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"✅ 文件处理成功：{name}.md"
    return "❌ 文件处理失败"


# Streamlit 页面配置
st.set_page_config(
    page_title="AI 虚拟顶尖律所",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设计系统：专业、权威、冷静、可信赖
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
    
    :root {
        --color-primary: #1a365d;
        --color-primary-light: #2c5282;
        --color-secondary: #4a5568;
        --color-accent: #d69e2e;
        --color-accent-light: #ecc94b;
        --color-bg: #f7fafc;
        --color-surface: #ffffff;
        --color-text: #2d3748;
        --color-text-light: #718096;
        --color-border: #e2e8f0;
        --color-success: #38a169;
        --color-error: #e53e3e;
        --space-xs: 4px;
        --space-sm: 8px;
        --space-md: 16px;
        --space-lg: 24px;
        --space-xl: 32px;
        --space-2xl: 48px;
        --space-3xl: 64px;
        --font-display: 'Playfair Display', Georgia, serif;
        --font-body: 'Source Sans 3', -apple-system, sans-serif;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .main .block-container {
        padding-top: var(--space-xl);
        padding-bottom: var(--space-3xl);
        max-width: 900px;
    }
    
    .header-container {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-light) 100%);
        padding: var(--space-2xl) var(--space-xl);
        border-radius: 12px;
        margin-bottom: var(--space-xl);
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    .header-title {
        font-family: var(--font-display);
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 var(--space-sm) 0;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-family: var(--font-body);
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        font-weight: 300;
        color: rgba(255,255,255,0.85);
        margin: 0;
        line-height: 1.6;
    }
    
    .stChatMessage {
        border: 1px solid var(--color-border);
        border-radius: 8px;
        padding: var(--space-md);
        margin-bottom: var(--space-md);
        background: var(--color-surface);
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }
    
    .stChatMessage:hover {
        box-shadow: var(--shadow-md);
    }
    
    .user-message {
        background: var(--color-primary);
        color: #ffffff;
        border: none;
    }
    
    .assistant-message {
        background: var(--color-surface);
        border-left: 3px solid var(--color-accent);
    }
    
    .sidebar-section {
        margin-bottom: var(--space-lg);
        padding: var(--space-md);
        background: var(--color-surface);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
    }
    
    .sidebar-title {
        font-family: var(--font-display);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--color-primary);
        margin-bottom: var(--space-sm);
        padding-bottom: var(--space-xs);
        border-bottom: 1px solid var(--color-border);
    }
    
    .stButton > button {
        border-radius: 6px;
        font-family: var(--font-body);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .empty-state {
        text-align: center;
        padding: var(--space-3xl) var(--space-xl);
        color: var(--color-text-light);
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: var(--space-md);
    }
    
    .empty-state-title {
        font-family: var(--font-display);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: var(--space-sm);
        color: var(--color-text);
    }
    
    .empty-state-text {
        font-family: var(--font-body);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 500px;
        margin: 0 auto;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: var(--space-md);
            padding-bottom: var(--space-xl);
        }
        
        .header-container {
            padding: var(--space-lg) var(--space-md);
        }
    }
</style>
""", unsafe_allow_html=True)

# 初始化数据库
init_db()

# 侧边栏
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">💬 对话管理</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ 新建对话", use_container_width=True):
        new_conv_id = create_conversation("新对话")
        st.session_state.conversation_id = new_conv_id
        st.session_state.messages = []
        st.session_state.conv_title = "新对话"
        st.rerun()
    
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    conversations = load_conversations()
    if conversations:
        for conv_id, title, created_at, updated_at in conversations:
            is_active = st.session_state.get("conversation_id") == conv_id
            btn_label = f"💬 {title}" if is_active else f"📄 {title}"
            if st.button(btn_label, key=f"conv_{conv_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.conversation_id = conv_id
                st.session_state.conv_title = title
                history_msgs = load_history(conv_id)
                messages = []
                for _, q, a, fb, ts in history_msgs:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
                st.session_state.messages = messages
                st.rerun()
            if st.button("🗑️", key=f"del_conv_{conv_id}"):
                delete_conversation(conv_id)
                if st.session_state.get("conversation_id") == conv_id:
                    st.session_state.conversation_id = None
                    st.session_state.messages = []
                st.rerun()
    else:
        st.info("暂无对话，点击「新建对话」开始")
    
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">📤 上传法律文档</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "选择 .docx 文件",
        type=["docx"],
        help="支持 Word 格式的法律文档"
    )
    
    if uploaded_file:
        with st.spinner("正在处理文档..."):
            temp_path = os.path.join(config.DATA_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = process_uploaded_file(temp_path)
            if "成功" in result:
                st.success(result)
            else:
                st.error(result)
            os.remove(temp_path)

# 主聊天区
if "conversation_id" not in st.session_state:
    conversations = load_conversations()
    if conversations:
        st.session_state.conversation_id = conversations[0][0]
        st.session_state.conv_title = conversations[0][1]
        history_msgs = load_history(st.session_state.conversation_id)
        messages = []
        for _, q, a, fb, ts in history_msgs:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        st.session_state.messages = messages
    else:
        st.session_state.conversation_id = create_conversation("新对话")
        st.session_state.conv_title = "新对话"
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# 头部
st.markdown(f"""
<div class="header-container">
    <h1 class="header-title">⚖️ AI 虚拟顶尖律所</h1>
    <p class="header-subtitle">当前对话：{st.session_state.conv_title} | 多智能体法律助手 | 混合检索+重排 | 自我反思 | 多轮对话</p>
</div>
""", unsafe_allow_html=True)

# 空状态
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">⚖️</div>
        <div class="empty-state-title">欢迎使用 AI 法律助手</div>
        <div class="empty-state-text">
            我可以帮您解答法律问题、分析案例、检索法条。
            请在下方输入您的法律问题开始咨询。
        </div>
    </div>
    """, unsafe_allow_html=True)

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="background: var(--color-primary); color: white; padding: 12px 16px; border-radius: 8px;">
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的法律问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(f"""
        <div style="background: var(--color-primary); color: white; padding: 12px 16px; border-radius: 8px;">
            {prompt}
        </div>
        """, unsafe_allow_html=True)
    
    # 如果是第一条消息，用问题作为对话标题
    if len(st.session_state.messages) == 1:
        title = prompt[:20] + ("..." if len(prompt) > 20 else "")
        st.session_state.conv_title = title
        conn = sqlite3.connect(SQLITE_DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, st.session_state.conversation_id))
        conn.commit()
        conn.close()
    
    # 助手回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            full_response = ""
            confidence_score = 1.0
            
            state_input = {
                "session_id": st.session_state.conversation_id,
                "user_query": prompt,
                "has_image": False
            }
            
            for chunk in legal_brain.stream(state_input):
                if "lawyer" in chunk:
                    full_response = chunk["lawyer"]["draft_opinion"]
                elif "common" in chunk:
                    full_response = chunk["common"]["draft_opinion"]
                elif "greeting" in chunk:
                    full_response = chunk["greeting"]["draft_opinion"]
                if "confidence_score" in chunk:
                    confidence_score = chunk["confidence_score"]
            
            st.markdown(full_response)
            
            # 检查是否是检索失败的回复
            is_retrieval_failure = "当前知识库中没有找到与您问题直接相关的法律依据" in full_response
            
            if is_retrieval_failure:
                st.error("⚠️ **检索失败提示**\n\n系统在当前知识库中未能找到与您问题直接相关的法律依据。")
                st.info("💡 **建议操作：**\n1. 上传相关法律文档到知识库（如《著作权法》《专利法》等）\n2. 尝试使用不同的问题表述\n3. 咨询专业律师获取权威意见")
            elif confidence_score < config.CONFIDENCE_THRESHOLD:
                st.warning(f"""
⚠️ **置信度较低 ({confidence_score:.2f})**

当前回答可能不够准确或完整。建议您：
1. 补充相关法律条文或案例信息
2. 上传相关法律文档到知识库
3. 参考以下方向进一步检索：
   - 相关法律法规原文
   - 最高人民法院指导案例
   - 地方性法规和司法解释
                """)
            
            # 反馈按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 有用", key=f"thumbs_up_{len(st.session_state.messages)}"):
                    save_chat(st.session_state.conversation_id, prompt, full_response, 1)
                    st.success("感谢反馈")
            with col2:
                if st.button("👎 无用", key=f"thumbs_down_{len(st.session_state.messages)}"):
                    save_chat(st.session_state.conversation_id, prompt, full_response, 0)
                    st.success("已记录")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat(st.session_state.conversation_id, prompt, full_response)
