import re
import json
import time
import requests
import uuid
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import config
from step04_test_retrieval_search_bm25_rerank import execute_legal_search
from memory_manager import memory_manager


# ==========================================
# 消息标准化工具（Harness工程）
# ==========================================
def normalize_messages(messages: list) -> list:
    """
    清洗消息列表以满足 OpenAI API 的三条硬性约束：
    1. 去除内部元数据字段（仅保留标准字段）
    2. 确保每个 tool_calls 都有匹配的 tool 结果（通过 tool_call_id 关联）
    3. 合并连续相同角色的消息（保证 user/assistant/tool 严格交替）
    
    参数: 
        messages: 原始消息列表 
    
    返回: 
        标准化后的消息列表 
    """
    # ---------- 1. 标准化：只保留 API 允许的字段 ----------
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        # 处理 content（可以是字符串或列表，但 OpenAI 一般用字符串）
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            # 过滤掉以 "_" 开头的内部字段
            clean["content"] = [
                {k: v for k, v in block.items() if not k.startswith("_")}
                for block in msg["content"] if isinstance(block, dict)
            ]
        else:
            clean["content"] = msg.get("content", "")
        # 保留 tool_calls 字段（如果存在）
        if "tool_calls" in msg:
            clean["tool_calls"] = msg["tool_calls"]
        cleaned.append(clean)

    # ---------- 2. 补全缺失的 tool 结果 ----------
    # 收集已有的 tool 消息的 tool_call_id
    existing_tool_ids = set()
    for msg in cleaned:
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            existing_tool_ids.add(msg["tool_call_id"])

    # 检查每个 assistant 消息中的 tool_calls，如果对应的 tool 结果缺失，则插入占位符
    new_entries = []
    for msg in cleaned:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            tc_id = tc.get("id")
            if tc_id and tc_id not in existing_tool_ids:
                # 插入一个占位的 tool 结果
                new_entries.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "(cancelled - result missing)"
                })
                existing_tool_ids.add(tc_id)  # 避免重复添加
    cleaned.extend(new_entries)

    # ---------- 3. 合并连续相同角色的消息 ----------
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        last = merged[-1]
        if msg["role"] == last["role"]:
            # 合并 content（简单拼接，适用于字符串）
            if isinstance(last.get("content"), str) and isinstance(msg.get("content"), str):
                last["content"] = last["content"] + "\n" + msg["content"]
        else:
            merged.append(msg)
    return merged


# ==========================================
# 错误恢复机制（Harness工程）
# ==========================================
MAX_RECOVERY_ATTEMPTS = 3


def is_prompt_too_long_error(error_body: str) -> bool:
    """判断是否为上下文过长错误"""
    return any(kw in error_body for kw in [
        "prompt too long", "context_length_exceeded",
        "maximum context length", "token limit"
    ])


def backoff_delay(attempt: int, base: float = 2.0, max_delay: float = 30.0) -> float:
    """计算指数退避延迟时间"""
    delay = min(base * (2 ** attempt), max_delay)
    return delay


def compact_history(messages: list, state: dict) -> list:
    """
    压缩对话历史以节省上下文空间
    保留最近的2轮对话，其余压缩为摘要
    """
    if len(messages) <= 4:
        return messages
    
    # 保留系统消息和最近4条消息
    system_msgs = [m for m in messages if m.get("role") == "system"]
    recent_msgs = messages[-4:]
    
    # 压缩中间部分为摘要
    middle_msgs = messages[len(system_msgs):-4]
    if middle_msgs:
        summary_text = "\n".join([f"{m['role']}: {m.get('content', '')[:100]}" for m in middle_msgs])
        summary_msg = {
            "role": "system",
            "content": f"【历史对话摘要】{summary_text[:500]}"
        }
        return system_msgs + [summary_msg] + recent_msgs
    
    return system_msgs + recent_msgs


class LegalCaseState(TypedDict):
    session_id: str
    user_query: str
    has_image: bool
    image_path: str | None
    question_type: str
    search_keywords: str
    retrieved_evidence: str
    draft_opinion: str
    review_feedback: str
    loop_count: int
    retrieval_retry_count: int
    is_compliant: str
    debate_round: int
    alternative_drafts: list
    selected_draft_index: int
    precedents: str


def call_llm(messages: list, model="zhipuai", temperature=None, max_retries=3, system_prompt=None):
    """调用大模型API（智谱AI或DeepSeek），带错误恢复机制"""
    if model == "zhipuai":
        api_key = config.ZHIPUAI_API_KEY
        base_url = config.ZHIPUAI_BASE_URL
        model_name = config.ZHIPUAI_MODEL
        temp = temperature if temperature is not None else config.ZHIPUAI_TEMP
    else:
        api_key = config.DEEPSEEK_API_KEY
        base_url = config.DEEPSEEK_BASE_URL
        model_name = config.DEEPSEEK_MODEL
        temp = temperature if temperature is not None else config.DEEPSEEK_TEMP

    if not api_key:
        if model == "deepseek":
            print("⚠️ DeepSeek API密钥未设置，回退到智谱AI")
            return call_llm(messages, model="zhipuai", temperature=temperature)
        raise ValueError(f"{model} API密钥未设置，请在环境变量中配置")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 标准化消息列表
    normalized_messages = normalize_messages(messages)
    
    # 构建API消息列表（添加系统提示词）
    if system_prompt:
        api_messages = [{"role": "system", "content": system_prompt}] + normalized_messages
    else:
        api_messages = normalized_messages

    data = {
        "model": model_name,
        "messages": api_messages,
        "temperature": temp,
        "max_tokens": 8000
    }

    # -- 尝试API调用，带错误恢复 --
    response = None
    for attempt in range(MAX_RECOVERY_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_body = response.text.lower()
                
                # 策略1: prompt too long -> 压缩后重试
                if is_prompt_too_long_error(error_body):
                    print(f"[错误恢复] 上下文过长，执行压缩... (尝试 {attempt + 1})")
                    compressed = compact_history(api_messages, {})
                    data["messages"] = compressed
                    continue  # 压缩后直接重试，不消耗重试预算
                
                # 其他API错误
                raise Exception(f"API错误: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError as e:
            error_body = str(e).lower()
            
            # 策略2: prompt too long -> 压缩后重试
            if is_prompt_too_long_error(error_body):
                print(f"[错误恢复] 上下文过长，执行压缩... (尝试 {attempt + 1})")
                compressed = compact_history(api_messages, {})
                data["messages"] = compressed
                continue
            
            # 策略3: 临时性网络/API错误 -> 指数退避重试
            if attempt < MAX_RECOVERY_ATTEMPTS:
                delay = backoff_delay(attempt)
                print(f"[错误恢复] API连接错误: {e}。"
                      f"等待 {delay:.1f}秒后重试 (尝试 {attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                time.sleep(delay)
                continue
            
            # 所有重试耗尽
            print(f"[错误] API调用失败，已重试 {MAX_RECOVERY_ATTEMPTS} 次: {e}")
            # 尝试回退到另一个模型
            if model == "zhipuai" and config.DEEPSEEK_API_KEY:
                print("🔄 回退到 DeepSeek API...")
                return call_llm(messages, model="deepseek", temperature=temperature)
            elif model == "deepseek" and config.ZHIPUAI_API_KEY:
                print("🔄 回退到智谱AI API...")
                return call_llm(messages, model="zhipuai", temperature=temperature)
            raise Exception(f"无法连接到 {model} API，请检查网络连接")
            
        except requests.exceptions.Timeout as e:
            # 策略3: 超时错误 -> 指数退避重试
            if attempt < MAX_RECOVERY_ATTEMPTS:
                delay = backoff_delay(attempt)
                print(f"[错误恢复] API请求超时: {e}。"
                      f"等待 {delay:.1f}秒后重试 (尝试 {attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                time.sleep(delay)
                continue
            
            print(f"[错误] API请求超时，已重试 {MAX_RECOVERY_ATTEMPTS} 次: {e}")
            raise Exception(f"{model} API 请求超时")
            
        except (ConnectionError, TimeoutError, OSError) as e:
            # 策略3: 网络层错误 -> 指数退避重试
            if attempt < MAX_RECOVERY_ATTEMPTS:
                delay = backoff_delay(attempt)
                print(f"[错误恢复] 连接错误: {e}。"
                      f"等待 {delay:.1f}秒后重试 (尝试 {attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                time.sleep(delay)
                continue
            
            print(f"[错误] 连接失败，已重试 {MAX_RECOVERY_ATTEMPTS} 次: {e}")
            raise


def router_node(state: LegalCaseState):
    """路由Agent：决策问题类型"""
    query = state["user_query"].strip()
    has_img = state.get("has_image", False)
    table_keywords = ["表格", "金额", "赔偿金", "计算", "费用"]
    greetings = ["你好", "您好", "hello", "hi", "嗨", "早上好", "晚上好", "下午好", "谢谢", "感谢"]

    if has_img:
        qtype = "image"
    elif any(kw in query.lower() for kw in greetings) and len(query) < 20:
        qtype = "greeting"
    elif any(kw in query for kw in table_keywords):
        qtype = "table"
    elif any(kw in query for kw in ["什么是", "如何", "为什么", "怎么", "哪些"]):
        qtype = "common"
    else:
        qtype = "legal"

    print(f"🔀 [路由Agent] 问题类型: {qtype}")
    return {"question_type": qtype}


def greeting_node(state: LegalCaseState):
    """问候语直接回复，不调用大模型"""
    print("👋 [问候处理] 直接回复...")
    query = state["user_query"].lower().strip()
    
    if any(kw in query for kw in ["你好", "您好", "hello", "hi", "嗨"]):
        response = "您好！我是AI法律助手，可以帮您解答法律问题、分析案例、检索法条。请问有什么可以帮您的？"
    elif any(kw in query for kw in ["谢谢", "感谢"]):
        response = "不客气！如果还有其他法律问题，随时告诉我。"
    else:
        response = "您好！请问有什么法律问题需要我帮忙解答？"
    
    return {"draft_opinion": response, "is_compliant": "PASS"}


def common_knowledge_node(state: LegalCaseState):
    """常识问题直接回答"""
    print("🧠 [常识回答] 直接生成答案...")
    session_id = state.get("session_id", str(uuid.uuid4()))
    context = memory_manager.get_context_for_llm(session_id)
    
    messages = [
        {"role": "system", "content": "你是一个专业的法律助手。请用简洁、准确的语言回答用户的法律常识问题。回答要简明扼要，控制在200字以内。"}
    ]
    if context:
        messages.extend(context[-4:])
    messages.append({"role": "user", "content": state["user_query"]})
    
    answer = call_llm(messages, model="zhipuai")
    
    memory_manager.add_message(session_id, "user", state["user_query"])
    memory_manager.add_message(session_id, "assistant", answer)
    
    if memory_manager.get_message_count(session_id) > 8:
        memory_manager.compress_history(session_id, call_llm)
    
    return {"draft_opinion": answer, "is_compliant": "PASS"}


def table_query_node(state: LegalCaseState):
    """表格问题进入RAG流程"""
    print("📊 [表格查询] 进入RAG流程...")
    return query_expander(state)


def query_expander(state: LegalCaseState):
    """案件分析员：查询扩展"""
    print("🕵️‍♂️ [案件分析员] 转化检索词...")
    session_id = state.get("session_id", str(uuid.uuid4()))
    context = memory_manager.get_context_for_llm(session_id)
    
    messages = [
        {"role": "system", "content": "将用户口语转化为法律检索词，直接输出。"}
    ]
    
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": state["user_query"]})
    
    keywords = call_llm(messages, model="zhipuai").strip()
    return {"search_keywords": keywords, "loop_count": 0, "retrieval_retry_count": 0}


def legal_researcher(state: LegalCaseState):
    """法务检索员"""
    print("📚 [法务检索员] 检索中...")
    evidence = execute_legal_search(
        state["search_keywords"],
        retrieve_k=config.RETRIEVE_K,
        rerank_k=config.RERANK_K
    )
    return {"retrieved_evidence": evidence}


def quality_check_node(state: LegalCaseState):
    """检索质量校验节点"""
    query = state["user_query"]
    evidence = state["retrieved_evidence"]

    # 如果证据中明显没有有效内容
    if "未检索到" in evidence or "未找到" in evidence or len(evidence.strip()) < 100:
        score = 0.0
    else:
        # 更严格的打分 prompt：要求只输出一个数字
        messages = [
            {"role": "system", "content": "你是一个检索质量评估专家。请仅输出一个0到1之间的数字，表示法条证据与问题的相关性。不要输出任何其他文字，只输出数字。"},
            {"role": "user", "content": f"问题：{query}\n\n法条证据（摘要）：\n{evidence[:500]}\n\n相关性分数（只输出数字）："}
        ]
        try:
            resp = call_llm(messages, model="zhipuai", temperature=0)
            # 提取数字
            num_match = re.search(r'(\d+(?:\.\d+)?)', resp)
            score = float(num_match.group(1)) if num_match else 0.5
            # 确保分数在 0-1 范围内
            score = max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ 质量打分异常: {e}")
            score = 0.5

    print(f"🎯 [质量校验] 检索质量得分: {score:.2f}")

    if score >= config.RETRIEVAL_RELEVANCE_THRESHOLD:
        return {"need_retry": False}
    else:
        new_retry = state.get("retrieval_retry_count", 0) + 1
        if new_retry <= config.MAX_RETRIEVAL_RETRIES:
            print(f"⚠️ 质量不合格，重试 ({new_retry}/{config.MAX_RETRIEVAL_RETRIES})")
            # 使用 FALLBACK 参数扩大召回
            new_evidence = execute_legal_search(
                state["search_keywords"] + " 关键 法条",
                retrieve_k=config.FALLBACK_RETRIEVE_K,
                rerank_k=config.FALLBACK_RERANK_K
            )
            return {"retrieved_evidence": new_evidence,
                    "search_keywords": state["search_keywords"] + " 关键",
                    "retrieval_retry_count": new_retry,
                    "need_retry": True}
        else:
            print("❌ 多次检索失败，放弃")
            return {"retrieved_evidence": "未找到足够相关的法律依据。", "need_retry": False}


def precedent_retriever(state: LegalCaseState):
    """判例检索节点"""
    print("⚖️ [判例检索员] 查找类似案例...")
    messages = [
        {"role": "system", "content": "你是一个判例检索专家。根据用户问题，输出1-2个相关指导案例的简要描述（案号、裁判要点）。"},
        {"role": "user", "content": state["user_query"]}
    ]
    precedents = call_llm(messages, model="zhipuai")
    print(f"   判例结果: {precedents[:100]}...")
    return {"precedents": precedents}


def senior_lawyer(state: LegalCaseState):
    """主审律师（增加判例融入）"""
    print(f"👨‍⚖️ [主审律师] 撰写第{state['loop_count']+1}稿...")
    feedback = state.get("review_feedback", "")
    evidence = state["retrieved_evidence"]
    
    # 如果证据为空或太短，直接回复
    if "未找到" in evidence or "未检索到" in evidence or len(evidence) < 50:
        return {"draft_opinion": "抱歉，当前知识库中没有找到与您问题直接相关的法律依据。\n\n建议您：\n1. 上传相关法律文档到知识库（如《著作权法》《专利法》等）\n2. 提供更具体的案件细节\n3. 咨询专业律师获取权威意见\n\n⚠️ 请注意：AI 助手不会编造任何法律条文，以上建议仅供参考。", 
                "is_compliant": "PASS", "loop_count": state["loop_count"] + 1}
    
    system_prompt = f"""你是一位顶尖的中国执业律师。你必须严格依据下方提供的【法条证据】来回答问题。

【重要规则】：
1. 你的回答必须明确引用证据中的原文，格式如"根据[证据X]中的《XXX法》第X条规定：..."。
2. 如果证据不足以支持完整回答，请明确指出缺失的部分，不得自行编造法条。
3. 最终结论必须基于证据中的具体条款。
4. 不要编造任何法条或案例，如果证据中没有，请说明"根据现有资料无法确定"。
5. 如果证据中只有部分相关条款，请说明哪些信息缺失。

{("【打回意见】：" + state.get("review_feedback", "")) if state.get("review_feedback") else ""}
{("【参考判例】：" + state.get("precedents", "")) if state.get("precedents") else ""}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"问题：{state['user_query']}\n\n【法条证据】：\n{evidence}"}
    ]
    draft = call_llm(messages, model="zhipuai", temperature=0.1)
    return {"draft_opinion": draft, "loop_count": state["loop_count"] + 1}


def verify_citation(state: LegalCaseState):
    """验证律师草稿是否引用了证据中的法条"""
    draft = state["draft_opinion"]
    evidence = state["retrieved_evidence"]
    
    # 提取证据中的关键语句（如"第.*条"）
    citation_pattern = r'第[一二三四五六七八九十百千万0-9]+条'
    evidence_citations = set(re.findall(citation_pattern, evidence))
    draft_citations = set(re.findall(citation_pattern, draft))
    
    if evidence_citations and not draft_citations:
        return {"is_compliant": "FAIL", 
                "review_feedback": "答案中没有引用任何法条原文，请基于证据中的具体条款重新回答。"}
    return {}


# ==========================================
# 审查模式权限控制（Harness工程）
# ==========================================
class EvidenceGuard:
    """
    证据保护器：确保审查阶段只读访问，防止修改原始证据
    用于多Agent系统中的权限控制
    """
    
    def __init__(self, evidence: str):
        self._original_evidence = evidence
        self._hash = hash(evidence)
        self._access_log = []
    
    @property
    def evidence(self) -> str:
        """只读访问证据内容"""
        self._access_log.append({"action": "read", "timestamp": time.time()})
        return self._original_evidence
    
    def verify_integrity(self) -> bool:
        """验证证据是否被篡改"""
        return hash(self._original_evidence) == self._hash
    
    def get_access_log(self) -> list:
        """获取访问日志"""
        return self._access_log.copy()
    
    def create_readonly_prompt(self, draft: str) -> str:
        """
        创建只读模式的审查prompt
        明确指示审查器只能评估，不能修改证据
        """
        return f"""【只读审查模式 - Plan Only】
你是严格的合规审查官。你只能阅读和评估以下内容，不得修改任何原始证据。

【审查权限】：只读（Read-Only）
【审查对象】：律师草稿
【参考依据】：原始法条证据（不可修改）

请从以下维度评估律师草稿：
1. 事实匹配性（是否与法条原文一致）
2. 法律条款适用性（引用的法条是否准确）
3. 推理链条完整性（逻辑是否连贯）
4. 逻辑无矛盾（是否存在自相矛盾）
5. 格式规范性（是否符合法律文书规范）
6. 风险点遗漏（是否遗漏重要法律风险）

⚠️ 重要约束：
- 你只能输出评估意见，不得修改或补充原始证据
- 如果发现证据不足，只能指出缺失，不得自行编造
- 所有评估必须基于下方提供的【法条原文】

输出格式为JSON：
{{
  "pass": true/false,
  "confidence": 0.0~1.0之间的数字,
  "feedback": "具体意见",
  "missing_citations": ["缺失的引用1", "缺失的引用2"],
  "logic_errors": ["逻辑矛盾描述"],
  "format_issues": ["格式问题描述"]
}}

【法条原文】（只读，不可修改）：
{self._original_evidence}

【律师草稿】（待审查）：
{draft}"""


def compliance_reviewer_structured(state: LegalCaseState):
    """结构化合规审查官（返回JSON + 置信度评分，启用只读模式）"""
    print("⚖️ [合规审查官] 结构化审查中（只读模式）...")
    
    # 创建证据保护器（只读访问）
    evidence_guard = EvidenceGuard(state["retrieved_evidence"])
    
    # 验证证据完整性
    if not evidence_guard.verify_integrity():
        print("⚠️ [安全警告] 证据完整性校验失败！")
        return {"is_compliant": "FAIL", 
                "review_feedback": "系统错误：证据数据异常，请重新检索。"}
    
    # 使用只读模式prompt进行审查
    readonly_prompt = evidence_guard.create_readonly_prompt(state["draft_opinion"])
    
    messages = [
        {"role": "user", "content": readonly_prompt}
    ]
    raw = call_llm(messages, model="deepseek")

    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_match:
        print(f"🧠 审查官思考: {think_match.group(1)[:150]}...")

    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    try:
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"pass": False, "confidence": 0.3, "feedback": clean}
    except:
        result = {"pass": False, "confidence": 0.3, "feedback": clean}

    confidence = result.get("confidence", 0.5)
    print(f"📊 [审查官] 置信度: {confidence:.2f}")

    if result.get("pass", False) and confidence >= 0.7:
        return {"is_compliant": "PASS", "review_feedback": "", "confidence_score": confidence}
    else:
        feedback = result.get("feedback", "")
        missing = result.get("missing_citations", [])
        logic = result.get("logic_errors", [])
        if missing:
            feedback += f"\n缺失引用: {missing}"
        if logic:
            feedback += f"\n逻辑错误: {logic}"
        return {"is_compliant": "FAIL", "review_feedback": feedback, "confidence_score": confidence}


def generate_alternatives(state: LegalCaseState):
    """生成备选草稿（简化树状搜索）"""
    print("🌲 [生成备选] 产生多个修改方案...")
    prompts = [
        "请严格依据法条原文，逐条对应回答，不要添加任何推测。",
        "请侧重逻辑推理，先分析法律适用条件，再得出结论。",
        "请侧重风险提示，指出用户可能忽视的法律风险。"
    ]
    alternatives = []
    for idx, style_prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": f"你是一位顶尖律师。{style_prompt}"},
            {"role": "user", "content": f"问题：{state['user_query']}\n证据：{state['retrieved_evidence']}"}
        ]
        draft = call_llm(messages, model="zhipuai")
        alternatives.append(draft)
        print(f"   备选方案{idx+1}生成完成")
    return {"alternative_drafts": alternatives, "debate_round": state.get("debate_round", 0) + 1}


def select_best_draft(state: LegalCaseState):
    """使用审查官再次评估备选草稿，选择最优"""
    print("🤖 [方案选择] 评估备选草稿...")
    best_idx = 0
    best_score = -1
    for idx, draft in enumerate(state["alternative_drafts"]):
        messages = [
            {"role": "system", "content": "请对以下法律草稿的质量进行评分（0-100分），只输出数字。"},
            {"role": "user", "content": f"问题：{state['user_query']}\n证据：{state['retrieved_evidence']}\n草稿：{draft}\n得分："}
        ]
        try:
            score = float(call_llm(messages, model="zhipuai").strip())
        except:
            score = 0
        if score > best_score:
            best_score = score
            best_idx = idx
    selected = state["alternative_drafts"][best_idx]
    print(f"   选中方案{best_idx+1}，得分{best_score}")
    return {"draft_opinion": selected, "selected_draft_index": best_idx, "loop_count": state["loop_count"] + 1}


def should_continue(state: LegalCaseState):
    """路由逻辑（支持打回检索员、生成备选等）"""
    if state["is_compliant"] == "PASS" or state["loop_count"] >= config.MAX_REWRITE_LOOPS:
        return "end"
    if "检索" in state.get("review_feedback", "") or "证据" in state.get("review_feedback", ""):
        return "retry_retrieval"
    if state.get("alternative_drafts") and len(state["alternative_drafts"]) > 0:
        print("⚠️ 多次修改仍未通过，标记为低置信度答案")
        return "end"
    return "generate_alternatives"


workflow = StateGraph(LegalCaseState)

workflow.add_node("router", router_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("common", common_knowledge_node)
workflow.add_node("table", table_query_node)
workflow.add_node("expander", query_expander)
workflow.add_node("researcher", legal_researcher)
workflow.add_node("quality", quality_check_node)
workflow.add_node("precedent", precedent_retriever)
workflow.add_node("lawyer", senior_lawyer)
workflow.add_node("verify_citation", verify_citation)
workflow.add_node("reviewer", compliance_reviewer_structured)
workflow.add_node("generate_alternatives", generate_alternatives)
workflow.add_node("select_best", select_best_draft)

workflow.set_entry_point("router")


def route_after_router(state: LegalCaseState) -> Literal["greeting", "common", "table", "expander"]:
    qtype = state.get("question_type", "legal")
    if qtype == "greeting":
        return "greeting"
    elif qtype == "common":
        return "common"
    elif qtype == "table":
        return "table"
    else:
        return "expander"


workflow.add_conditional_edges("router", route_after_router, {
    "greeting": "greeting",
    "common": "common",
    "table": "table",
    "expander": "expander"
})

workflow.add_edge("greeting", END)
workflow.add_edge("common", END)
workflow.add_edge("table", "expander")
workflow.add_edge("expander", "researcher")
workflow.add_edge("researcher", "quality")


def after_quality(state: LegalCaseState) -> Literal["expander", "precedent"]:
    if state.get("need_retry", False):
        return "expander"
    else:
        return "precedent"


workflow.add_conditional_edges("quality", after_quality, {"expander": "expander", "precedent": "precedent"})
workflow.add_edge("precedent", "lawyer")
workflow.add_edge("lawyer", "verify_citation")
workflow.add_edge("verify_citation", "reviewer")


def after_reviewer(state: LegalCaseState) -> Literal["end", "expander", "generate_alternatives"]:
    decision = should_continue(state)
    if decision == "end":
        return "end"
    elif decision == "retry_retrieval":
        return "expander"
    else:
        return "generate_alternatives"


workflow.add_conditional_edges("reviewer", after_reviewer, {
    "end": END,
    "expander": "expander",
    "generate_alternatives": "generate_alternatives"
})

workflow.add_edge("generate_alternatives", "select_best")
workflow.add_edge("select_best", "reviewer")

legal_brain = workflow.compile()


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    test_query = "别人偷偷抄了我的包装盒设计拿去卖，我要去法院告他，最多能拿多少赔偿金？"
    final_state = legal_brain.invoke({"session_id": session_id, "user_query": test_query, "has_image": False})
    print("\n最终意见书：", final_state["draft_opinion"])
