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
# 法律工具集（Harness工程：规范化 + 重试）
# ==========================================
class ToolResult:
    """工具调用结果的规范化封装"""
    
    def __init__(self, tool_name: str, success: bool, data: any, error: str = None):
        self.tool_name = tool_name
        self.success = success
        self.data = data
        self.error = error
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error
        }
    
    @classmethod
    def from_error(cls, tool_name: str, error: str) -> "ToolResult":
        return cls(tool_name=tool_name, success=False, data=None, error=error)


def tool_retry(max_retries: int = 2):
    """工具调用重试装饰器（Harness工程）"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return ToolResult(tool_name=tool_name, success=True, data=result)
                except Exception as e:
                    if attempt < max_retries:
                        delay = 2 ** attempt
                        print(f"[工具重试] {tool_name} 失败 (尝试 {attempt+1}/{max_retries}): {str(e)[:80]}")
                        time.sleep(delay)
                    else:
                        print(f"[工具失败] {tool_name} 已耗尽重试: {e}")
                        return ToolResult.from_error(tool_name, str(e))
        return wrapper
    return decorator


@tool_retry(max_retries=2)
def search_legal_articles(query: str, top_k: int = None) -> str:
    """
    工具1：法条检索工具
    执行混合检索（BM25 + 向量 + RRF + CrossEncoder重排）
    """
    k = top_k or config.RETRIEVE_K
    evidence = execute_legal_search(query, retrieve_k=k, rerank_k=config.RERANK_K)
    if "未检索到" in evidence or "未找到" in evidence or len(evidence.strip()) < 100:
        raise ValueError(f"检索结果为空: {query}")
    return evidence


@tool_retry(max_retries=2)
def search_precedents(query: str) -> str:
    """
    工具2：判例检索工具
    基于 RAG 检索相关案例/裁判文书，返回真实依据
    在检索词后追加"案例 裁判 判决"引导词，优先匹配案例类文档
    """
    precedent_query = query + " 案例 裁判 判决 指导案例"
    evidence = execute_legal_search(
        precedent_query,
        retrieve_k=config.FALLBACK_RETRIEVE_K,
        rerank_k=config.FALLBACK_RERANK_K
    )
    if "未检索到" in evidence or "未找到" in evidence or len(evidence.strip()) < 100:
        raise ValueError(f"判例检索结果为空: {query}")
    return evidence


@tool_retry(max_retries=2)
def calculate_compensation(case_type: str, evidence: str) -> str:
    """
    工具3：赔偿计算工具
    根据案件类型和法条证据，计算可能的赔偿范围
    """
    messages = [
        {"role": "system", "content": """你是一个赔偿计算专家。请根据提供的法条证据，分析赔偿计算方式和可能的金额范围。
输出格式：
- 计算依据：引用具体法条
- 计算方式：说明计算方法
- 赔偿范围：给出金额区间（如无法确定具体金额，说明原因）"""},
        {"role": "user", "content": f"案件类型：{case_type}\n\n法条证据：\n{evidence}"}
    ]
    result = call_llm(messages, model="zhipuai")
    return result


@tool_retry(max_retries=2)
def verify_citations(draft: str, evidence: str) -> dict:
    """
    工具4：引用验证工具
    验证法律意见是否正确引用了法条原文
    """
    citation_pattern = r'第[一二三四五六七八九十百千万0-9]+条'
    evidence_citations = set(re.findall(citation_pattern, evidence))
    draft_citations = set(re.findall(citation_pattern, draft))
    
    missing = evidence_citations - draft_citations
    extra = draft_citations - evidence_citations
    
    return {
        "evidence_citations": list(evidence_citations),
        "draft_citations": list(draft_citations),
        "missing_citations": list(missing),
        "extra_citations": list(extra),
        "has_citation": len(draft_citations) > 0
    }


# ==========================================
# 消息标准化工具（Harness工程）
# ==========================================
def normalize_messages(messages: list) -> list:
    """
    清洗消息列表以满足 OpenAI API 的三条硬性约束：
    1. 去除内部元数据字段（仅保留标准字段）
    2. 确保每个 tool_calls 都有匹配的 tool 结果（通过 tool_call_id 关联）
    3. 合并连续相同角色的消息（保证 user/assistant/tool 严格交替）
    """
    # ---------- 1. 标准化：只保留 API 允许的字段 ----------
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            clean["content"] = [
                {k: v for k, v in block.items() if not k.startswith("_")}
                for block in msg["content"] if isinstance(block, dict)
            ]
        else:
            clean["content"] = msg.get("content", "")
        if "tool_calls" in msg:
            clean["tool_calls"] = msg["tool_calls"]
        cleaned.append(clean)

    # ---------- 2. 补全缺失的 tool 结果 ----------
    existing_tool_ids = set()
    for msg in cleaned:
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            existing_tool_ids.add(msg["tool_call_id"])

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
                new_entries.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "(cancelled - result missing)"
                })
                existing_tool_ids.add(tc_id)
    cleaned.extend(new_entries)

    # ---------- 3. 合并连续相同角色的消息 ----------
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        last = merged[-1]
        if msg["role"] == last["role"]:
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
    return min(base * (2 ** attempt), max_delay)


def compact_history(messages: list, state: dict) -> list:
    """压缩对话历史以节省上下文空间"""
    if len(messages) <= 4:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    recent_msgs = messages[-4:]
    middle_msgs = messages[len(system_msgs):-4]
    if middle_msgs:
        summary_text = "\n".join([f"{m['role']}: {m.get('content', '')[:100]}" for m in middle_msgs])
        summary_msg = {
            "role": "system",
            "content": f"【历史对话摘要】{summary_text[:500]}"
        }
        return system_msgs + [summary_msg] + recent_msgs
    return system_msgs + recent_msgs


# ==========================================
# 状态定义
# ==========================================
class LegalCaseState(TypedDict):
    session_id: str
    user_query: str
    has_image: bool
    image_path: str | None
    question_type: str
    search_keywords: str
    retrieved_evidence: str
    precedents: str
    compensation_info: str
    draft_opinion: str
    review_feedback: str
    loop_count: int
    retrieval_retry_count: int
    is_compliant: str
    confidence_score: float
    alternative_drafts: list
    selected_draft_index: int
    citation_check: dict


# ==========================================
# LLM 调用（集成错误恢复）
# ==========================================
def call_llm(messages: list, model="zhipuai", temperature=None, system_prompt=None):
    """调用大模型API，带错误恢复机制"""
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
        raise ValueError(f"{model} API密钥未设置")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    normalized_messages = normalize_messages(messages)
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
                if is_prompt_too_long_error(error_body):
                    print(f"[错误恢复] 上下文过长，执行压缩... (尝试 {attempt + 1})")
                    compressed = compact_history(api_messages, {})
                    data["messages"] = compressed
                    continue
                raise Exception(f"API错误: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError as e:
            error_body = str(e).lower()
            if is_prompt_too_long_error(error_body):
                print(f"[错误恢复] 上下文过长，执行压缩... (尝试 {attempt + 1})")
                compressed = compact_history(api_messages, {})
                data["messages"] = compressed
                continue
            if attempt < MAX_RECOVERY_ATTEMPTS:
                delay = backoff_delay(attempt)
                print(f"[错误恢复] 连接错误，等待 {delay:.1f}秒后重试 ({attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                time.sleep(delay)
                continue
            print(f"[错误] API调用失败，已重试 {MAX_RECOVERY_ATTEMPTS} 次: {e}")
            if model == "zhipuai" and config.DEEPSEEK_API_KEY:
                print("🔄 回退到 DeepSeek API...")
                return call_llm(messages, model="deepseek", temperature=temperature)
            elif model == "deepseek" and config.ZHIPUAI_API_KEY:
                print("🔄 回退到智谱AI API...")
                return call_llm(messages, model="zhipuai", temperature=temperature)
            raise Exception(f"无法连接到 {model} API")

        except requests.exceptions.Timeout as e:
            if attempt < MAX_RECOVERY_ATTEMPTS:
                delay = backoff_delay(attempt)
                print(f"[错误恢复] 请求超时，等待 {delay:.1f}秒后重试 ({attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                time.sleep(delay)
                continue
            raise Exception(f"{model} API 请求超时")

        except (ConnectionError, TimeoutError, OSError) as e:
            if attempt < MAX_RECOVERY_ATTEMPTS:
                delay = backoff_delay(attempt)
                print(f"[错误恢复] 连接错误，等待 {delay:.1f}秒后重试 ({attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                time.sleep(delay)
                continue
            raise


# ==========================================
# Agent 1: Router（路由 Agent）
# ==========================================
def router_agent(state: LegalCaseState):
    """
    路由 Agent：基于规则识别问题类型，动态分流
    - greeting: 问候语，直接回复
    - common: 法律常识，轻量回答
    - legal: 法律专业问题，进入完整 RAG 流程
    """
    query = state["user_query"].strip()
    has_img = state.get("has_image", False)
    greetings = ["你好", "您好", "hello", "hi", "嗨", "早上好", "晚上好", "下午好", "谢谢", "感谢"]

    if has_img:
        qtype = "legal"
    elif any(kw in query.lower() for kw in greetings) and len(query) < 20:
        qtype = "greeting"
    elif any(kw in query for kw in ["什么是", "如何", "为什么", "怎么", "哪些"]) and len(query) < 50:
        qtype = "common"
    else:
        qtype = "legal"

    print(f"🔀 [Router] 问题类型: {qtype}")
    return {"question_type": qtype}


def greeting_handler(state: LegalCaseState):
    """问候语处理，不调用大模型"""
    print("👋 [问候处理] 直接回复...")
    query = state["user_query"].lower().strip()
    
    if any(kw in query for kw in ["你好", "您好", "hello", "hi", "嗨"]):
        response = "您好！我是AI法律助手，可以帮您解答法律问题、分析案例、检索法条。请问有什么可以帮您的？"
    elif any(kw in query for kw in ["谢谢", "感谢"]):
        response = "不客气！如果还有其他法律问题，随时告诉我。"
    else:
        response = "您好！请问有什么法律问题需要我帮忙解答？"
    
    return {"draft_opinion": response, "is_compliant": "PASS"}


def common_handler(state: LegalCaseState):
    """常识问题处理，轻量级回答"""
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


# ==========================================
# Agent 2: Retriever（检索 Agent）
# ==========================================
def query_expander(state: LegalCaseState):
    """查询扩展：将口语转化为专业法律检索词"""
    print("🕵️ [查询扩展] 转化检索词...")
    session_id = state.get("session_id", str(uuid.uuid4()))
    context = memory_manager.get_context_for_llm(session_id)
    
    messages = [
        {"role": "system", "content": "将用户口语转化为法律检索词，直接输出关键词，用空格分隔。"}
    ]
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": state["user_query"]})
    
    keywords = call_llm(messages, model="zhipuai").strip()
    return {"search_keywords": keywords, "loop_count": 0, "retrieval_retry_count": 0}


def retriever_agent(state: LegalCaseState):
    """
    检索 Agent：调用法条检索工具 + 判例检索工具
    工具结果规范化封装，支持异常重试
    """
    print("📚 [Retriever] 执行检索...")
    keywords = state["search_keywords"]
    
    # 调用工具1：法条检索
    article_result = search_legal_articles(keywords)
    if article_result.success:
        evidence = article_result.data
        print(f"   法条检索成功，内容长度: {len(evidence)}")
    else:
        evidence = f"法条检索失败: {article_result.error}"
        print(f"   法条检索失败: {article_result.error}")
    
    # 调用工具2：判例检索
    precedent_result = search_precedents(keywords)
    if precedent_result.success:
        precedents = precedent_result.data
        print(f"   判例检索成功")
    else:
        precedents = f"判例检索失败: {precedent_result.error}"
        print(f"   判例检索失败: {precedent_result.error}")
    
    return {
        "retrieved_evidence": evidence,
        "precedents": precedents
    }


def quality_check_node(state: LegalCaseState):
    """检索质量校验"""
    query = state["user_query"]
    evidence = state["retrieved_evidence"]

    if "未检索到" in evidence or "未找到" in evidence or "失败" in evidence or len(evidence.strip()) < 100:
        score = 0.0
    else:
        messages = [
            {"role": "system", "content": "你是一个检索质量评估专家。请仅输出一个0到1之间的数字，表示法条证据与问题的相关性。不要输出任何其他文字，只输出数字。"},
            {"role": "user", "content": f"问题：{query}\n\n法条证据（摘要）：\n{evidence[:500]}\n\n相关性分数（只输出数字）："}
        ]
        try:
            resp = call_llm(messages, model="zhipuai", temperature=0)
            num_match = re.search(r'(\d+(?:\.\d+)?)', resp)
            score = float(num_match.group(1)) if num_match else 0.5
            score = max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ 质量打分异常: {e}")
            score = 0.5

    print(f"🎯 [质量校验] 得分: {score:.2f}")

    if score >= config.RETRIEVAL_RELEVANCE_THRESHOLD:
        return {"need_retry": False}
    else:
        new_retry = state.get("retrieval_retry_count", 0) + 1
        if new_retry <= config.MAX_RETRIEVAL_RETRIES:
            print(f"⚠️ 质量不足，扩大召回 ({new_retry}/{config.MAX_RETRIEVAL_RETRIES})")
            new_evidence = execute_legal_search(
                state["search_keywords"] + " 关键 法条",
                retrieve_k=config.FALLBACK_RETRIEVE_K,
                rerank_k=config.FALLBACK_RERANK_K
            )
            return {
                "retrieved_evidence": new_evidence,
                "search_keywords": state["search_keywords"] + " 关键",
                "retrieval_retry_count": new_retry,
                "need_retry": True
            }
        else:
            print("❌ 多次检索失败，使用兜底")
            return {"retrieved_evidence": "未找到足够相关的法律依据。", "need_retry": False}


# ==========================================
# Agent 3: Generator（生成 Agent）
# ==========================================
def generator_agent(state: LegalCaseState):
    """
    生成 Agent：依据证据生成法律意见
    内部调用工具3（赔偿计算）和工具4（引用验证）
    """
    print(f"👨‍⚖️ [Generator] 撰写第{state['loop_count']+1}稿...")
    feedback = state.get("review_feedback", "")
    evidence = state["retrieved_evidence"]
    
    # 证据不足时直接回复
    if "未找到" in evidence or "未检索到" in evidence or "失败" in evidence or len(evidence) < 50:
        return {
            "draft_opinion": "抱歉，当前知识库中没有找到与您问题直接相关的法律依据。\n\n建议您：\n1. 上传相关法律文档到知识库\n2. 提供更具体的案件细节\n3. 咨询专业律师获取权威意见\n\n⚠️ AI 助手不会编造任何法律条文，以上建议仅供参考。",
            "is_compliant": "PASS",
            "loop_count": state["loop_count"] + 1
        }
    
    # 判断是否涉及赔偿计算，如果是则调用赔偿计算工具
    compensation_info = ""
    query_lower = state["user_query"].lower()
    if any(kw in query_lower for kw in ["赔偿", "金额", "多少钱", "补偿", "罚款", "罚金"]):
        print("   检测到赔偿相关，调用赔偿计算工具...")
        comp_result = calculate_compensation("知识产权侵权", evidence)
        if comp_result.success:
            compensation_info = f"\n\n【赔偿计算参考】\n{comp_result.data}"
            print("   赔偿计算完成")
        else:
            print(f"   赔偿计算失败: {comp_result.error}")
    
    # 构建生成 prompt
    system_prompt = f"""你是一位顶尖的中国执业律师。你必须严格依据下方提供的【法条证据】来回答问题。

【重要规则】：
1. 你的回答必须明确引用证据中的原文，格式如"根据[证据X]中的《XXX法》第X条规定：..."。
2. 如果证据不足以支持完整回答，请明确指出缺失的部分，不得自行编造法条。
3. 最终结论必须基于证据中的具体条款。
4. 不要编造任何法条或案例，如果证据中没有，请说明"根据现有资料无法确定"。
5. 如果证据中只有部分相关条款，请说明哪些信息缺失。

{("【打回意见】：" + feedback) if feedback else ""}
{("【参考判例】：" + state.get("precedents", "")) if state.get("precedents") else ""}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"问题：{state['user_query']}\n\n【法条证据】：\n{evidence}"}
    ]
    draft = call_llm(messages, model="zhipuai", temperature=0.1)
    
    # 追加赔偿计算结果
    if compensation_info:
        draft += compensation_info
    
    # 调用工具4：引用验证
    citation_result = verify_citations(draft, evidence)
    print(f"   引用验证: 证据引用{len(citation_result['evidence_citations'])}条, 草稿引用{len(citation_result['draft_citations'])}条")
    
    return {
        "draft_opinion": draft,
        "citation_check": citation_result.to_dict() if hasattr(citation_result, 'to_dict') else citation_result,
        "loop_count": state["loop_count"] + 1
    }


# ==========================================
# Agent 4: Reviewer（审查 Agent）
# ==========================================
class EvidenceGuard:
    """
    证据保护器：审查阶段只读访问，防止修改原始证据
    Harness工程：权限控制
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
        """创建只读模式的审查prompt（Plan Only）"""
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


def reviewer_agent(state: LegalCaseState):
    """
    审查 Agent：结构化合规审查，启用只读模式
    从6个维度校验，未通过时驱动"检索增强/答案重写"
    """
    print("⚖️ [Reviewer] 结构化审查中（只读模式）...")
    
    # 创建证据保护器（只读访问）
    evidence_guard = EvidenceGuard(state["retrieved_evidence"])
    
    # 验证证据完整性
    if not evidence_guard.verify_integrity():
        print("⚠️ [安全警告] 证据完整性校验失败！")
        return {
            "is_compliant": "FAIL",
            "review_feedback": "系统错误：证据数据异常，请重新检索。",
            "confidence_score": 0.0
        }
    
    # 使用只读模式prompt进行审查
    readonly_prompt = evidence_guard.create_readonly_prompt(state["draft_opinion"])
    
    messages = [{"role": "user", "content": readonly_prompt}]
    raw = call_llm(messages, model="deepseek")

    # 清理思考标签
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
    print(f"📊 [Reviewer] 置信度: {confidence:.2f}")

    if result.get("pass", False) and confidence >= config.CONFIDENCE_THRESHOLD:
        return {
            "is_compliant": "PASS",
            "review_feedback": "",
            "confidence_score": confidence
        }
    else:
        feedback = result.get("feedback", "")
        missing = result.get("missing_citations", [])
        logic = result.get("logic_errors", [])
        if missing:
            feedback += f"\n缺失引用: {missing}"
        if logic:
            feedback += f"\n逻辑错误: {logic}"
        return {
            "is_compliant": "FAIL",
            "review_feedback": feedback,
            "confidence_score": confidence
        }


def generate_alternatives(state: LegalCaseState):
    """生成备选方案（审查多次未通过时）"""
    print("🌲 [备选生成] 产生多个修改方案...")
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
    return {"alternative_drafts": alternatives}


def select_best_draft(state: LegalCaseState):
    """选择最优备选方案"""
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
    return {"draft_opinion": selected, "selected_draft_index": best_idx}


# ==========================================
# LangGraph 工作流（4 Agent 固定架构）
# ==========================================
workflow = StateGraph(LegalCaseState)

# 注册节点
workflow.add_node("router", router_agent)
workflow.add_node("greeting", greeting_handler)
workflow.add_node("common", common_handler)
workflow.add_node("expander", query_expander)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("quality", quality_check_node)
workflow.add_node("generator", generator_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("generate_alternatives", generate_alternatives)
workflow.add_node("select_best", select_best_draft)

# 入口
workflow.set_entry_point("router")


def route_after_router(state: LegalCaseState) -> Literal["greeting", "common", "expander"]:
    qtype = state.get("question_type", "legal")
    if qtype == "greeting":
        return "greeting"
    elif qtype == "common":
        return "common"
    else:
        return "expander"


workflow.add_conditional_edges("router", route_after_router, {
    "greeting": "greeting",
    "common": "common",
    "expander": "expander"
})

workflow.add_edge("greeting", END)
workflow.add_edge("common", END)

# 主流程：Router → 查询扩展 → 检索 → 质量校验 → 生成 → 审查
workflow.add_edge("expander", "retriever")
workflow.add_edge("retriever", "quality")


def after_quality(state: LegalCaseState) -> Literal["expander", "generator"]:
    if state.get("need_retry", False):
        return "expander"
    return "generator"


workflow.add_conditional_edges("quality", after_quality, {
    "expander": "expander",
    "generator": "generator"
})

workflow.add_edge("generator", "reviewer")


def after_reviewer(state: LegalCaseState) -> Literal["end", "expander", "generate_alternatives"]:
    if state["is_compliant"] == "PASS":
        return "end"
    if state["loop_count"] >= config.MAX_REWRITE_LOOPS:
        return "generate_alternatives"
    if "检索" in state.get("review_feedback", "") or "证据" in state.get("review_feedback", ""):
        return "expander"
    return "generator"


workflow.add_conditional_edges("reviewer", after_reviewer, {
    "end": END,
    "expander": "expander",
    "generator": "generator",
    "generate_alternatives": "generate_alternatives"
})

workflow.add_edge("generate_alternatives", "select_best")
workflow.add_edge("select_best", "reviewer")

legal_brain = workflow.compile()


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    test_query = "别人偷偷抄了我的包装盒设计拿去卖，我要去法院告他，最多能拿多少赔偿金？"
    final_state = legal_brain.invoke({
        "session_id": session_id,
        "user_query": test_query,
        "has_image": False,
        "image_path": None,
        "question_type": "",
        "search_keywords": "",
        "retrieved_evidence": "",
        "precedents": "",
        "compensation_info": "",
        "draft_opinion": "",
        "review_feedback": "",
        "loop_count": 0,
        "retrieval_retry_count": 0,
        "is_compliant": "",
        "confidence_score": 0.0,
        "alternative_drafts": [],
        "selected_draft_index": 0,
        "citation_check": {}
    })
    print("\n" + "="*60)
    print("最终法律意见：")
    print("="*60)
    print(final_state["draft_opinion"])
