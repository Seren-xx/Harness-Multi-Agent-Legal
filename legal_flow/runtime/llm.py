"""LLM 客户端：指数退避重试 / 双模型互备回退 / JSON 输出解析。"""
import json
import re
import time
from typing import Optional

import requests

from legal_flow import config

MAX_ATTEMPTS = 3
TIMEOUT = 60


def _normalize(messages: list) -> list:
    """合并连续相同角色的消息（满足 OpenAI 兼容 API 的硬性约束）"""
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg.get("content", "")
        else:
            merged.append({"role": msg["role"], "content": msg.get("content", "")})
    return merged


def _backoff(attempt: int) -> float:
    """指数退避：2s → 4s → 8s，封顶 30s"""
    return min(2.0 * (2 ** attempt), 30.0)


def _providers() -> dict:
    return {
        "dashscope": (config.DASHSCOPE_API_KEY, config.DASHSCOPE_BASE_URL,
                      config.DASHSCOPE_MODEL, config.DASHSCOPE_TEMP),
        "deepseek": (config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL,
                     config.DEEPSEEK_MODEL, config.DEEPSEEK_TEMP),
    }


def _fallback_of(model: str) -> str:
    """互备回退对：qwen ⇄ deepseek（主备链）"""
    return {"dashscope": "deepseek", "deepseek": "dashscope"}.get(model, "dashscope")


def call_llm(messages: list, model: str = "dashscope",
             temperature: Optional[float] = None,
             system_prompt: Optional[str] = None,
             _tried: tuple = ()) -> str:
    """调用大模型 API：429/5xx/网络错误自动重试；重试耗尽或账户问题（401/402/403）回退备用模型。"""
    providers = _providers()
    if model not in providers:
        raise ValueError(f"未知模型: {model}")
    api_key, base_url, model_name, default_temp = providers[model]

    if not api_key:
        fallback = _fallback_of(model)
        if providers[fallback][0]:
            return call_llm(messages, model=fallback, _tried=_tried + (model,),
                            temperature=temperature, system_prompt=system_prompt)
        raise ValueError("未配置任何 LLM API Key（请设置 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY）")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    api_messages = _normalize(messages)
    if system_prompt:
        api_messages = [{"role": "system", "content": system_prompt}] + api_messages
    data = {
        "model": model_name,
        "messages": api_messages,
        "temperature": temperature if temperature is not None else default_temp,
        "max_tokens": 8000,
    }

    last_error: Exception = RuntimeError("LLM 调用失败")
    for attempt in range(MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(f"{base_url}/chat/completions",
                                 headers=headers, json=data, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code in (401, 402, 403):
                # 账户问题（未授权/欠费/封禁）：重试无意义，立即回退另一家
                last_error = RuntimeError(f"API错误: {resp.status_code} - {resp.text[:200]}")
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.exceptions.ConnectionError(f"API {resp.status_code}")
            raise RuntimeError(f"API错误: {resp.status_code} - {resp.text[:200]}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(_backoff(attempt))

    # 重试耗尽 / 账户问题 → 回退到备用模型（_tried 防止互备双方都挂时循环回退）
    fallback = _fallback_of(model)
    if providers[fallback][0] and fallback not in _tried:
        print(f"⚠️ [LLM] {model} 不可用（{last_error}），回退到 {fallback}")
        return call_llm(messages, model=fallback, _tried=_tried + (model,),
                        temperature=temperature, system_prompt=system_prompt)
    raise last_error


def extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取第一个 JSON 对象（容忍 markdown 代码块 / 思考标签）"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def call_llm_json(messages: list, model: str = "dashscope",
                  temperature: float = 0) -> dict:
    """调用 LLM 并解析为 JSON；解析失败时追加修复指令重试一次；仍失败返回 {}。"""
    raw = call_llm(messages, model=model, temperature=temperature)
    parsed = extract_json(raw)
    if parsed is None:
        raw = call_llm(
            messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "上面的输出无法解析为 JSON。请只输出一个合法的 JSON 对象，不要任何其他文字。"},
            ],
            model=model, temperature=temperature,
        )
        parsed = extract_json(raw)
    return parsed or {}
