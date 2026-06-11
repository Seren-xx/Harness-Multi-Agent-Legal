import json
import hashlib
import redis
import config
from typing import List, Dict, Optional


class ConversationMemory:
    """基于Redis的对话记忆管理器"""

    def __init__(self, redis_url=None):
        self.redis_url = redis_url or config.REDIS_URL
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            self.client.ping()
            self.use_redis = True
        except:
            self.use_redis = False
            self.local_store = {}

    def _get_key(self, session_id: str) -> str:
        return f"legal_agent:session:{session_id}"

    def _get_summary_key(self, session_id: str) -> str:
        return f"legal_agent:summary:{session_id}"

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到对话历史"""
        message = {"role": role, "content": content}
        key = self._get_key(session_id)

        if self.use_redis:
            self.client.rpush(key, json.dumps(message))
        else:
            if key not in self.local_store:
                self.local_store[key] = []
            self.local_store[key].append(message)

    def get_messages(self, session_id: str, limit: int = None) -> List[Dict]:
        """获取对话历史"""
        key = self._get_key(session_id)

        if self.use_redis:
            if limit:
                messages = self.client.lrange(key, -limit, -1)
            else:
                messages = self.client.lrange(key, 0, -1)
            return [json.loads(m) for m in messages]
        else:
            messages = self.local_store.get(key, [])
            if limit:
                return messages[-limit:]
            return messages

    def get_message_count(self, session_id: str) -> int:
        """获取消息数量"""
        key = self._get_key(session_id)
        if self.use_redis:
            return self.client.llen(key)
        return len(self.local_store.get(key, []))

    def compress_history(self, session_id: str, llm_callback) -> str:
        """压缩早期对话历史为摘要"""
        messages = self.get_messages(session_id)

        if len(messages) <= 4:
            return ""

        conversation_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages[:-2]]
        )

        prompt = f"""请将以下对话历史压缩为简洁的摘要，保留关键法律问题和结论：
{conversation_text}

摘要："""

        try:
            summary = llm_callback([{"role": "user", "content": prompt}], model="zhipuai")
            key = self._get_summary_key(session_id)
            if self.use_redis:
                self.client.set(key, summary)
            else:
                self.local_store[key] = summary

            messages_to_keep = messages[-2:]
            key = self._get_key(session_id)
            if self.use_redis:
                self.client.delete(key)
                for msg in messages_to_keep:
                    self.client.rpush(key, json.dumps(msg))
            else:
                self.local_store[key] = messages_to_keep

            return summary
        except:
            return ""

    def get_summary(self, session_id: str) -> str:
        """获取压缩摘要"""
        key = self._get_summary_key(session_id)
        if self.use_redis:
            return self.client.get(key) or ""
        return self.local_store.get(key, "")

    def clear_session(self, session_id: str):
        """清空会话"""
        key = self._get_key(session_id)
        summary_key = self._get_summary_key(session_id)
        if self.use_redis:
            self.client.delete(key, summary_key)
        else:
            self.local_store.pop(key, None)
            self.local_store.pop(summary_key, None)

    def get_context_for_llm(self, session_id: str, max_messages: int = 10) -> List[Dict]:
        """获取用于LLM的上下文（包含摘要）"""
        summary = self.get_summary(session_id)
        messages = self.get_messages(session_id, limit=max_messages)

        context = []
        if summary:
            context.append({
                "role": "system",
                "content": f"【历史对话摘要】{summary}"
            })

        context.extend(messages)
        return context


memory_manager = ConversationMemory()
