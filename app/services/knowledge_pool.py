from typing import Dict, Any

class KnowledgePool:
    def __init__(self):
        self.pool: Dict[str, Any] = {}

    def add(self, key: str, value: Any):
        self.pool[key] = value

    def get(self, key: str, default=None):
        return self.pool.get(key, default)
