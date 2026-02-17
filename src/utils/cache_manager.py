import pickle
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
import time


class CacheManager:
    """Управление кэшированием"""

    def __init__(self, cache_dir: str, ttl: int = 604800):  # 7 дней
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _get_cache_path(self, key: str) -> Path:
        """Получение пути к файлу кэша"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _get_meta_path(self, key: str) -> Path:
        """Получение пути к файлу метаданных"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}_meta.json"

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if not cache_path.exists() or not meta_path.exists():
            return None

        try:
            # Проверка срока действия
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if time.time() - meta['timestamp'] > self.ttl:
                self.delete(key)
                return None

            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def set(self, key: str, value: Any):
        """Сохранение в кэш"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            meta = {
                'key': key,
                'timestamp': time.time(),
                'ttl': self.ttl
            }

            with open(meta_path, 'w') as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"Ошибка сохранения в кэш: {e}")

    def delete(self, key: str):
        """Удаление из кэша"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if cache_path.exists():
            cache_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def clear(self):
        """Очистка всего кэша"""
        for file in self.cache_dir.glob("*"):
            file.unlink()