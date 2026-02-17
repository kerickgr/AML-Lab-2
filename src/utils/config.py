import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Загрузка конфигурационного файла"""
    path = Path(config_path)

    if not path.exists():
        return {}

    try:
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    """Сохранение конфигурации"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Ошибка сохранения конфига: {e}")