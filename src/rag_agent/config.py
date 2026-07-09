"""配置加载: 读取 config.yaml。

api_key 等敏感值不写进 yaml, 在 models.py 按 provider 从 .env 读。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# 项目根目录: src/rag_agent/config.py -> parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 支持 RAG_CONFIG_PATH 环境变量覆盖: pip 安装后 parents[2] 不再指向仓库根,
# 可显式指定 config.yaml 位置。未设置时回退到仓库内默认路径。
_env_config = os.environ.get("RAG_CONFIG_PATH")
CONFIG_PATH = Path(_env_config) if _env_config else PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """加载并缓存 config.yaml(同时载入 .env 供 models.py 读环境变量)。"""
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"找不到配置文件 {CONFIG_PATH}。"
            "可通过环境变量 RAG_CONFIG_PATH 指定 config.yaml 路径。"
        ) from e


cfg = load_config()
