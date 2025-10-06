"""
A thread-safe, lazy-loading Python class that reads a JSON or YAML configuration
file once and then exposes it globally through an importable singleton instance.

It supports environment-variable override for the path, attribute-style access,
dict-like access, and an explicit reload method.

Notes and recommendations:
--------------------------
Install PyYAML to use YAML files: pip install pyyaml.
This wrapper expects top-level object mapping; adapt the loader if you want
lists or other top-level types.
For nested attribute-style access (e.g., config.database.host), you can extend
the loader to wrap dicts in a lightweight namespace object; keep the current implementation explicit and predictable.


Usage Patterns:
---------------
1. Minimal using an environment variable
export CONFIG_PATH=/path/to/config.yaml
from config_loader import config
# first access triggers load
print(config.get("database"))
host = config.database["host"]

2. Explicit instantiate with path (non-singleton optional)
from config_loader import Config
c = Config("/path/to/config.json")  # returns the singleton instance
print(c.as_dict())

# If you need an independent instance (not global), create with _singleton=False:
temp_conf = Config(_singleton=False, path="tests/test_config.json")


3. Force reload
from config_loader import config
config.reload()

"""
# from __future__ import annotations
import json
import os
import threading
from typing import Any, Dict, Optional

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None  # YAML support optional; JSON always works


class ConfigError(RuntimeError):
    pass


class Config:
    """
    Thread-safe, lazy-loading config wrapper for JSON or YAML files.

    Behavior:
    - Load the file only once on first access (lazy).
    - Accept JSON or YAML by file extension (.json/.yml/.yaml).
    - Allow overriding the file path via environment variable CONFIG_PATH.
    - Provide attribute-style access (config.section.key) and dict-like access.
    - Provide reload() to force re-read from disk.
    """

    _instance_lock = threading.Lock()
    _global_instance: Optional["Config"] = None

    def __new__(cls, *args, _singleton: bool = True, **kwargs):
        if not _singleton:
            return super().__new__(cls)
        if Config._global_instance is None:
            with Config._instance_lock:
                if Config._global_instance is None:
                    Config._global_instance = super().__new__(cls)
        return Config._global_instance

    def __init__(self, config_path: Optional[str] = None, env_var: str = "CONFIG_PATH"):
        # __init__ may be called multiple times on the singleton; guard actual init
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._loaded = False
        self._env_var = env_var
        self._path = config_path
        self._initialized = True

    # --- Loading logic ------------------------------------------------
    def _resolve_path(self) -> str:
        env_path = os.getenv(self._env_var)
        res_path =  self._path if self._path is not None else env_path
        if res_path is None:
            raise ConfigError(f"No configuration path provided and {self._env_var} not set")
        return res_path

    @staticmethod
    def _load_from_file(path: str) -> Dict[str, Any]:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext in (".yml", ".yaml"):
                if yaml is None:
                    raise ConfigError("PyYAML is required to read YAML config files")
                return yaml.safe_load(f) or {}
            elif ext in (".jsn", ".json"):
                return json.load(f)
            else:
                # Try JSON first then YAML as fallback
                text = f.read()
                try:
                    return json.loads(text)
                except Exception:
                    if yaml is None:
                        raise ConfigError("Unknown file extension and PyYAML not installed")
                    return yaml.safe_load(text) or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            path = self._resolve_path()
            try:
                data = self._load_from_file(path)
            except FileNotFoundError as e:
                raise ConfigError(f"Configuration file not found: {path}") from e
            except Exception as e:
                raise ConfigError(f"Failed to load configuration: {e}") from e
            if not isinstance(data, dict):
                raise ConfigError("Configuration root must be a mapping (dict/object)")
            self._data = data
            self._loaded = True

    # --- Public API ---------------------------------------------------
    def reload(self):
        """Force re-read of the underlying file on next access."""
        with self._lock:
            self._loaded = False
            self._data = {}
        self._ensure_loaded()

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the loaded configuration dict."""
        self._ensure_loaded()
        return dict(self._data)

    # attribute-style access
    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        self._ensure_loaded()
        try:
            return self._data[item]
        except KeyError as e:
            raise AttributeError(item) from e

    # dict-like access
    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded()
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return self._data.get(key, default)

    def keys(self):
        self._ensure_loaded()
        return self._data.keys()

    def items(self):
        self._ensure_loaded()
        return self._data.items()

    def __contains__(self, key: str) -> bool:
        self._ensure_loaded()
        return key in self._data

    def __repr__(self) -> str:
        loaded = "loaded" if self._loaded else "not loaded"
        return f"<Config {loaded} keys={len(self._data)}>"

