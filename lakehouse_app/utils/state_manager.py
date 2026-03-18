"""State Manager for the SLM Lakehouse App."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st


class StateManager:
    """Manage application state and session persistence."""

    DEFAULT_STATE = {
        "current_config": None,
        "config_path": None,
        "selected_task": "instruction_tuning",
        "selected_model": None,
        "active_training_run": None,
        "training_history": [],
        "registered_models": [],
        "endpoints": [],
        "default_catalog": "main",
        "default_schema": "slm_models",
        "default_volume": "slm_data",
        "workspace_email": None,
        "recent_configs": [],
    }

    @classmethod
    def initialize(cls):
        for key, value in cls.DEFAULT_STATE.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any):
        st.session_state[key] = value

    @classmethod
    def update(cls, updates: Dict[str, Any]):
        for key, value in updates.items():
            st.session_state[key] = value

    @classmethod
    def get_current_config(cls) -> Optional[Dict[str, Any]]:
        return cls.get("current_config")

    @classmethod
    def set_current_config(cls, config: Dict[str, Any], config_path: Optional[str] = None):
        cls.set("current_config", config)
        if config_path:
            cls.set("config_path", config_path)
            cls.add_recent_config(config_path)

    @classmethod
    def add_recent_config(cls, config_path: str, max_recent: int = 10):
        recent = cls.get("recent_configs", [])
        if config_path in recent:
            recent.remove(config_path)
        recent.insert(0, config_path)
        cls.set("recent_configs", recent[:max_recent])

    @classmethod
    def add_training_run(cls, run_info: Dict[str, Any]):
        history = cls.get("training_history", [])
        if "timestamp" not in run_info:
            run_info["timestamp"] = datetime.now().isoformat()
        history.insert(0, run_info)
        cls.set("training_history", history[:50])

    @classmethod
    def set_active_training_run(cls, run_id: Optional[str]):
        cls.set("active_training_run", run_id)

    @classmethod
    def get_active_training_run(cls) -> Optional[str]:
        return cls.get("active_training_run")

    @classmethod
    def add_registered_model(cls, model_info: Dict[str, Any]):
        models = cls.get("registered_models", [])
        existing_idx = next(
            (i for i, m in enumerate(models) if m.get("name") == model_info.get("name")),
            None,
        )
        if existing_idx is not None:
            models[existing_idx] = model_info
        else:
            models.insert(0, model_info)
        cls.set("registered_models", models)

    @classmethod
    def add_endpoint(cls, endpoint_info: Dict[str, Any]):
        endpoints = cls.get("endpoints", [])
        existing_idx = next(
            (i for i, e in enumerate(endpoints) if e.get("endpoint_name") == endpoint_info.get("endpoint_name")),
            None,
        )
        if existing_idx is not None:
            endpoints[existing_idx] = endpoint_info
        else:
            endpoints.insert(0, endpoint_info)
        cls.set("endpoints", endpoints)

    @classmethod
    def get_user_preferences(cls) -> Dict[str, Any]:
        return {
            "default_catalog": cls.get("default_catalog", "main"),
            "default_schema": cls.get("default_schema", "slm_models"),
            "default_volume": cls.get("default_volume", "slm_data"),
            "workspace_email": cls.get("workspace_email"),
        }

    @classmethod
    def set_user_preferences(cls, preferences: Dict[str, Any]):
        cls.update(preferences)

    @classmethod
    def get_default_paths(cls) -> Dict[str, str]:
        prefs = cls.get_user_preferences()
        catalog = prefs.get("default_catalog", "main")
        schema = prefs.get("default_schema", "slm_models")
        volume = prefs.get("default_volume", "slm_data")
        email = prefs.get("workspace_email", "user@email.com")
        base_path = f"/Volumes/{catalog}/{schema}/{volume}"
        return {
            "data_path": f"{base_path}/data",
            "checkpoint_path": f"{base_path}/checkpoints",
            "volume_checkpoint_path": f"{base_path}/volume_checkpoints",
            "results_path": f"{base_path}/results",
            "experiment_path": f"/Users/{email}/slm_experiments",
        }

    @classmethod
    def reset_state(cls):
        for key, value in cls.DEFAULT_STATE.items():
            st.session_state[key] = value
