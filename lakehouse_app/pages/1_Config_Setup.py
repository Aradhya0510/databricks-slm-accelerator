"""Config Setup — Build and manage YAML configurations for SLM fine-tuning."""

import streamlit as st
from components.theme import inject_theme, page_header, section_title
from components.config_forms import ConfigFormBuilder
from utils.config_generator import ConfigGenerator
from utils.state_manager import StateManager

inject_theme()
StateManager.initialize()

page_header("Config Setup", "Build YAML configurations for your fine-tuning pipeline")

tab_build, tab_load, tab_preview = st.tabs(["Build Config", "Load Config", "Preview"])

with tab_build:
    section_title("Task & Model")
    task = ConfigFormBuilder.task_selector()
    model_name, model_info = ConfigFormBuilder.model_selector(task)

    section_title("Quantization & LoRA")
    lora_config = ConfigFormBuilder.lora_config_form()

    section_title("Data")
    data_config = ConfigFormBuilder.data_config_form(task)

    section_title("Training")
    training_config = ConfigFormBuilder.training_config_form(task)

    section_title("MLflow")
    mlflow_config = ConfigFormBuilder.mlflow_config_form()

    section_title("Output")
    output_config = ConfigFormBuilder.output_config_form()

    st.markdown("")
    if st.button("Generate Configuration", type="primary", use_container_width=True):
        config = ConfigGenerator.generate_config(
            task=task,
            model_name=model_name,
            data_config=data_config,
            training_config=training_config,
            lora_config=lora_config,
            mlflow_config=mlflow_config,
            output_config=output_config,
        )
        is_valid, errors = ConfigGenerator.validate_config(config)
        if not is_valid:
            for err in errors:
                st.error(err)
        else:
            StateManager.set_current_config(config)
            st.success("Configuration generated and saved to session.")
            st.code(ConfigGenerator.get_config_preview(config), language="yaml")

    st.markdown("")
    section_title("Save to File")
    save_path = st.text_input("Save Path", value="/Volumes/<catalog>/<schema>/<volume>/configs/my_config.yaml")
    if st.button("Save Config", use_container_width=True):
        config = StateManager.get_current_config()
        if config:
            try:
                ConfigGenerator.save_config(config, save_path)
                StateManager.set_current_config(config, save_path)
                st.success(f"Saved to `{save_path}`")
            except Exception as e:
                st.error(f"Save failed: {e}")
        else:
            st.warning("Generate a configuration first.")

with tab_load:
    section_title("Load from File")
    load_path = st.text_input("Config File Path", value="")
    if st.button("Load Config", use_container_width=True) and load_path:
        try:
            config = ConfigGenerator.load_config(load_path)
            is_valid, errors = ConfigGenerator.validate_config(config)
            if not is_valid:
                for err in errors:
                    st.warning(err)
            StateManager.set_current_config(config, load_path)
            st.success("Configuration loaded.")
            st.code(ConfigGenerator.get_config_preview(config), language="yaml")
        except Exception as e:
            st.error(f"Load failed: {e}")

    section_title("Recent Configs")
    recent = StateManager.get("recent_configs", [])
    if recent:
        for path in recent:
            c1, c2 = st.columns([4, 1])
            c1.text(path)
            if c2.button("Load", key=f"recent_{path}"):
                try:
                    config = ConfigGenerator.load_config(path)
                    StateManager.set_current_config(config, path)
                    st.success(f"Loaded: {path}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.info("No recent configs.")

with tab_preview:
    config = StateManager.get_current_config()
    if config:
        st.code(ConfigGenerator.get_config_preview(config), language="yaml")
        st.download_button(
            "Download YAML",
            data=ConfigGenerator.get_config_preview(config),
            file_name="slm_config.yaml",
            mime="text/yaml",
        )
    else:
        st.info("No configuration loaded. Build or load one first.")
