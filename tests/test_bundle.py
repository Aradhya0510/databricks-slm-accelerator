"""The bundle stays on serverless compute and its task graph stays wired up.

Bundle validation only checks the schema. It accepts a classic cluster, a
notebook path that does not exist, and a task value read from a task that
never ran first -- all of which fail at run time, after GPUs are provisioned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
BUNDLE = yaml.safe_load((REPO / "databricks.yml").read_text())
JOBS = BUNDLE["resources"]["jobs"]

GPU_STAGES = {"train", "evaluate", "register"}
ACCELERATORS = {"GPU_1xA10", "GPU_1xH100", "GPU_8xH100", "GPU_8xB300"}

# Fields that only exist on classic compute.
_CLASSIC_KEYS = {
    "job_clusters", "job_cluster_key", "new_cluster", "existing_cluster_id",
    "spark_version", "node_type_id", "spark_python_task", "libraries",
}


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def _tasks():
    for job_key, job in JOBS.items():
        for task in job["tasks"]:
            yield job_key, job, task


def test_bundle_defines_no_classic_compute():
    found = _CLASSIC_KEYS & set(_keys(BUNDLE))
    assert not found, f"classic compute fields in databricks.yml: {sorted(found)}"


def test_default_accelerator_is_a_real_ai_runtime_type():
    assert BUNDLE["variables"]["accelerator"]["default"] in ACCELERATORS


@pytest.mark.parametrize(
    "job_key, job, task", list(_tasks()), ids=[f"{j}.{t['task_key']}" for j, _, t in _tasks()],
)
def test_notebook_paths_exist(job_key, job, task):
    path = task["notebook_task"]["notebook_path"]
    assert (REPO / path).is_file(), f"{job_key}.{task['task_key']}: {path} does not exist"


def test_gpu_stages_run_on_ai_runtime():
    job = JOBS["slm_finetuning_pipeline"]
    environments = {e["environment_key"] for e in job["environments"]}

    for task in job["tasks"]:
        if task["task_key"] in GPU_STAGES:
            assert task["compute"]["hardware_accelerator"] == "${var.accelerator}"
            assert task["environment_key"] in environments
        else:
            # Anything else stays off the GPU: an endpoint warming up for half
            # an hour on an idle accelerator is pure cost.
            assert "compute" not in task, f"{task['task_key']} should run on serverless CPU"


def test_task_values_are_read_from_upstream_tasks():
    """A task can only read the values of tasks that have already run."""
    reference = re.compile(r"\{\{tasks\.([a-z_]+)\.values\.[a-z_]+\}\}")

    for job_key, job in JOBS.items():
        upstream = {
            t["task_key"]: {d["task_key"] for d in t.get("depends_on", [])}
            for t in job["tasks"]
        }

        def ancestors(key):
            seen = set()
            stack = list(upstream[key])
            while stack:
                parent = stack.pop()
                if parent not in seen:
                    seen.add(parent)
                    stack.extend(upstream[parent])
            return seen

        for task in job["tasks"]:
            params = task["notebook_task"].get("base_parameters", {})
            for value in params.values():
                for source in reference.findall(str(value)):
                    assert source in ancestors(task["task_key"]), (
                        f"{job_key}.{task['task_key']} reads {source}'s values "
                        f"but does not run after it"
                    )


def _widgets(task):
    source = (REPO / task["notebook_task"]["notebook_path"]).read_text()
    return set(re.findall(r'dbutils\.widgets\.\w+\("([a-z_]+)"', source))


@pytest.mark.parametrize(
    "job_key, job, task", list(_tasks()), ids=[f"{j}.{t['task_key']}" for j, _, t in _tasks()],
)
def test_every_parameter_a_task_is_given_is_a_widget_it_reads(job_key, job, task):
    """A parameter with no matching widget is dropped without an error, so a
    misspelt hand-off would leave the next stage running on its default."""
    widgets = _widgets(task)
    assert "config_path" in widgets, f"{task['task_key']} ignores config_path"
    for name in task["notebook_task"].get("base_parameters", {}):
        assert name in widgets, f"{job_key}.{task['task_key']} is passed {name} but never reads it"
