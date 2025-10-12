"""Distributed Training Worker Stub (US-024 Phase 5).

This module provides a stub/interface for future distributed training engines
(e.g., Kubernetes, Airflow, Celery). The current implementation uses Python's
ProcessPoolExecutor for local multi-core parallelism.

Future integrations could replace this with:
- Kubernetes Jobs: One pod per training window
- Airflow Tasks: Dynamic task generation for batch training
- Celery Workers: Distributed task queue across multiple machines
- Ray: Distributed computing framework for ML workloads

Example Kubernetes integration:
    from kubernetes import client, config

    def submit_k8s_training_job(task: dict) -> str:
        config.load_kube_config()
        batch_api = client.BatchV1Api()

        job = client.V1Job(
            metadata=client.V1ObjectMeta(name=f"teacher-{task['window_label']}"),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="trainer",
                                image="sensequant/teacher-trainer:latest",
                                args=[
                                    "--symbol", task["symbol"],
                                    "--start", task["start_date"],
                                    "--end", task["end_date"],
                                ],
                            )
                        ],
                        restart_policy="Never",
                    )
                )
            ),
        )
        batch_api.create_namespaced_job(namespace="default", body=job)
        return job.metadata.name
"""

from __future__ import annotations

from typing import Any, Protocol


class DistributedExecutor(Protocol):
    """Protocol for distributed training executors.

    Implementations must provide:
    - submit_task(): Submit training task to distributed system
    - wait_for_completion(): Wait for task completion
    - get_result(): Retrieve task result
    - cancel_task(): Cancel running task
    """

    def submit_task(self, task: dict[str, Any]) -> str:
        """Submit training task to distributed system.

        Args:
            task: Training task dictionary with symbol, dates, window info

        Returns:
            Task ID for tracking
        """
        ...

    def wait_for_completion(self, task_id: str, timeout: int = 600) -> bool:
        """Wait for task completion.

        Args:
            task_id: Task identifier from submit_task()
            timeout: Max wait time in seconds

        Returns:
            True if completed successfully, False if failed/timeout
        """
        ...

    def get_result(self, task_id: str) -> dict[str, Any]:
        """Retrieve task result.

        Args:
            task_id: Task identifier

        Returns:
            Result dictionary with status, metrics, error
        """
        ...

    def cancel_task(self, task_id: str) -> bool:
        """Cancel running task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled successfully
        """
        ...


class LocalProcessExecutor:
    """Local multi-core executor using ProcessPoolExecutor.

    This is the current default implementation. For distributed execution,
    replace this with KubernetesExecutor, AirflowExecutor, etc.
    """

    def submit_task(self, task: dict[str, Any]) -> str:
        """Submit task to local process pool.

        Note: Actual implementation is in train_teacher_batch.py using
        ProcessPoolExecutor. This is a stub for future refactoring.
        """
        raise NotImplementedError("Use train_teacher_batch.py --workers N for local parallelism")

    def wait_for_completion(self, task_id: str, timeout: int = 600) -> bool:
        """Wait for local process completion."""
        raise NotImplementedError("Use train_teacher_batch.py --workers N for local parallelism")

    def get_result(self, task_id: str) -> dict[str, Any]:
        """Get local process result."""
        raise NotImplementedError("Use train_teacher_batch.py --workers N for local parallelism")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel local process."""
        raise NotImplementedError("Use train_teacher_batch.py --workers N for local parallelism")


# Future executors to implement:

# class KubernetesExecutor:
#     """Kubernetes-based distributed executor."""
#     pass

# class AirflowExecutor:
#     """Apache Airflow-based executor."""
#     pass

# class CeleryExecutor:
#     """Celery-based distributed task queue executor."""
#     pass

# class RayExecutor:
#     """Ray-based distributed computing executor."""
#     pass
