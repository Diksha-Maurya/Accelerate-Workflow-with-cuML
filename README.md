# Accelerate-Workflow-with-cuML

This project showcases how to leverage NVIDIA RAPIDS cuML to accelerate machine learning tasks on both CPU-only and GPU-enabled systems with minimal or zero code changes.

Key highlights:
- CPU Execution: Using cuml-cpu for environments without GPU support.
- GPU Execution: Running high-performance ML algorithms on NVIDIA GPUs.
- UMAP Dimensionality Reduction: Fast embedding of datasets with trustworthiness score evaluation.
- Device Management: Controlling execution with using_device_type for selective CPU/GPU processing.
- Cross-Device Model Serialization: Train on one device, save with pickle, and load for inference on another device.

Technologies & Libraries:
- RAPIDS cuML (cuml, cuml-cpu)
- scikit-learn
- Pandas
- NumPy
