# CATHACTION MICCAI 2026 submission image (Task 1 segmentation + Task 2 collision).
#
# BUILD STATUS: UNVERIFIED. `docker` is not installed on the development host, so
# this image has NOT been built or smoke-tested here (recorded as a setup gap in
# quality_reports/plans/2026-06-15_prevalidation_roadmap.md). Build + in-container
# smoke must run on the submission machine (`apt install docker.io`) or in CI.
#
# Inference is fully automated (INV-6): no user interaction, no GT required.
# Select the task with the TASK env var (task1 | task2).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    TASK=task1 \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    DEVICE=cuda

WORKDIR /app

# Runtime deps beyond the base torch image. Mirrors environment-task2-gpu.yml.
# NOTE: ultralytics (YOLO11) is AGPL-3.0 — see the external-asset license manifest.
RUN pip install --no-cache-dir \
        "ultralytics>=8.3" \
        "segmentation-models-pytorch>=0.5" \
        "timm>=1.0" \
        "monai>=1.5" \
        opencv-python-headless \
        pillow numpy scipy scikit-learn pandas pyyaml tqdm

# Source, scripts, configs. Only the champion weights under outputs/ are copied;
# the rest of outputs/ (33G), datasets/, and external_repos/ are excluded via
# .dockerignore (which does NOT read .gitignore).
COPY pyproject.toml /app/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
COPY outputs/ /app/outputs/

COPY scripts/docker_entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
