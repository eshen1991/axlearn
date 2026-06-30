#!/bin/bash
# Copyright © 2026 Apple Inc.
# Shell script to export and print GKE TPU Pathways training environment variables.
# IMPORTANT: To make these variables persist in your active terminal, run: source ./axlearn/tools/setup_grpo_env.sh

export BASTION_TIER=disabled
export NAME=eshen-v6e-rl-grpo-pathways
export RUNNER_NAME=gke_tpu_pathways
export CONFIG=grpo-fuji-8B-v1
export INSTANCE_TYPE=tpu-v6e-16
export CLUSTER=ericshen-axlearn
export NOW=$(date +%s)
export OUTPUT_DIR="gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/${NAME}/${NOW}"
export MLDIAGNOSTICS_RUN_GROUP="eshen-grpo-experiment-${NOW}"
# export COLOCATED_PYTHON_DESERIALIZE=0
# export GRPO_REWARD_TYPE=gsm8k

echo "=============================================================="
echo "GKE TPU PATHWAYS ENVIRONMENT VARIABLES SET SUCCESSFULLY:"
echo "=============================================================="
echo "BASTION_TIER  : ${BASTION_TIER}"
echo "NAME          : ${NAME}"
echo "RUNNER_NAME   : ${RUNNER_NAME}"
echo "CONFIG        : ${CONFIG}"
echo "INSTANCE_TYPE : ${INSTANCE_TYPE}"
echo "CLUSTER       : ${CLUSTER}"
echo "OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "MLDIAGNOSTICS_RUN_GROUP : ${MLDIAGNOSTICS_RUN_GROUP}"
#echo "COLOCATED_PYTHON_DESERIALIZE : ${COLOCATED_PYTHON_DESERIALIZE}"
#echo "GRPO_REWARD_TYPE : ${GRPO_REWARD_TYPE}"
echo "=============================================================="
