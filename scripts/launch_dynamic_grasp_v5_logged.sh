#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=/data1/linsixu/simtoolreal/nohup_dynamic_grasp_v5.log
: > "${LOG_FILE}"

exec bash /data1/linsixu/simtoolreal/scripts/launch_dynamic_grasp_v5.sh \
  >> "${LOG_FILE}" 2>&1
