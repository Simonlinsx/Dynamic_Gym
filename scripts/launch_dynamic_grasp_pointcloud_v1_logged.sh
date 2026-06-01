#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=${LOG_FILE:-/data1/linsixu/simtoolreal/nohup_dynamic_grasp_pointcloud_v1.log}
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to $LOG_FILE"
bash /data1/linsixu/simtoolreal/scripts/launch_dynamic_grasp_pointcloud_v1.sh
