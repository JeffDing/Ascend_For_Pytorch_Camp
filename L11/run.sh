#!/bin/bash
# 一键启动脚本：使用 torchrun 拉起 2 个进程（对应 2 张 NPU 卡）
# 用法：bash run.sh

set -e
cd "$(dirname "$0")"

torchrun --nproc_per_node=2 allgather_matmul_allreduce.py 2>&1 | tee run.log
