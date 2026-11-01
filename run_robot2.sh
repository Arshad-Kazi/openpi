#!/bin/bash
# run_robot2.sh — full xarm inference stack, one tmux WINDOW per component.
# Windows have no size limit, so this never hits "no space for new pane".
#   ./run_robot2.sh        # start + attach
#   ./run_robot2.sh stop   # kill session + sweep orphaned ros2 nodes
# Switch windows: Ctrl-b then 0..5  |  detach: Ctrl-b d  |  reattach: tmux attach -t robot
set -eu
SESSION="robot"

OPENPI_DIR="/home/rt2/arshad/openpi"
TEST_WS="/home/rt2/Documents/test_ws"
AIML_WS="/home/rt2/Documents/aiml_ws_seongheon"
ROBOT_IP="192.168.1.102"
CONFIG_PATH="/mnt/elements/sparse_block_pick_ckpt/sparse_block_picking_with_fix_start_robot_trimmed/openpi_sparse_block_picking.json"
CKPT_DIR="/mnt/elements/sparse_block_pick_ckpt/sparse_block_picking_with_fix_start_robot_trimmed/31999"
# asset_id = subfolder under <CKPT_DIR>/assets/local/ that holds norm_stats.json
ASSET_ID="local/sparse_block_picking_with_fix_start_robot_trimmed"
CONDA_ENV="aiml_ws_sh"
SRC_ROS="source /opt/ros/\$ROS_DISTRO/setup.bash 2>/dev/null; source $TEST_WS/install/setup.bash"

if [ "${1:-}" = "stop" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed '$SESSION'" || echo "no session"
  for p in serve_policy xarm6_driver foxglove_bridge camera_only_launch inference_test inference_watcher; do
    pkill -f "$p" 2>/dev/null || true
  done
  echo "swept orphaned ros2 nodes."
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running — attaching. (stop: $0 stop)"
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n server -c "$OPENPI_DIR"
tmux send-keys -t "$SESSION:server" \
  "uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_sparse_block_picking --policy.dir '$CKPT_DIR' --policy.asset_id '$ASSET_ID'" C-m

tmux new-window -t "$SESSION" -n xarm -c "$TEST_WS"
tmux send-keys -t "$SESSION:xarm" \
  "$SRC_ROS; ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=$ROBOT_IP report_type:=dev add_gripper:=true" C-m

tmux new-window -t "$SESSION" -n cameras -c "$TEST_WS"
tmux send-keys -t "$SESSION:cameras" \
  "$SRC_ROS; ros2 launch data_collection camera_only_launch.py" C-m

tmux new-window -t "$SESSION" -n foxglove -c "$TEST_WS"
tmux send-keys -t "$SESSION:foxglove" \
  "$SRC_ROS; ros2 launch foxglove_bridge foxglove_bridge_launch.xml" C-m

tmux new-window -t "$SESSION" -n inference -c "$TEST_WS"
tmux send-keys -t "$SESSION:inference" \
  "$SRC_ROS; echo 'waiting 10s...'; sleep 10; ros2 launch xarm_stack inference_test.launch.py config_path:=\"$CONFIG_PATH\"" C-m

tmux new-window -t "$SESSION" -n watcher -c "$AIML_WS"
tmux send-keys -t "$SESSION:watcher" \
  "conda activate $CONDA_ENV; echo 'waiting 15s...'; sleep 15; python scripts/inference_watcher.py --config \"$CONFIG_PATH\"" C-m

tmux select-window -t "$SESSION:server"
tmux attach -t "$SESSION"
