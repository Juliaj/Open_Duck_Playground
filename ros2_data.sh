cd /home/juliajia/dev/Open_Duck_Playground
model="/home/juliajia//dev/Open_Duck_Playground/checkpoints/2025_12_26_165635_300482560.onnx"
uv run python3 ~/ros2_ws/src/ros-controls/ros2_control_demos/example_18/fine_tuning/manual_control_ros2.py   --output mujoco_manual_data.h5   --onnx-model $model
