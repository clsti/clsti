#!/bin/sh
# Download the robot models (sparse, only the folders that are used), about 950 MB in total.
#   assets/menagerie  MuJoCo Menagerie: G1, Apollo, N1, ANYmal C, Spot          (BSD-3 / Apache-2.0)
#   assets/extra      unitree_mujoco + unitree_ros: H1-2, R1, B2-W, Go2-W, A2    (BSD-3)
#                     DEEP Robotics URDF_model: DR02, M20, X30                    (BSD-3)
#                     Fourier Wiki-GRx-Models: GR-3                               (GPL-3.0)
#                     Drake models: Atlas (DRC)                                   (BSD-style, MIT)
#                     ROS-Industrial robotiq: Robotiq 3-finger gripper for Atlas  (BSD-2)
set -e
cd "$(dirname "$0")/.."
sparse() {  # sparse <url> <dir> <paths...>
  url=$1; dir=$2; shift 2
  [ -d "$dir/.git" ] || git clone -q --depth 1 --filter=blob:none --sparse "$url" "$dir"
  git -C "$dir" sparse-checkout set "$@"
}
mkdir -p assets/extra
sparse https://github.com/google-deepmind/mujoco_menagerie assets/menagerie \
  unitree_g1 apptronik_apollo fourier_n1 anybotics_anymal_c boston_dynamics_spot
sparse https://github.com/unitreerobotics/unitree_mujoco assets/extra/unitree_mujoco \
  unitree_robots/h1_2 unitree_robots/b2w unitree_robots/go2w unitree_robots/a2
sparse https://github.com/unitreerobotics/unitree_ros assets/extra/unitree_ros robots/r1_description
sparse https://github.com/DeepRoboticsLab/URDF_model assets/extra/deeprobotics DR02 M20 X30
sparse https://github.com/FFTAI/Wiki-GRx-Models assets/extra/fourier GRX/GR3/gr3v2_1_1
sparse https://github.com/RobotLocomotion/models assets/extra/drake_models atlas
sparse https://github.com/ros-industrial-attic/robotiq assets/extra/robotiq robotiq_3f_gripper_visualization
# Atlas' visual meshes are glTF; MuJoCo reads the OBJ copies
python tools/gltf2obj.py assets/extra/drake_models/atlas/meshes > /dev/null
python tools/dae2obj.py assets/extra/robotiq/robotiq_3f_gripper_visualization/meshes/robotiq-3f-gripper_articulated/visual > /dev/null
echo "models ready"
