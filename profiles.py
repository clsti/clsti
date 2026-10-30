"""Robot profiles: everything model-specific about the humanoids and quadrupeds.

Joint and body names are templates: {s} left/right, {S} Left/Right, {c} l/r.  Standing poses are given for
the left side; the right side mirrors them (joints about x or z flip sign, joints about y keep it).

Hand frames follow one convention for every humanoid: x along the fingers, y such that the left palm faces
-y and the right palm +y.  Grasp offsets and hand collision parts are given for the left hand."""
import numpy as np

SIDES = (dict(s="left", S="Left", c="l"), dict(s="right", S="Right", c="r"))

HUMANOIDS = {
    "g1": dict(
        label="Unitree G1", xml="unitree_g1/g1_with_hands.xml", root="pelvis",
        leg=["{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_hip_yaw_joint", "{s}_knee_joint",
             "{s}_ankle_pitch_joint", "{s}_ankle_roll_joint"],
        foot="{s}_ankle_roll_link",
        arm=["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint", "{s}_elbow_joint",
             "{s}_wrist_roll_joint", "{s}_wrist_pitch_joint", "{s}_wrist_yaw_joint"],
        elbow="{s}_elbow_joint",
        waist=dict(yaw=("waist_yaw_joint", 1), roll=("waist_roll_joint", 1), pitch=("waist_pitch_joint", 1)),
        neck=None,
        stand={"{s}_hip_pitch_joint": -0.3, "{s}_knee_joint": 0.6, "{s}_ankle_pitch_joint": -0.3,
               "{s}_shoulder_pitch_joint": 0.2, "{s}_shoulder_roll_joint": 0.2, "{s}_elbow_joint": 1.0,
               "{s}_hand_thumb_1_joint": 1.05},
        hand=dict(kind="tripod", body="{s}_wrist_yaw_link", site=(0.0931, 0.0038, -0.0143), rot=None,
                  grasp=(0.081, -0.0563, 0.0263),
                  # (x0, x1, o0, o1, z0, z1) around the contact centre: x along fingers, o away from the block
                  parts=[(-0.06, 0.045, 0.0, 0.022, -0.042, 0.042),      # index + middle finger
                         (-0.11, -0.02, 0.0, 0.045, -0.015, 0.05),       # thumb
                         (-0.19, -0.04, 0.03, 0.085, -0.045, 0.045)]),   # palm + wrist
        fingers=dict(joints=["{s}_hand_thumb_0_joint", "{s}_hand_thumb_1_joint", "{s}_hand_thumb_2_joint",
                             "{s}_hand_middle_0_joint", "{s}_hand_middle_1_joint", "{s}_hand_index_0_joint",
                             "{s}_hand_index_1_joint"],
                     mirror=[1, -1, -1, -1, -1, -1, -1],
                     rest=[0.0, 1.05, 0.0, -0.35, -0.45, -0.35, -0.45],
                     open=[-0.55, 1.05, 0.10, -0.12, -0.16, -0.12, -0.16],
                     grip=[-0.55, 1.05, 0.38, -0.39, -0.51, -0.39, -0.51]),   # thumb + 2 fingertips on a plane
    ),
    "apollo": dict(
        label="Apptronik Apollo", xml="apptronik_apollo/apptronik_apollo.xml", root="base_link",
        leg=["{c}_hip_ie", "{c}_hip_aa", "{c}_hip_fe", "{c}_knee_fe", "{c}_ankle_ie", "{c}_ankle_pd"],
        foot="{c}_foot_link",
        arm=["{c}_shoulder_aa", "{c}_shoulder_ie", "{c}_shoulder_fe", "{c}_elbow_fe", "{c}_wrist_roll",
             "{c}_wrist_yaw", "{c}_wrist_pitch"],
        elbow="{c}_elbow_fe",
        waist=dict(yaw=("torso_yaw", -1), roll=("torso_roll", 1), pitch=("torso_pitch", 1)),
        neck=dict(yaw=("neck_yaw", 1), pitch=("neck_pitch", 1)),
        stand={"{c}_hip_ie": 0.08, "{c}_hip_aa": 0.1, "{c}_hip_fe": -0.48, "{c}_knee_fe": 1.03,
               "{c}_ankle_ie": -0.03, "{c}_ankle_pd": -0.58,
               "{c}_shoulder_aa": 0.12, "{c}_shoulder_fe": 0.1, "{c}_elbow_fe": -0.4},
        hand=dict(kind="palm", body="{c}_wrist_pitch_link", site=(0.0, 0.0, -0.12),
                  rot=((0, 0, 1), (0, 1, 0), (-1, 0, 0)),       # rows: fingers hang along -z of the wrist link
                  grasp=(0.035, -0.078, 0.0),
                  parts=[(-0.03, 0.065, 0.0, 0.04, -0.045, 0.045),        # fingers
                         (-0.125, -0.03, 0.017, 0.064, -0.045, 0.045),    # palm
                         (-0.125, -0.03, 0.017, 0.044, 0.045, 0.115),     # thumb
                         (-0.2, -0.12, 0.02, 0.12, -0.05, 0.05)]),        # wrist
    ),
    "n1": dict(
        label="Fourier N1", xml="fourier_n1/n1.xml", root="base_link",
        leg=["{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_hip_yaw_joint", "{s}_knee_pitch_joint",
             "{s}_ankle_pitch_joint", "{s}_ankle_roll_joint"],
        foot="{s}_foot_pitch_link",
        arm=["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint", "{s}_elbow_pitch_joint",
             "{s}_wrist_yaw_joint"],
        elbow="{s}_elbow_pitch_joint",
        waist=dict(yaw=("waist_yaw_joint", 1)),
        neck=None,
        stand={"{s}_hip_pitch_joint": -0.3, "{s}_knee_pitch_joint": 0.6, "{s}_ankle_pitch_joint": -0.3,
               "{s}_shoulder_roll_joint": 0.1, "{s}_elbow_pitch_joint": 1.3},
        hand=dict(kind="ball", body="{s}_end_effector_link", site=(0.0, 0.0, -0.005), radius=0.028),
    ),
    # ---- models from assets/extra (see tools/fetch_models.sh)
    "h1_2": dict(
        label="Unitree H1-2", src="extra", xml="unitree_mujoco/unitree_robots/h1_2/h1_2_handless.xml", root="pelvis",
        leg=["{s}_hip_yaw_joint", "{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_knee_joint",
             "{s}_ankle_pitch_joint", "{s}_ankle_roll_joint"],
        foot="{s}_ankle_roll_link",
        arm=["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint", "{s}_elbow_joint",
             "{s}_wrist_roll_joint", "{s}_wrist_pitch_joint", "{s}_wrist_yaw_joint"],
        elbow="{s}_elbow_joint", waist=dict(yaw=("torso_joint", 1)), neck=None,
        stand={"{s}_hip_pitch_joint": -0.35, "{s}_knee_joint": 0.7, "{s}_ankle_pitch_joint": -0.35,
               "{s}_shoulder_roll_joint": 0.12, "{s}_elbow_joint": 1.2},      # elbow zero is bent 90 degrees
        hand=dict(kind="ball", body="{s}_wrist_yaw_link", site=(0.035, 0.0, 0.0), radius=0.03),
    ),
    "r1": dict(
        label="Unitree R1", src="extra", xml="unitree_ros/robots/r1_description/R1.urdf", root="pelvis_link",
        leg=["{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_hip_yaw_joint", "{s}_knee_joint",
             "{s}_ankle_pitch_joint", "{s}_ankle_roll_joint"],
        foot="{s}_ankle_roll_link",
        arm=["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint", "{s}_elbow_joint",
             "{s}_wrist_roll_joint"],
        elbow="{s}_elbow_joint", waist=dict(yaw=("waist_yaw_joint", 1), roll=("waist_roll_joint", 1)),
        neck=dict(yaw=("head_yaw_joint", 1), pitch=("head_pitch_joint", 1)),
        stand={"{s}_hip_pitch_joint": -0.3, "{s}_knee_joint": 0.6, "{s}_ankle_pitch_joint": -0.3,
               "{s}_shoulder_roll_joint": 0.1, "{s}_elbow_joint": 1.2},        # elbow zero is bent 90 degrees
        hand=dict(kind="ball", body="{s}_wrist_roll_link", site=(0.085, 0.0, 0.0), radius=0.035),   # in the fist
    ),
    "dr02": dict(
        label="DEEP Robotics DR02", src="extra", xml="deeprobotics/DR02/mjcf/pro/DR02_pro.xml", root="base_link",
        leg=["{s}_hip_y_joint", "{s}_hip_x_joint", "{s}_hip_z_joint", "{s}_knee_joint", "{s}_ankle_y_joint",
             "{s}_ankle_x_joint"],
        foot="{s}_ankle_x_link",
        arm=["{s}_shoulder_y_joint", "{s}_shoulder_x_joint", "{s}_shoulder_z_joint", "{s}_elbow_joint",
             "{s}_wrist_z_joint", "{s}_wrist_y_joint", "{s}_wrist_x_joint"],
        elbow="{s}_elbow_joint",
        waist=dict(yaw=("waist_z_joint", 1), roll=("waist_x_joint", 1), pitch=("waist_y_joint", 1)),
        neck=dict(yaw=("neck_z_joint", 1), pitch=("neck_y_joint", 1)),
        stand={"{s}_hip_y_joint": -0.3, "{s}_knee_joint": 0.6, "{s}_ankle_y_joint": -0.3,
               "{s}_shoulder_x_joint": 0.1, "{s}_elbow_joint": 1.2},
        hand=dict(kind="ball", body="{s}_hand_link", site=(0.02, -0.01, -0.10), radius=0.045),
    ),
    "gr3": dict(
        label="Fourier GR-3", src="extra", xml="fourier/GRX/GR3/gr3v2_1_1/mjcf/gr3v2_1_1_dummy_hand.xml",
        root="base_link",
        leg=["{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_hip_yaw_joint", "{s}_knee_pitch_joint",
             "{s}_ankle_pitch_joint", "{s}_ankle_roll_joint"],
        foot="{s}_foot_roll_link",
        arm=["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint",
             "{s}_elbow_pitch_joint", "{s}_wrist_yaw_joint", "{s}_wrist_pitch_joint", "{s}_wrist_roll_joint"],
        elbow="{s}_elbow_pitch_joint",
        waist=dict(yaw=("waist_yaw_joint", 1), roll=("waist_roll_joint", 1), pitch=("waist_pitch_joint", 1)),
        neck=dict(yaw=("head_yaw_joint", 1), pitch=("head_pitch_joint", 1)),
        stand={"{s}_hip_pitch_joint": -0.3, "{s}_knee_pitch_joint": 0.6, "{s}_ankle_pitch_joint": -0.3},
        hand=dict(kind="ball", body="{s}_hand_roll_link", site=(0.05, 0.0, -0.13), radius=0.05),
    ),
    "atlas": dict(
        label="Boston Dynamics Atlas (DRC)", src="extra", xml="drake_models/atlas/atlas_convex_hull.urdf",
        packages={"drake_models": "drake_models"}, root="pelvis",
        leg=["{c}_leg_hpz", "{c}_leg_hpx", "{c}_leg_hpy", "{c}_leg_kny", "{c}_leg_aky", "{c}_leg_akx"],
        foot="{c}_foot",
        arm=["{c}_arm_shz", "{c}_arm_shx", "{c}_arm_ely", "{c}_arm_elx", "{c}_arm_uwy", "{c}_arm_mwx", "{c}_arm_lwy"],
        elbow="{c}_arm_elx",
        waist=dict(yaw=("back_bkz", 1), roll=("back_bkx", 1), pitch=("back_bky", 1)),
        neck=dict(pitch=("neck_ay", 1)),
        stand={"{c}_leg_hpy": -0.3, "{c}_leg_kny": 0.6, "{c}_leg_aky": -0.3, "{c}_arm_shx": -1.45, "{c}_arm_elx": 0.3},
        max_turn=0.23,      # hip yaw turns 0.17 rad inwards but 0.79 outwards; turns mostly use the outward range
        # the Drake model's hands are a convex hull of an open Robotiq 3-finger gripper: replaced by the
        # articulated gripper (palm frame: fingers along +y, finger 1 and 2 at -x, the middle finger at +x)
        gripper=dict(body="{s}_palm", pos=(0.0, 0.054, 0.0), rgba=(0.16, 0.16, 0.17, 1.0)),
        # fingertip pinch: the block's face lies 13.6 cm out from the palm frame, the finger pads on the faces
        # next to it; the fingers close horizontally (fixed alpha), the left palm faces -y as for every hand
        hand=dict(kind="gripper", body="{s}_rq_palm", site=(0.0, 0.136, 0.0), mirror=False, alpha=0.0, tilt=0.75,
                  rot=((1, 0, 0), (0, -1, 0), (0, 0, -1)), rot_right=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                  grasp=(0.0, 0.0, 0.0),
                  parts=[(-0.065, -0.043, -0.04, 0.085, -0.05, 0.05),     # finger 1 + 2
                         (0.043, 0.065, -0.04, 0.085, -0.02, 0.02),       # middle finger
                         (-0.07, 0.065, 0.085, 0.19, -0.065, 0.065)]),    # palm
        fingers=dict(joints=[f"{{s}}_rq_finger_{f}_joint_{k}" for f in ("1", "2", "middle") for k in (1, 2, 3)],
                     mirror=[1] * 9,
                     rest=[0.35, 0.3, -0.5] * 3, open=[0.12, 0.0, -0.12] * 3, grip=[0.445, 0.0, -0.445] * 3),
    ),
}

QUADRUPEDS = {
    "anymalw": dict(
        label="ANYmal on wheels", xml="anybotics_anymal_c/anymal_c.xml", root="base", yaw_offset=0.0,
        legs=["LF", "RF", "LH", "RH"], joints=["{l}_HAA", "{l}_HFE", "{l}_KFE"],
        shank="{l}_SHANK", seed=None,
        wheel=dict(radius=0.11, width=0.06, lateral=0.04),
        ride_z=0.48, stance=(0.42, 0.33),
    ),
    "spot": dict(
        label="Boston Dynamics Spot", xml="boston_dynamics_spot/spot.xml", root="body", yaw_offset=0.0,
        legs=["fl", "fr", "hl", "hr"], joints=["{l}_hx", "{l}_hy", "{l}_kn"],
        shank="{l}_lleg", seed=(0.0, 1.04, -1.8),
        wheel=None, ride_z=0.47, stance=(0.30, 0.17),
    ),
    # ---- models from assets/extra (see tools/fetch_models.sh)
    "b2w": dict(
        label="Unitree B2-W", src="extra", xml="unitree_mujoco/unitree_robots/b2w/b2w.xml", root="base_link",
        yaw_offset=0.0, legs=["FL", "FR", "RL", "RR"],
        joints=["{l}_hip_joint", "{l}_thigh_joint", "{l}_calf_joint"], shank="{l}_calf", seed=(0.0, 0.8, -1.6),
        wheel=dict(native=True, body="{l}_wheel_link", joint="{l}_wheel_joint"), ride_z="auto", stance="auto",
    ),
    "go2w": dict(
        label="Unitree Go2-W", src="extra", xml="unitree_mujoco/unitree_robots/go2w/go2w.xml", root="base_link",
        yaw_offset=0.0, legs=["FL", "FR", "RL", "RR"],
        joints=["{l}_hip_joint", "{l}_thigh_joint", "{l}_calf_joint"], shank="{l}_calf", seed=(0.0, 0.8, -1.6),
        wheel=dict(native=True, body="{l}_wheel_link", joint="{l}_wheel_joint"), ride_z="auto", stance="auto",
    ),
    "m20": dict(
        label="DEEP Robotics M20", src="extra", xml="deeprobotics/M20/mjcf/M20.xml", root="base_link",
        yaw_offset=0.0, legs=["fl", "fr", "hl", "hr"],
        joints=["{l}_hipx_joint", "{l}_hipy_joint", "{l}_knee_joint"], shank="{l}_knee", seed=None, knees="x",
        wheel=dict(native=True, body="{l}_wheel", joint="{l}_wheel_joint"), ride_z="auto", stance="auto",
    ),
    "x30": dict(
        label="DEEP Robotics X30", src="extra", xml="deeprobotics/X30/mjcf/X30.xml", root="TORSO",
        yaw_offset=0.0, legs=["FL", "FR", "HL", "HR"],
        joints=["{l}_HipX_joint", "{l}_HipY_joint", "{l}_Knee_joint"], shank="{l}_SHANK", seed=None, knees="back",
        wheel=None, ride_z="auto", stance="auto",
    ),
    "a2": dict(
        label="Unitree A2", src="extra", xml="unitree_mujoco/unitree_robots/a2/a2.xml", root="base_link",
        yaw_offset=0.0, legs=["FL", "FR", "RL", "RR"],
        joints=["{l}_hip_joint", "{l}_thigh_joint", "{l}_calf_joint"], shank="{l}_calf", seed=(0.0, 0.8, -1.6),
        wheel=None, ride_z="auto", stance="auto",
    ),
    "anymal": dict(
        label="ANYbotics ANYmal", xml="anybotics_anymal_c/anymal_c.xml", root="base", yaw_offset=0.0,
        legs=["LF", "RF", "LH", "RH"], joints=["{l}_HAA", "{l}_HFE", "{l}_KFE"], shank="{l}_SHANK", seed=None,
        wheel=None, ride_z="auto", stance="auto",
    ),
}

ORDER_H = ["g1", "apollo", "n1", "h1_2", "r1", "dr02", "gr3", "atlas"]
ORDER_Q = ["go2w", "b2w", "anymalw", "m20", "spot", "x30", "a2", "anymal"]


def pair_for(date):
    """Robots of the day: every humanoid meets every quadruped once in 64 days, in a fixed shuffled order."""
    combos = [(h, q) for h in ORDER_H for q in ORDER_Q]
    perm = np.random.default_rng(7).permutation(len(combos))
    return combos[perm[date.toordinal() % len(combos)]]


def name(template, side):
    return template.format(**side)


def mirror_sign(axis, right):
    """Right-side joints about x or z take the opposite sign of the left-side value."""
    return -1.0 if right and (abs(axis[0]) > 0.5 or abs(axis[2]) > 0.5) else 1.0
