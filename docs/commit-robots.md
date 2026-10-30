# commit-robots

A GitHub profile animation of your commits per day, shot like lab footage. Every active day of the
contribution calendar (53 weeks × 7 days) is a physical wooden block on a printed rubber mat. Busier
days get taller, darker blocks.

1. A quadruped crashes through the calendar and knocks it apart. The wheeled ones slalom, spin a donut and do a
   **backflip**; the walking ones trot through, hop round 180 degrees and dance. Then it play-bows and wiggles at
   the humanoid.
2. The humanoid facepalms and shakes its head. Then it walks over and rebuilds the calendar block by block,
   week by week, so the loop ends where it started.

**The robots change every day.** Every humanoid meets every quadruped once in a 64-day cycle (a fixed shuffled
order, so the pair of the day is the same on every machine):

| Humanoids | Quadrupeds on wheels | Walking quadrupeds |
|---|---|---|
| Unitree G1 (three-finger Dex3 hands) | Unitree Go2-W | Boston Dynamics Spot |
| Unitree H1-2 | Unitree B2-W | Unitree A2 |
| Unitree R1 | DEEP Robotics M20 | DEEP Robotics X30 |
| Apptronik Apollo | ANYmal on wheels (Swiss-Mile style) | ANYbotics ANYmal |
| Fourier N1 | | |
| Fourier GR-3 | | |
| DEEP Robotics DR02 | | |
| Boston Dynamics Atlas (DRC version, the one with a public model), with Robotiq 3-finger grippers | | |

Not in the rota: Unitree H1 and H2, Fourier GR-2 and Booster T1 cannot reach blocks on the floor (too little
waist / hip / ankle range for their arm length); PAL Talos was dropped for its looks.

## How it is made

| Part | How |
|---|---|
| Robots | Real meshes and kinematics from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) / [unitree_ros](https://github.com/unitreerobotics/unitree_ros), [DEEP Robotics URDF_model](https://github.com/DeepRoboticsLab/URDF_model), [Fourier Wiki-GRx-Models](https://github.com/FFTAI/Wiki-GRx-Models), [Drake's Atlas](https://github.com/RobotLocomotion/models) and the [ROS-Industrial Robotiq gripper](https://github.com/ros-industrial-attic/robotiq) (Atlas' hands; the Drake model only has a convex hull of it), MJCF or URDF. Everything model-specific (joints, hands, standing pose, wheels) is a profile in `profiles.py`; body proportions, walking height, the deepest squat the joint limits allow, the turn per step and a carry pose the arms can hold are measured from the models. The models' own lights and floors are removed. Wheels are added at the ANYmal feet; the other wheeled robots have their own. |
| Collapse | MuJoCo rigid-body simulation: box–box contacts, friction, toppling, chain reactions. The quadruped's joint positions *and velocities* are imposed every 1 ms step, so its wheels, legs and belly push the blocks physically. |
| Quadruped choreography | Spline paths with curvature/acceleration-limited speed, body lean and braking pitch. A ballistic backflip (crouch, push-off, 0.78 s flight with tucked legs, damped landing). Wheel spin is integrated from each wheel's ground travel. The walking robots trot: footholds are placed where each hip will be half a stance later, so stance feet never slide. Leg IK at 240 Hz, with a check that keeps every knee bending the way it started. |
| Humanoid walking | Footstep planner: the stance foot never slides. Swing arcs with toe-off and heel-strike, pelvis bob and sway, counter-swinging arms. It turns while walking. |
| Humanoid IK | Damped least squares per limb: 6-DoF sole pose for the legs, full 6-DoF hand pose for 7-DoF arms while grasping. A posture term in the remaining freedom keeps elbows low and close to the body and wrists straight. Same seed every frame, so no flicker. |
| Grasping | Two-handed, on opposite side faces near the top of the block, fitted to each hand: a tripod of thumb and fingertips (G1 Dex3), the palm and curled fingers (Apollo), a fingertip pinch from above (Atlas' Robotiq grippers), or the centre of the hand or fist pressed against the side (all others). Fingers point down near the floor and forward when carrying, forearms level. Open hands reach to just outside the faces, close in, squeeze; on release the fingers loosen before the hands withdraw. The robot picks its standing spot and grasp side so the hands do not run into neighbouring blocks, clears a block out of the way first when needed, and fills cells in an order that leaves room for the hands. A lying block pivots upright between the hands while lifting. |
| Rebuild planning | Week-by-week task planner. Walking paths are shortest paths over a cost map that keeps both feet clear of the debris where there is room. Several standing spots are tried per pick and place, and in the scenes shown in real time the planner also tries the next-best actions until it finds one whose footsteps stay off the blocks. Moves blocks that lie on target cells and unstacks covered blocks first. Keeps nearby debris for the last three cells so the real-time finale stays short. |
| Pacing | Real time (1×) for the crash, the first two placements and the last three. The rest is a speed-ramped time-lapse (several hundred ×) with motion blur and a camera-style badge; a long walk home at the end is fast-forwarded too. |
| Look | Concrete floor, printed calendar mat (months, weekdays, title, legend all in the scene), warm key and cool fill light, soft shadows, light colour grade and vignette, security-camera overlay (CAM 2, date, clock, time-lapse badge). |

## Outputs (`out/`)

- `commit-robots.mp4`: 960×420, 20 fps, about 65–85 s depending on the robots, seamless loop
- `commit-robots.gif`, `commit-robots.webp`: the same, encoded down until they are at most 8 MB
  (720 px / 15 fps first, then 12 fps, then 640 px), so GitHub's image proxy serves them
- `physics.pkl`, `sim.pkl`: cached crash simulation and rebuild plan (`--reuse`, `--reuse-plan`)

## Run locally

```sh
pip install -r requirements.txt          # also needs ffmpeg on PATH
sh tools/fetch_models.sh                 # robot models, about 950 MB (licences listed in the script)

python make_anim.py                      # sample calendar, robots of the day
python make_anim.py --preview            # stills of the key moments
GITHUB_TOKEN=... python make_anim.py --user <login>   # your real calendar
python make_anim.py --humanoid n1 --quadruped spot     # pick the robots
python make_anim.py --date 2026-12-24                  # the robots of another day
python make_anim.py --fast                             # lighter shadows and anti-aliasing (much quicker)
python make_anim.py --stage plan                       # physics + plan only
python make_anim.py --reuse --reuse-plan --resume      # continue an interrupted render
python make_anim.py --reuse --reuse-plan --fast --part 0/4   # render every 4th frame (parallel jobs)
python make_anim.py --encode-only                      # encode the frames in out/frames
```

## On your profile

This repository is meant to be your profile repository (`<login>/<login>`); its `README.md` is the profile page.

1. Push it to a public repository named after your GitHub username.
2. Run the `commit-robots` workflow once from the Actions tab. After that it runs every night and pushes the
   animation to the `output` branch. It has three stages: simulate + plan, render in four parallel jobs (GitHub's
   runners have no GPU, so MuJoCo renders in software, with lighter shadows and anti-aliasing), then encode and
   publish. The robot models are downloaded once and cached.
3. The animation links in `README.md` point to `clsti/clsti`; change them if you use this under another username.
4. The block between `website:start` and `website:end` in `README.md` is copied from the personal website
   every night by the `profile` workflow (`tools/sync_profile.py`: About Me, current position and a link to the
   project overview). Edit the website, not that block; run the workflow from the Actions tab to sync
   right away. It also re-enables both workflows every night, because GitHub pauses scheduled workflows of
   public repositories after 60 days without activity.

The default workflow token only sees public contributions. To include private ones, add a personal access token
with `read:user` as the repository secret `CONTRIB_TOKEN`.

`commit-robots.webp` is smaller and keeps full colour; browsers that cannot play it get the GIF.

The robot models are not part of this repository; they keep their own licences (listed in
`tools/fetch_models.sh`). The Fourier GR-3 model is GPL-3.0, the others are BSD/Apache-style.

## Files

| File | Purpose |
|---|---|
| `make_anim.py` | Pipeline: data → physics → planning → time remap → render → grade → encode |
| `scene.py` | MJCF scene: floor, printed mat, bevelled blocks, lights |
| `profiles.py` | Robot profiles (model-specific joints, hands, poses, wheels) and the robots-of-the-day rota |
| `robots.py` | Model preparation (sole/hand sites, wheels, Robotiq grippers), body measurements, humanoid and quadruped IK |
| `quadruped.py` | Quadruped choreography: wheeled (slalom, donut, backflip) and walking (trot, hop turn, dance), play-bow, exit |
| `humanoid.py` | Humanoid motion planning: footsteps, walk / pick / carry / place, facepalm |
| `rebuild.py` | Rebuild task planner, path planner, standing-spot search, finale reservation |
| `textures.py` | Procedural concrete and the printed calendar mat |
| `contrib.py` | Sample calendar or GitHub GraphQL calendar |
| `common.py` | Small math helpers |
| `tools/` | `fetch_models.sh` (all robot models), `gltf2obj.py` / `dae2obj.py` (glTF / COLLADA meshes to OBJ for MuJoCo), `sync_profile.py` (website → README) |
| `.github/workflows/commit-robots.yml` | The nightly build |
| `.github/workflows/profile.yml` | The nightly website → README sync |
