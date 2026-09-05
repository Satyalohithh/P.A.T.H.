# Feature catalog

Status: frozen (groups A-F). MVP implements A.1-A.7, B.1-B.6, C.1-C.10.

Feature ids are canonical; vector column order is fixed by
`src/safewatch/features/feature_names.py`.

## A. Pose (single-person static)

| Id | Definition |
|---|---|
| A.1 `pose.elbow_angle.L/R` | Interior angle at elbow joint (rad) |
| A.2 `pose.knee_angle.L/R` | Interior angle at knee joint (rad) |
| A.3 `pose.torso_lean` | Lean of spine vs vertical (rad) |
| A.4 `pose.shoulder_orientation` | Shoulder-line vs horizontal (rad) |
| A.5 `pose.upper_asymmetry` | L/R upper-limb extension asymmetry |
| A.6 `pose.complexity` | 1 - (L1 distance to mean pose) / spread |
| A.7 `pose.com_height` | CoM height normalized by H |

## B. Motion (single-person temporal)

| Id | Definition |
|---|---|
| B.1 `motion.{arm,leg}_angular_vel.L/R` | Joint-angle central difference / dt |
| B.2 `motion.{arm,leg}_angular_acc.L/R` | Second central difference / dt^2 |
| B.3 `motion.com_vel.{x,y}` / `motion.com_speed` | CoM translation velocity |
| B.4 `motion.com_acc.{x,y,mag}` | CoM acceleration |
| B.5 `motion.limb_energy` | Σ (limb length · angular velocity)² |
| B.6 `motion.stride_length`, `motion.gait_regularity` | Stride kinematics |

## C. Interpersonal (pair)

| Id | Definition |
|---|---|
| C.1 `inter.distance` | CoM distance scaled by mean H |
| C.2 `inter.approach_rate` | Negative derivative of C.1 |
| C.3 `inter.mutual_approach` | Both persons moving toward each other |
| C.4 `inter.facing_angle` | Angle between facing dirs |
| C.5 `inter.confrontation_index` | Facing × proximity weighting |
| C.6 `inter.energy_ratio` | Aggressor/peer limb-energy ratio |
| C.7 `inter.pose_mirroring` | Synchrony of mirrored joints |
| C.8 `inter.contact_proximity` | Below-gate clip of C.1 (gate = 3.0h) |
| C.9 `inter.reach_vector` | Vector from peer to aggressor grasp region |
| C.10 `inter.retreat_velocity` | Peer's velocity away from subject |

## D. Temporal / dynamics (window-level)
## E. Group (crowd-level statistics)
## F. Risk (contextual priors: zone, time-of-day, repeat offenders)

Groups D-F are scoped after MVP; their ids will be added without reordering
A-C.