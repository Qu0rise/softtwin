# SoftTwin tactile grasp dataset (synthetic, auto-annotated)

- episodes: 420 (7 object types incl. 2 look-alikes), generated in 3.8 min on 2 CPU cores; seeds 1000 and up
  (disjoint from the benchmark seeds 0-29)
- every episode: firmness plunger probe (load + encoder, 50 Hz), finger probe to 0.8 N, then one grasp with a random target force
  F* ~ U(0.5, 4.0) N and no regrasp
- control-rate samples (50 Hz, grip to lower): 156423; each has pressure command / pressure / model free bend / measured and true
  bend of the 3 fingers / estimated and true grip force / 24 skin taxels (sensor model) / 24 true taxel forces
- images per episode: wrist RGB + depth + segmentation mask (pre-grasp), scene RGB + mask (grasp, lift); 192x144
- labels per episode: object, looks-like, true handling class, visual prior class, allowable force, hidden physics
  (mass, friction, contact stiffness [N/m], yaw), target force, outcome (success 45%, crushed 22%,
  slipped 14%, dropped 19%, no contact 0%), bounding boxes, feature vectors
- files: `index.jsonl` / `index.parquet` (one row per episode), `episodes/ep_XXXXX.npz`
- all annotations come from the simulator; sensor values pass through the sensor models of `sim_mujoco/twin_core.py`
  (assumptions for a textile piezoresistive skin, a bend sensor and a small load cell + encoder, not calibrated devices)
- skipped episodes (renderer crash): 0
