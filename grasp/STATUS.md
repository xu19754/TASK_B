# Work in progress - 2026-10-27

This integration is incomplete. Source baseline: luyuqi-tim/ATEC_B branch-Mink_IK1, 3bc5822401bd82e00f49b0c6dbb518f9d46d64ba; historical nine-case report from main eab3a9754ef93027bfd719aebcf68b36f5963d90 is not a result on this server.

- Server environment: Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1, Mink 1.0.0, MuJoCo 3.3.7, qpsolvers 4.8.2, DAQP 0.7.2. pip check reports packaging 23.0 conflicts with ray and wheel; effect on grasp is not established.
- The team uses the v3 basket URDF; no shared team assets or code were modified. The basket branch must be coordinated for integration.
- Known-target grasp test, mustard yaw -45: one physical attempt; rise 0.1473 m, independent valid hold 5.86 s, native contacts present, original path audit timed out. Video and raw evidence on server at /root/grasp-validation/nine-basket-20261027/. The former path audit was removed at owner's request; no collision clearance is claimed. Remaining eight physical cases have not run.
- Basket-transfer work is exploratory only: `basket_transfer/ik_preflight.py` is not integrated into the controller. Initial lift target failed position/orientation tolerances (0.051 m / 38.9 degrees); fixed-orientation retraction failed (0.0128 m / 8.73 degrees); free-orientation retraction reached its position but the subsequent lift failed (0.224 m / 44.0 degrees); vertical-tool candidate also failed the lift (0.0821 m / 37.6 degrees). Outputs and earlier script snapshots are in /root/grasp-validation/basket-ik-preflight-*. No physical basket transfer, release or retention test has been performed.
- The old nine-case run retains its first timeout as historical evidence. Any future sequence must not repeat that physical combination without the owner's permission; each next physical case requires the preceding physical validation to pass. The current nine-case script starts a fresh sequence and is not a resume mechanism.
- Before deployment, develop and validate a carry-and-release state machine, contact/retention checks and basket clearance in collaboration with the basket owner. Do not infer safety or success from IK alone.

## Subsequent basket-transfer diagnostic

After the owner authorized retries, attempts 01-09 failed and are retained alongside attempt 10. Attempt 10 (sugar box, yaw 0) physically grasped, carried, released, and remained in the v3 basket for 6.04 s; independent trace checks are in `basket_transfer/evidence/attempt_10_independent.json`. Full traces and video remain at `/root/grasp-validation/basket-physical-10/`. This is one known-target simulation, not validated collision-free planning or a complete task policy; see `basket_transfer/README.md`.
