# B2 + Z1 whole-body grasp integration

Source: luyuqi-tim/ATEC_B branch-Mink_IK1 at 3bc5822401bd82e00f49b0c6dbb518f9d46d64ba. Historical nine-case report: `history/report_20261026.md`; full historical evidence remains at source main eab3a9754ef93027bfd719aebcf68b36f5963d90. Historical results are not validation of this server or the team's basket model.

This folder contains the original experiments, controllers, configs and diagnostic evaluator. The team `taskb_student` package and assets remain shared. The team's default URDF includes the 0.2 kg v3 rear basket; its center of mass and collision geometry differ from the prior no-basket tests. Path audit is not implemented in this integration and is not run by the nine-case entry. The bench is a known-target diagnostic, not a perception/navigation/basket-loading integration or a competition submission policy.

On this server use `/opt/conda/envs/isaaclab/bin/python`; Isaac Sim 5.1, Isaac Lab 2.3.2, Mink 1.0.0, MuJoCo 3.3.7, qpsolvers 4.8.2 and DAQP 0.7.2. The old `requirements.lock.txt` is a historical snapshot containing old absolute paths, not an install instruction. `pip check` reports existing packaging version conflicts for ray and wheel; their effect on this workflow has not been established.

From the repository root, run a new nine-case set only when authorized:

```bash
bash grasp/run_nine.sh /absolute/path/to/new-output-directory
```

This entry freezes source and shared asset hashes, runs one attempt per object/yaw at 0.55 m, checks physical lift, contact and HOLD and stops on first physical failure. Path audit is not run; each result records `path_audit_status=not_implemented` and `path_clear=null`. Output must not exist. Do not use older multi-case scripts to satisfy the stop-on-first-failure requirement: they may continue after failures. Records belong outside the repository; preserve all failures. Never treat IK reachability, a zero exit status, or historical results alone as grasp success.

The original code uses known target and grasp poses; vision, navigation handover, placement into the basket, loaded locomotion and physical robot use remain unverified. The separate team basket branch must be reviewed before interface integration.

## Path audit is not implemented

The legacy path-audit implementations and suite were removed from this integration at the owner's request. The first basket-model test's original 180-second timeout and all physical evidence remain in its output; its historical outcome is not rewritten. The original source and history remain accessible at the source repository. A future team implementation would need to cover robot, floor and basket geometry and be checked against known clear and colliding trajectories. Until then `path_clear=null` means unknown, not safe. Physical success does not establish self-collision avoidance or basket clearance.
