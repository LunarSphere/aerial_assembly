from aerial_gripper_sim.cli import main

raise SystemExit(main(["run", *(__import__("sys").argv[1:])]))
