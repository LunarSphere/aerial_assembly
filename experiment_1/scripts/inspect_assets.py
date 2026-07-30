from aerial_gripper_sim.cli import main

raise SystemExit(main(["inspect-assets", *(__import__("sys").argv[1:])]))
