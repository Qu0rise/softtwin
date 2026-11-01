# Contributing

Issues and pull requests are welcome (bug reports, a real-hardware log, a better tactile model, an Isaac Sim fix).

1. Fork → create a branch → make the change.
2. `pip install -r requirements.txt pytest ruff`
3. `ruff check . && pytest -q` must pass (CI runs the same: lint, dataset validator, metrics recompute, a headless MuJoCo episode, the ROS 2 node and the MCP bridge with stand-ins).
4. If you change a reported number, re-run the script that produces it and commit the updated `results/*.json` together with the code.
5. Do not open a PR with code you have not run yourself.

Good first issues: replace the synthetic actuator data with a real log (`learning/actuator_net.py --csv`), run `ros2/softtwin_grasp_node.py` on a real ROS 2 Jazzy install,
add a softer object than tofu to the dataset (the learned force window cannot extrapolate, see `docs/evaluation_protocol.md`), port the soft contact to Isaac Sim deformable bodies.
