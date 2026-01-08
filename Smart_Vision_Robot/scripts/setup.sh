#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Creating project skeleton at $ROOT"

mkdir -p "$ROOT"/{data,env,agents,training,utils,models,scripts}
touch "$ROOT"/env/simulation_env.py
touch "$ROOT"/agents/dqn_agent.py
touch "$ROOT"/agents/replay_buffer.py
touch "$ROOT"/training/train_agent.py
touch "$ROOT"/training/evaluate.py
touch "$ROOT"/training/play.py
touch "$ROOT"/utils/metrics.py
touch "$ROOT"/requirements.txt
touch "$ROOT"/README.md

cat > "$ROOT"/requirements.txt <<'EOF'
tensorflow>=2.10
numpy
opencv-python
matplotlib
tqdm
EOF

echo "Project skeleton created. Edit files and then run training with:"
echo "  python training/train_agent.py"
