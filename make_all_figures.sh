#!/usr/bin/env bash
# Reproduce all paper figures (except Fig 2 savings, which lives in eamematsro/e2e).
set -e
cd "$(dirname "$0")"

echo "### Fig 1: MTL infers latent context ###"
python ECtoCB_train_MSE_12C.py
python MLP_12c.py
python replot_mtl_geometry.py

echo "### Fig 3: structure learning ###"
python structure_learning_rot.py   --skip-train --baselines --matched-oracle
python structure_learning_rot.py   --skip-train --train-angles 0 90
python structure_learning_drift.py --skip-train --baselines --matched-oracle

echo "### Fig 4: context inference from reward-prediction error ###"
python context_inference_sro_rpe.py --skip-train

echo "### Fig 5: MTL vs ventral hippocampus (motor) ###"
python geometry_section1_vhpc.py
python make_context_3angle_section1.py

echo "### Fig 6: MTL vs HPC/PFC geometry (SRO) ###"
python geometry_sro.py
python geometry_sro_panels.py
python within_context_decode_sro.py

echo "### Done. Outputs under Paper/ and paper/ . ###"
