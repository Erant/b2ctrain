#!/bin/bash
# Reproduce b2crunner's warm-start invocations (polish: growth off, refine never) against a dataset + trained ply.
set -e
DATASET=$1; PLY=$2; OUT=$3; BIN=${4:-$(dirname $0)/../build/b2ctrain}
mkdir -p $OUT/ds
for f in cameras.txt images.txt points3D.txt images masks normals weights; do [ -e $DATASET/$f ] && ln -sfn $(realpath $DATASET/$f) $OUT/ds/$f; done
ln -sfn $(realpath $PLY) $OUT/ds/init.ply
$BIN $OUT/ds --total-train-iters 9000 --sh-degree 3 --export-path $OUT --export-name polished.ply --export-every 9000 --max-resolution 1920 --max-splats 10000000 --refine-every 1000000 --match-alpha-weight 0.5 --growth-stop-iter 0 --normalize-masked-loss --normal-loss-weight 0.05 --normal-loss-start-iter 0 --normal-loss-every 1 --export-evidence --sparse-adam
