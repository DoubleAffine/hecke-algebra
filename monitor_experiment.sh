#!/bin/bash
# Monitor running experiments in real-time

EXPERIMENT_DIR=${1:-"experiments/current/intersection_proper"}

echo "=========================================="
echo " EXPERIMENT MONITOR"
echo "=========================================="
echo ""
echo "Monitoring: $EXPERIMENT_DIR"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=========================================="
    echo " EXPERIMENT STATUS - $(date '+%H:%M:%S')"
    echo "=========================================="
    echo ""

    # Check if experiment is running
    if pgrep -f "run_proper_intersection" > /dev/null; then
        echo "Status: ✓ RUNNING"
    else
        echo "Status: ✗ NOT RUNNING"
    fi

    echo ""

    # Show configuration
    if [ -f "$EXPERIMENT_DIR/config.json" ]; then
        echo "Configuration:"
        python3 -c "import json; cfg=json.load(open('$EXPERIMENT_DIR/config.json')); print(f\"  Models per task: {cfg['models_per_task']}\"); print(f\"  Tasks: {', '.join(cfg['tasks'])}\"); print(f\"  Total models: {cfg['models_per_task'] * len(cfg['tasks'])}\"); print(f\"  Started: {cfg['started'][:19]}\")"
        echo ""
    fi

    # Count completed models
    if [ -d "$EXPERIMENT_DIR" ]; then
        weight_files=$(ls $EXPERIMENT_DIR/weights_*_batch_*.npy 2>/dev/null | wc -l)
        echo "Progress:"
        echo "  Weight batches saved: $weight_files"

        # Check for results
        if [ -f "$EXPERIMENT_DIR/results.json" ]; then
            echo ""
            echo "✓ EXPERIMENT COMPLETE"
            echo ""
            echo "Results:"
            python3 -c "
import json
r = json.load(open('$EXPERIMENT_DIR/results.json'))
print(f\"  Global dimension: {r['intersection']['global_dimension']:.1f}D\")
print(f\"  Intersection estimate: {r['intersection']['intersection_estimate']:.1f}D\")
print(f\"  Mean angle: {r['intersection']['mean_angle']:.1f}°\")
"
            break
        fi
    fi

    echo ""
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
    sleep 5
done

echo ""
echo "Monitoring stopped."
