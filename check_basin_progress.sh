#!/bin/bash
# Check progress of basin discovery experiment

echo "=== Basin Discovery Progress ==="
echo ""

if [ -f "results_basin_discovery/experiment.log" ]; then
    echo "Last 20 lines of log:"
    tail -20 results_basin_discovery/experiment.log
    echo ""

    # Count completed models
    completed=$(grep -c "best_test_accuracy" results_basin_discovery/experiment.log 2>/dev/null || echo "0")
    total=48

    echo "Progress: $completed / $total models completed"

    # Estimate time remaining
    if [ -f "results_basin_discovery/metadata.json" ]; then
        echo ""
        echo "Metadata file exists - checking..."
        models=$(python3 -c "import json; print(len(json.load(open('results_basin_discovery/metadata.json'))))" 2>/dev/null || echo "0")
        echo "Models saved: $models"
    fi
else
    echo "Experiment not started yet or log file not created."
fi

echo ""
echo "To view full log: tail -f results_basin_discovery/experiment.log"
