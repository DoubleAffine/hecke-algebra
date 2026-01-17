#!/bin/bash
# Live monitoring of the intersection experiment

clear
echo "=========================================="
echo " INTERSECTION EXPERIMENT MONITOR"
echo "=========================================="
echo ""
echo "Press Ctrl+C to exit (experiment continues in background)"
echo ""

while true; do
    # Clear screen and show header
    tput cup 5 0
    tput ed  # Clear from cursor to end of screen

    echo "Current Time: $(date '+%H:%M:%S')"
    echo ""

    # Show latest progress from log
    if [ -f "experiments/current/intersection_proper.log" ]; then
        echo "=========================================="
        echo " PROGRESS"
        echo "=========================================="
        tail -n 1 experiments/current/intersection_proper.log 2>/dev/null
        echo ""
        echo ""
    fi

    # Show configuration if available
    if [ -f "experiments/current/intersection_proper/config.json" ]; then
        echo "=========================================="
        echo " CONFIGURATION"
        echo "=========================================="
        cat experiments/current/intersection_proper/config.json 2>/dev/null | head -20
        echo ""
    fi

    # Show results if experiment is complete
    if [ -f "experiments/current/intersection_proper/results.json" ]; then
        echo "=========================================="
        echo " RESULTS (EXPERIMENT COMPLETE!)"
        echo "=========================================="
        cat experiments/current/intersection_proper/results.json 2>/dev/null
        echo ""
        echo "Experiment finished! Exiting monitor..."
        exit 0
    fi

    # Wait before refresh
    sleep 2
done
