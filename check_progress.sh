#!/bin/bash
echo "=== TRAINING PROGRESS ==="
if [ -f large_scale.pid ]; then
    PID=$(cat large_scale.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: RUNNING (PID: $PID)"
    else
        echo "Status: COMPLETED or FAILED"
    fi
fi

echo ""
echo "=== LATEST OUTPUT ==="
tail -30 large_scale_output.log 2>/dev/null || echo "No output yet"

echo ""
echo "=== RESULTS SO FAR ==="
if [ -d results_large_scale ]; then
    ls -lh results_large_scale/ 2>/dev/null | tail -10
else
    echo "No results directory yet"
fi
