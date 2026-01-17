#!/bin/bash
echo "=========================================="
echo " DISK USAGE ANALYSIS"
echo "=========================================="
echo ""

echo "Repository total size:"
du -sh . 2>/dev/null

echo ""
echo "Breakdown by directory:"
du -sh */ 2>/dev/null | sort -rh | head -20

echo ""
echo "Large files (>10MB):"
find . -type f -size +10M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}' | sort -rh

echo ""
echo "Results directories:"
for dir in results* archive final_results experiments; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
        count=$(find "$dir" -name "*.npy" -o -name "*.npz" 2>/dev/null | wc -l)
        echo "  $dir: $size ($count weight files)"
    fi
done

echo ""
echo "Data cache:"
if [ -d "data_cache" ]; then
    du -sh data_cache/ 2>/dev/null
    ls -lh data_cache/ 2>/dev/null | tail -n +2
else
    echo "  No data_cache directory"
fi

echo ""
echo "Hidden datasets in src or other locations:"
find src -name "*.csv" -o -name "*.pkl" -o -name "*.npy" 2>/dev/null | while read f; do
    ls -lh "$f"
done

echo ""
echo "=========================================="
echo " CLEANUP RECOMMENDATIONS"
echo "=========================================="
echo ""

# Check what we need vs what we can delete
echo "For this experiment we only need:"
echo "  - The framework (src/)"
echo "  - No pre-downloaded datasets (we download on-the-fly and delete)"
echo ""

echo "Can safely delete:"
total_deletable=0

if [ -d "archive" ]; then
    size=$(du -s archive 2>/dev/null | awk '{print $1}')
    total_deletable=$((total_deletable + size))
    echo "  ✓ archive/ - old invalid experiments"
fi

if [ -d "results" ]; then
    size=$(du -s results 2>/dev/null | awk '{print $1}')
    total_deletable=$((total_deletable + size))
    echo "  ✓ results/ - early experiments"
fi

for dir in results_*; do
    if [ -d "$dir" ] && [ "$dir" != "results_saturation" ]; then
        size=$(du -s "$dir" 2>/dev/null | awk '{print $1}')
        total_deletable=$((total_deletable + size))
        echo "  ✓ $dir - old experiments"
    fi
done

# Convert KB to human readable
if [ $total_deletable -gt 1048576 ]; then
    deletable_gb=$(echo "scale=2; $total_deletable/1048576" | bc)
    echo ""
    echo "Total deletable: ~${deletable_gb}GB"
elif [ $total_deletable -gt 1024 ]; then
    deletable_mb=$(echo "scale=2; $total_deletable/1024" | bc)
    echo ""
    echo "Total deletable: ~${deletable_mb}MB"
else
    echo ""
    echo "Total deletable: ${total_deletable}KB"
fi

echo ""
echo "Keep (needed for current work):"
echo "  ✓ src/ - framework"
echo "  ✓ final_results/ - valid results"
echo "  ✓ experiments/ - current work"
echo "  ✓ results_saturation/ - still running"
echo ""
