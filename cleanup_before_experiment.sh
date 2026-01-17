#!/bin/bash
echo "=========================================="
echo " PRE-EXPERIMENT CLEANUP"
echo "=========================================="
echo ""

# Create backup list before deletion
echo "Creating deletion log..."
find archive results results_attraction results_basin_discovery results_distant_convergence -type f 2>/dev/null | wc -l > /tmp/cleanup_files_count.txt

echo "Files to be deleted: $(cat /tmp/cleanup_files_count.txt)"
echo ""

echo "Deleting old experiments..."

# Delete old experiments (keep archive folder structure for documentation)
rm -rf results/
rm -rf results_attraction/
rm -rf results_basin_discovery/
rm -rf results_distant_convergence/

# Keep archive but compress it
if [ -d "archive" ]; then
    echo "Compressing archive..."
    tar -czf archive_old_experiments.tar.gz archive/
    rm -rf archive/
    echo "  ✓ Created archive_old_experiments.tar.gz"
fi

echo ""
echo "Cleanup complete!"
echo ""

# Show new size
echo "Repository size after cleanup:"
du -sh .

echo ""
echo "Ready for new experiment!"
echo ""
