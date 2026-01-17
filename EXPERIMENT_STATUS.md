# Current Experiments Running

## 1. Universal Intersection Test (CORRECT) ✓ PRIORITY

**Status:** Running (40% complete - model 20/50 on dataset 1/2)

**What it tests:**
- Do signal and noise tasks share a common subspace?
- This is the CORRECT test of universality!

**Expected completion:** 30-45 minutes

**Will answer:**
- Intersection dimension between tasks
- Whether tasks use same or different subspaces
- The TRUE universal subspace dimension (if it exists)

---

## 2. Dimension Saturation Test (300 models)

**Status:** Running in background

**What it tests:**
- Does dimension keep growing or saturate?
- Single-task manifold dimension at large scale

**Expected completion:** 2-3 hours

**Will answer:**
- Whether 72D is real or artifact of undersampling
- If dimension → 200D+, we're filling the space
- If dimension → 80-120D, real manifold exists

---

## Priority:

**WAIT FOR:** Universal Intersection Test (30-45 min)
- This is the key experiment!
- Answers the actual question about universality

**Then check:** Saturation Test (2-3 hours)
- Confirms if single-task dimension is real
- Less critical than intersection test

---

## What We Know So Far:

1. **Single task (100 models):** ~72D effective dimension
2. **But:** Dimension grows with sample size (not saturated)
3. **Compared to random:** 11.5% more compressed
4. **Single basin:** All models from same task converge together

## What We're About To Learn:

1. **Do different tasks share a subspace?** ← Intersection test
2. **What's the true single-task dimension?** ← Saturation test

