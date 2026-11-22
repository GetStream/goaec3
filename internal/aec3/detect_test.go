package aec3

import "testing"

// TestDetectOptimizationForcedLane checks that SetForcedLaneForTesting
// forces a lane, refuses one the architecture cannot run, and that
// ClearForcedLaneForTesting restores the detected lane.
func TestDetectOptimizationForcedLane(t *testing.T) {
	t.Cleanup(ClearForcedLaneForTesting)
	native := DetectOptimization()
	for _, tc := range []struct {
		opt Aec3Optimization
		ok  bool
	}{
		{OptNone, true},
		{OptSse2, isAMD64},
		{OptAvx2, isAMD64},
		{OptNeon, isARM64},
	} {
		ClearForcedLaneForTesting()
		if got := SetForcedLaneForTesting(tc.opt); got != tc.ok {
			t.Errorf("SetForcedLaneForTesting(%v) = %v, want %v", tc.opt, got, tc.ok)
		}
		want := native
		if tc.ok && !purego {
			want = tc.opt
		}
		if got := DetectOptimization(); got != want {
			t.Errorf("forced %v: DetectOptimization() = %v, want %v", tc.opt, got, want)
		}
	}
	ClearForcedLaneForTesting()
	if got := DetectOptimization(); got != native {
		t.Errorf("cleared: DetectOptimization() = %v, want %v", got, native)
	}
}
