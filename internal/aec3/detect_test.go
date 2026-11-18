package aec3

import "testing"

// TestDetectOptimizationForcedLane checks the GOAEC3_FORCE_LANE override and
// that a lane the architecture cannot run is ignored.
func TestDetectOptimizationForcedLane(t *testing.T) {
	native := DetectOptimization()
	for _, tc := range []struct {
		env  string
		want Aec3Optimization
		ok   bool
	}{
		{"none", OptNone, true},
		{"sse2", OptSse2, isAMD64},
		{"avx2", OptAvx2, isAMD64},
		{"neon", OptNeon, isARM64},
		{"bogus", native, false},
	} {
		t.Setenv("GOAEC3_FORCE_LANE", tc.env)
		want := native
		if tc.ok && !purego {
			want = tc.want
		}
		if got := DetectOptimization(); got != want {
			t.Errorf("GOAEC3_FORCE_LANE=%s: got %v, want %v", tc.env, got, want)
		}
	}
}
