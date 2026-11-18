//go:build webrtc_src_oracle && !webrtc_oracle

package webrtcoracle

import (
	"debug/elf"
	"debug/macho"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// srcLaneNames are the GOAEC3_EXPECT_LANE values, by Aec3Optimization.
var srcLaneNames = map[aec3.Aec3Optimization]string{
	aec3.OptNone: "none", aec3.OptSse2: "sse2", aec3.OptAvx2: "avx2", aec3.OptNeon: "neon",
}

// srcCheckLane verifies that Go and the C++ build run the same lane, that
// the lane is the one CI asked for (GOAEC3_EXPECT_LANE), and that the CPU
// can run the C++ AVX2 kernels, which use FMA.
func srcCheckLane() error {
	goLane := aec3.DetectOptimization()
	cppLane := SrcLane()
	if goLane != cppLane {
		return fmt.Errorf("Go runs lane %s, the C++ build %s", srcLaneNames[goLane], srcLaneNames[cppLane])
	}
	if want := os.Getenv("GOAEC3_EXPECT_LANE"); want != "" && want != srcLaneNames[goLane] {
		return fmt.Errorf("lane is %s, GOAEC3_EXPECT_LANE=%s", srcLaneNames[goLane], want)
	}
	if goLane == aec3.OptAvx2 && !SrcCPUHasFMA() && os.Getenv("GOAEC3_FORCE_FMA") != "1" {
		return fmt.Errorf("the AVX2 lane is selected but the CPU does not report FMA, which the C++ " +
			"AVX2 kernels use; run the SSE2 lane with GODEBUG=cpu.avx2=off, or set GOAEC3_FORCE_FMA=1 " +
			"where FMA executes without being reported (Rosetta 2)")
	}
	return nil
}

// srcLibmImports lists the libm functions the test binary imports that the
// oracle interposes (src_libm_interpose.c): any of them would mean a C++
// call reaches the host libm instead of the Go helpers.
func srcLibmImports() ([]string, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, err
	}
	var syms []string
	if runtime.GOOS == "darwin" {
		f, err := macho.Open(exe)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		if syms, err = f.ImportedSymbols(); err != nil {
			return nil, err
		}
	} else {
		f, err := elf.Open(exe)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		imp, err := f.ImportedSymbols()
		if err != nil {
			return nil, err
		}
		for _, s := range imp {
			syms = append(syms, s.Name)
		}
	}
	interposed := map[string]bool{"exp2f": true, "__exp10f": true, "exp10f": true, "log10f": true,
		"powf": true, "sin": true, "cos": true, "pow": true, "exp2": true, "exp10": true, "log10": true,
		"expf": true, "logf": true, "log2f": true, "sinf": true, "cosf": true}
	var bad []string
	for _, s := range syms {
		if interposed[strings.TrimPrefix(s, "_")] {
			bad = append(bad, s)
		}
	}
	return bad, nil
}

func TestMain(m *testing.M) {
	if err := srcCheckLane(); err != nil {
		fmt.Fprintln(os.Stderr, "webrtc_src_oracle:", err)
		os.Exit(1)
	}
	if bad, err := srcLibmImports(); err != nil || len(bad) > 0 {
		fmt.Fprintln(os.Stderr, "webrtc_src_oracle: libm imports", bad, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "webrtc_src_oracle: %s/%s lane %s (cpu_info: %s, FMA %v)\n", runtime.GOOS,
		runtime.GOARCH, srcLaneNames[aec3.DetectOptimization()], srcLaneNames[SrcCPUInfoLane()], SrcCPUHasFMA())
	os.Exit(m.Run())
}

// TestSrcLaneMatchesCPUInfo checks that Go's DetectOptimization picks the
// lane libwebrtc's own CPU detection (rtc_base/cpu_info.cc) picks on this
// machine. A GODEBUG cpu override moves the Go lane on purpose; under
// Rosetta 2, Go's internal/cpu reports the AVX2 that Rosetta executes while
// CPUID (which cpu_info.cc reads) does not, so neither is comparable.
func TestSrcLaneMatchesCPUInfo(t *testing.T) {
	if strings.Contains(os.Getenv("GODEBUG"), "cpu.") {
		t.Skip("GODEBUG overrides the CPU features")
	}
	if srcRosetta() {
		t.Skip("Rosetta 2: CPUID hides the AVX2 Go detects")
	}
	if purego {
		t.Skip("the purego build always runs the scalar lane")
	}
	if g, c := aec3.DetectOptimization(), SrcCPUInfoLane(); g != c {
		t.Fatalf("Go selects %s, libwebrtc's cpu_info %s", srcLaneNames[g], srcLaneNames[c])
	}
}
