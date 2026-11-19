package ooura

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// The FFT passes index a fixed-size frame through windows whose bounds the
// compiler proves (AGENTS.md, "Memory safety: no unsafe"). boundsCheckFree
// lists the functions that must compile without a single bounds check, per
// GOEXPERIMENT setting.
var boundsCheckFree = map[string][]string{
	"": {
		"cft1stFastLane", "cftmdlFastLane", "cft1stFastScalar", "cftmdlFastScalar",
		"cftfsubTailFast", "cftbsubTailFast", "rftfsubFast", "rftbsubFast",
		"fftFastLane", "inverseFftFastLane", "fftFastScalar", "inverseFftFastScalar",
	},
	"simd": {
		"bitrv2Cft1stSimd", "cftmdlSimd", "cftfsubTailSimd", "cftbsubTailSimd",
		"rftfsubSimd", "rftbsubSimd", "fftSimd", "inverseFftSimd",
	},
	"simd/amd64": {
		"bitrv2Cft1stPair", "cftmdlPair", "cftfsubTailPair", "cftbsubTailScatter",
		"rftfsubScatter", "rftbsubGather", "rftfsubEnd", "rftbsubBegin",
		"fftPairSimd", "inverseFftPairSimd",
	},
}

// TestNoBoundsChecks compiles the package for arm64 and amd64 and fails if
// one of the FFT passes contains a call to a bounds-check panic.
func TestNoBoundsChecks(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles packages")
	}
	if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		t.Skip("no toolchain")
	}
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	panicRe := regexp.MustCompile(`runtime\.panic(Bounds|Index|Slice|ExtendIndex|ExtendSlice)`)
	fnRe := regexp.MustCompile(`^(\S+) STEXT`)
	for _, arch := range []string{"arm64", "amd64"} {
		for _, exp := range []string{"", "simd"} {
			cmd := exec.Command("go", "build", "-gcflags=-S", "./internal/ooura")
			cmd.Dir = root
			cmd.Env = append(os.Environ(), "GOARCH="+arch, "GOOS=linux", "CGO_ENABLED=0", "GOEXPERIMENT="+exp)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go build (arch=%s exp=%q): %v\n%s", arch, exp, err, out)
			}
			checks := map[string]int{}
			seen := map[string]bool{}
			fn := ""
			for _, line := range strings.Split(string(out), "\n") {
				if m := fnRe.FindStringSubmatch(line); m != nil {
					fn = m[1][strings.LastIndex(m[1], ".")+1:]
					seen[fn] = true
					continue
				}
				if panicRe.MatchString(line) && (strings.Contains(line, "CALL") || strings.Contains(line, "JMP")) {
					checks[fn]++
				}
			}
			want := append([]string(nil), boundsCheckFree[""]...)
			if exp == "simd" {
				want = append(want, boundsCheckFree["simd"]...)
				want = append(want, boundsCheckFree["simd/"+arch]...)
			}
			var bad []string
			for _, name := range want {
				if !seen[name] {
					t.Errorf("arch=%s exp=%q: function %s not found in the compiler output", arch, exp, name)
				}
				if checks[name] > 0 {
					bad = append(bad, name)
				}
			}
			if len(bad) > 0 {
				sort.Strings(bad)
				t.Errorf("arch=%s exp=%q: bounds checks in:\n  %s", arch, exp, strings.Join(bad, "\n  "))
			}
		}
	}
}
