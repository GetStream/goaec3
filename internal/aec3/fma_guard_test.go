package aec3

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// fmaAllowed lists the functions whose code may contain fused float32
// multiply-adds: kernels that mirror an FMA of the C++, and fusions that are
// exact.
var fmaAllowed = map[string]string{
	"internal/aec3.dlyAxpyFma":                        "AVX2 matched filter update (_mm256_fmadd_ps)",
	"internal/aec3.(*reverbDecayState).estimateDecay": "2*tailGain is exact",
}

// TestNoUnintendedFMA compiles the signal-path packages for arm64 and fails
// on a fused float32 multiply-add outside fmaAllowed. libwebrtc is built
// with -ffp-contract=off, and other tests catch a fusion only when the
// operands make it visible.
func TestNoUnintendedFMA(t *testing.T) {
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
	module := strings.TrimSuffix(reflect.TypeOf(FftData{}).PkgPath(), "/internal/aec3")
	fused := regexp.MustCompile(`\bF(N)?M(ADD|SUB)S\b|\bVFML[AS]\b`)
	fnRe := regexp.MustCompile(`^(\S+) STEXT`)
	for _, exp := range []string{"", "simd"} {
		for _, tags := range []string{"", "purego"} {
			cmd := exec.Command("go", "build", "-tags", tags, "-gcflags=-S",
				"./internal/aec3", "./internal/ooura", "./bands")
			cmd.Dir = root
			cmd.Env = append(os.Environ(), "GOARCH=arm64", "GOOS=darwin", "GOEXPERIMENT="+exp)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go build (exp=%q tags=%q): %v\n%s", exp, tags, err, out)
			}
			bad := map[string]bool{}
			fn := ""
			for _, line := range strings.Split(string(out), "\n") {
				if m := fnRe.FindStringSubmatch(line); m != nil {
					fn = m[1]
					continue
				}
				if fused.MatchString(line) {
					short := strings.TrimPrefix(fn, module+"/")
					if _, ok := fmaAllowed[short]; !ok {
						bad[short] = true
					}
				}
			}
			if len(bad) > 0 {
				names := make([]string, 0, len(bad))
				for n := range bad {
					names = append(names, n)
				}
				sort.Strings(names)
				t.Errorf("exp=%q tags=%q: unintended fused float32 ops in:\n  %s\nwrap the product in float32(...)",
					exp, tags, strings.Join(names, "\n  "))
			}
		}
	}
}
