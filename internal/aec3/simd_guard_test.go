package aec3

import (
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unicode"
	"unicode/utf8"
)

// The tests of this file read the compiler's assembly listing of the amd64
// GOEXPERIMENT=simd build of the signal-path packages (cross-compiled, so
// they run on every host) and prove two properties of the machine code that
// no test on an AVX2 machine can observe:
//
//   - TestSimdInstructionsGated: every vector instruction runs only when the
//     CPU feature it needs was tested. A CPU with AVX but not AVX2 faults on
//     an AVX2 instruction, and Go 1.27 saves only the low 128 bits of the Y
//     registers at an asynchronous preemption on such a CPU, so 256-bit code
//     there computes wrong values without faulting.
//   - TestNoLegacySSEWithDirtyUpperState: no legacy-SSE (non-VEX) instruction
//     and no call runs between a 256-bit instruction and VZEROUPPER, which
//     costs a state transition per instruction on Intel.

// simdFeat is a set of CPU features.
type simdFeat uint8

const (
	featAVX simdFeat = 1 << iota
	featAVX2
	featFMA
	featAll = featAVX | featAVX2 | featFMA
)

func (f simdFeat) String() string {
	var s []string
	for i, n := range []string{"AVX", "AVX2", "FMA"} {
		if f&(1<<i) != 0 {
			s = append(s, n)
		}
	}
	if s == nil {
		return "none"
	}
	return strings.Join(s, "+")
}

type asmInst struct {
	pc   int
	pos  string // file:line
	op   string
	args string
}

func (in *asmInst) String() string { return fmt.Sprintf("%s\t%s %s", in.pos, in.op, in.args) }

type asmFunc struct {
	name  string
	insts []asmInst
}

// asmListing is the parsed output of go build -gcflags=-S.
type asmListing struct {
	funcs map[string]*asmFunc
	names []string
	// dataRefs maps a function to the data symbols that hold its address
	// (function values in tables, interface method tables).
	dataRefs map[string][]string
}

var (
	asmInstRe = regexp.MustCompile(`^\t0x[0-9a-f]+ (\d+) \((.*):(\d+)\)\t(\S+)(?:\t(.*))?$`)
	asmRelRe  = regexp.MustCompile(`^\trel \d+\+\d+ t=(\S+) (.+)\+-?\d+$`)
	asmSymRe  = regexp.MustCompile(`^(.+?) (S[A-Z]+) `)
	asmSBRe   = regexp.MustCompile(`([^\s,()$]+)\(SB\)`)
	// asmNameRe matches a named operand: a symbol or a named stack slot.
	asmNameRe = regexp.MustCompile(`[^\s,()$]+\((SB|SP|FP)\)`)
	asmWideRe = regexp.MustCompile(`\b[YZK]\d+\b`)
	asmYRe    = regexp.MustCompile(`\bY\d+\b`)
	asmXRe    = regexp.MustCompile(`\bX\d+\b`)
)

func parseAsmListing(out string) *asmListing {
	l := &asmListing{funcs: map[string]*asmFunc{}, dataRefs: map[string][]string{}}
	var fn *asmFunc
	data := ""
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		if line[0] != '\t' {
			fn, data = nil, ""
			m := asmSymRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			switch {
			case m[2] == "STEXT":
				fn = &asmFunc{name: m[1]}
				l.funcs[fn.name] = fn
				l.names = append(l.names, fn.name)
			case !strings.HasPrefix(m[2], "SDWARF"):
				data = m[1]
			}
			continue
		}
		if fn != nil {
			m := asmInstRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			switch m[4] {
			case "TEXT", "PCDATA", "FUNCDATA", "NOP":
				continue
			}
			pc, _ := strconv.Atoi(m[1])
			fn.insts = append(fn.insts, asmInst{pc: pc, pos: filepath.Base(m[2]) + ":" + m[3], op: m[4], args: m[5]})
			continue
		}
		if data != "" {
			if m := asmRelRe.FindStringSubmatch(line); m != nil {
				// An unexported method in a type's method table cannot be
				// called through it: R_METHODOFF entries are reachable only
				// by an interface (an R_ADDR in its itab) or by name.
				if m[1] == "R_METHODOFF" {
					continue
				}
				l.dataRefs[m[2]] = append(l.dataRefs[m[2]], data)
			}
		}
	}
	return l
}

var asmNoReturnRe = regexp.MustCompile(`^runtime\.(panic|goPanic|gopanic|throw|fatal)`)

// callee returns the symbol an instruction calls (also as a tail call),
// "?" for an indirect call and "" for other instructions.
func (in *asmInst) callee() string {
	if in.op != "CALL" && in.op != "JMP" {
		return ""
	}
	if m := asmSBRe.FindStringSubmatch(in.args); m != nil {
		return m[1]
	}
	if in.op == "CALL" {
		return "?"
	}
	if _, err := strconv.Atoi(in.args); err != nil {
		return "?" // indirect jump
	}
	return ""
}

// succ returns the indices of the instructions that may run after insts[i]:
// the branch target (-1 if none) and the next instruction (-1 if none).
func (f *asmFunc) succ(i int) (target, next int, err error) {
	in := &f.insts[i]
	target, next = -1, i+1
	if next == len(f.insts) {
		next = -1
	}
	switch {
	case in.op == "RET" || in.op == "UNDEF":
		return -1, -1, nil
	case in.op == "CALL":
		if asmNoReturnRe.MatchString(in.callee()) {
			next = -1
		}
		return -1, next, nil
	case strings.HasPrefix(in.op, "J"):
		pc, aerr := strconv.Atoi(in.args)
		if aerr != nil {
			if in.op == "JMP" && asmSBRe.MatchString(in.args) {
				return -1, -1, nil // tail call
			}
			return 0, 0, fmt.Errorf("%s: %s: unsupported jump", f.name, in)
		}
		// The target may be a NOP, which the parser drops.
		t := sort.Search(len(f.insts), func(k int) bool { return f.insts[k].pc >= pc })
		if t == len(f.insts) {
			return 0, 0, fmt.Errorf("%s: %s: no instruction at the target", f.name, in)
		}
		if in.op == "JMP" {
			next = -1
		}
		return t, next, nil
	}
	return -1, next, nil
}

// simdVexOps classifies the VEX mnemonics the kernels compile to by the
// feature their 128-bit form needs. A mnemonic that is not listed fails the
// test: classify it from the instruction reference before adding it.
var simdVexOps = map[string]simdFeat{
	"VADDPS": featAVX, "VSUBPS": featAVX, "VMULPS": featAVX, "VDIVPS": featAVX,
	"VSQRTPS": featAVX, "VMINPS": featAVX, "VMAXPS": featAVX, "VCMPPS": featAVX,
	"VSHUFPS": featAVX, "VHADDPS": featAVX, "VMOVMSKPS": featAVX,
	"VMOVDQU": featAVX, "VMOVSS": featAVX, "VMOVD": featAVX,
	"VCVTTPS2DQ": featAVX, "VCVTDQ2PS": featAVX,
	"VINSERTF128": featAVX, "VEXTRACTF128": featAVX, "VPERM2F128": featAVX,
	"VZEROUPPER": featAVX,
	"VPBLENDVB":  featAVX, "VPALIGNR": featAVX, "VPSHUFD": featAVX,
	"VPUNPCKHDQ": featAVX, "VPUNPCKLDQ": featAVX, "VPUNPCKHWD": featAVX,
	"VPXOR": featAVX, "VPOR": featAVX, "VPAND": featAVX, "VPANDN": featAVX,
	"VPADDD": featAVX, "VPSUBD": featAVX, "VPCMPGTD": featAVX, "VPCMPEQD": featAVX,
	"VPINSRD": featAVX, "VPEXTRD": featAVX, "VPSRAD": featAVX,
	"VPMOVSXWD": featAVX, "VPACKSSDW": featAVX,
	// VBROADCASTSS from a register is AVX2; from memory it is AVX.
	"VBROADCASTSS": featAVX, "VPBROADCASTD": featAVX | featAVX2,
	"VFMADD213PS": featAVX | featFMA,
}

// needs returns the CPU features an instruction needs. Every instruction on
// a Y register counts as AVX2, whatever its own requirement: without AVX2
// the runtime does not preserve the upper halves across a preemption.
func (in *asmInst) needs() (simdFeat, error) {
	args := asmNameRe.ReplaceAllString(in.args, "")
	wide := asmWideRe.FindString(args)
	if !strings.HasPrefix(in.op, "V") {
		if wide != "" {
			return 0, fmt.Errorf("%s: %s register in a non-VEX instruction", in, wide)
		}
		return 0, nil
	}
	f, ok := simdVexOps[in.op]
	if !ok {
		return 0, fmt.Errorf("%s: unclassified instruction (see simdVexOps)", in)
	}
	if wide != "" {
		if wide[0] != 'Y' {
			return 0, fmt.Errorf("%s: AVX-512 register", in)
		}
		f |= featAVX2
	}
	if in.op == "VBROADCASTSS" && asmXRe.MatchString(strings.SplitN(args, ",", 2)[0]) {
		f |= featAVX2
	}
	return f, nil
}

// simdGates finds the package-level bool variables of the amd64 simd build
// whose initializer is a conjunction of archsimd.X86 feature tests and of
// other such variables, and returns the features each one implies, keyed by
// symbol name.
func simdGates(t *testing.T, root, module string) map[string]simdFeat {
	gates := map[string]simdFeat{}
	tags := func(tag string) bool {
		return tag == "amd64" || tag == "linux" || tag == "unix" || tag == "goexperiment.simd" || tag == "gc"
	}
	for _, pkg := range simdGuardPackages {
		dir := filepath.Join(root, pkg)
		ents, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		inits := map[string]ast.Expr{}
		fset := token.NewFileSet()
		for _, e := range ents {
			name := e.Name()
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") ||
				strings.HasSuffix(name, "_arm64.go") {
				continue
			}
			f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.ParseComments)
			if err != nil {
				t.Fatal(err)
			}
			skip := false
			for _, cg := range f.Comments {
				if cg.Pos() > f.Package {
					break
				}
				for _, c := range cg.List {
					if constraint.IsGoBuild(c.Text) {
						x, err := constraint.Parse(c.Text)
						if err != nil {
							t.Fatal(err)
						}
						skip = !x.Eval(tags)
					}
				}
			}
			if skip {
				continue
			}
			for _, d := range f.Decls {
				gd, ok := d.(*ast.GenDecl)
				if !ok || gd.Tok != token.VAR {
					continue
				}
				for _, s := range gd.Specs {
					vs := s.(*ast.ValueSpec)
					if len(vs.Values) != len(vs.Names) {
						continue
					}
					for i, n := range vs.Names {
						inits[n.Name] = vs.Values[i]
					}
				}
			}
		}
		var eval func(e ast.Expr, depth int) (simdFeat, bool)
		eval = func(e ast.Expr, depth int) (simdFeat, bool) {
			if depth > 8 {
				return 0, false
			}
			switch e := e.(type) {
			case *ast.ParenExpr:
				return eval(e.X, depth)
			case *ast.BinaryExpr:
				if e.Op != token.LAND {
					return 0, false
				}
				a, ok1 := eval(e.X, depth)
				b, ok2 := eval(e.Y, depth)
				return a | b, ok1 && ok2
			case *ast.Ident:
				if init, ok := inits[e.Name]; ok {
					return eval(init, depth+1)
				}
			case *ast.CallExpr:
				sel, ok := e.Fun.(*ast.SelectorExpr)
				if !ok || len(e.Args) != 0 {
					return 0, false
				}
				if x, ok := sel.X.(*ast.SelectorExpr); !ok || x.Sel.Name != "X86" {
					return 0, false
				} else if p, ok := x.X.(*ast.Ident); !ok || p.Name != "archsimd" {
					return 0, false
				}
				switch sel.Sel.Name {
				case "AVX":
					return featAVX, true
				case "AVX2":
					return featAVX | featAVX2, true
				case "FMA":
					return featAVX | featFMA, true
				}
			}
			return 0, false
		}
		for name, init := range inits {
			if f, ok := eval(init, 0); ok {
				gates[module+"/"+pkg+"."+name] = f
			}
		}
	}
	return gates
}

var simdGuardPackages = []string{"internal/aec3", "internal/ooura", "internal/bands"}

type simdGuardData struct {
	root, module string
	listing      *asmListing
	gates        map[string]simdFeat
	err          error
}

var simdGuard = sync.OnceValue(func() *simdGuardData {
	d := &simdGuardData{}
	d.root, d.err = filepath.Abs("../..")
	if d.err != nil {
		return d
	}
	d.module = strings.TrimSuffix(reflect.TypeOf(FftData{}).PkgPath(), "/internal/aec3")
	out, err := simdGuardBuild(d.root, "")
	if err != nil {
		d.err = err
		return d
	}
	d.listing = parseAsmListing(out)
	return d
})

func simdGuardBuild(root, tags string) (string, error) {
	args := []string{"build", "-tags", tags, "-gcflags=-S"}
	for _, p := range simdGuardPackages {
		args = append(args, "./"+p)
	}
	cmd := exec.Command("go", args...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "GOEXPERIMENT=simd", "GOFLAGS=")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("go build (tags=%q): %v\n%s", tags, err, out)
	}
	return string(out), nil
}

func simdGuardSetup(t *testing.T) *simdGuardData {
	if testing.Short() {
		t.Skip("compiles packages")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no toolchain")
	}
	d := simdGuard()
	if d.err != nil {
		t.Fatal(d.err)
	}
	if d.gates == nil {
		d.gates = simdGates(t, d.root, d.module)
	}
	return d
}

// short strips the module path from a symbol name.
func (d *simdGuardData) short(sym string) string {
	return strings.TrimPrefix(sym, d.module+"/")
}

// isRoot reports whether code outside the listing can call a function
// without passing an instruction of the listing: exported functions and
// methods, package initializers, and functions whose address is stored in
// data. Closures are reached through their parent.
func (d *simdGuardData) isRoot(name string) (bool, string) {
	for _, from := range d.listing.dataRefs[name] {
		if from != name+"·f" {
			return true, "its address is stored in " + d.short(from)
		}
	}
	for _, from := range d.listing.dataRefs[name+"·f"] {
		return true, "its address is stored in " + d.short(from)
	}
	s := name[strings.LastIndex(name, "/")+1:]
	s = s[strings.Index(s, ".")+1:]
	if strings.HasPrefix(s, "(") || strings.HasPrefix(s, "[") {
		if i := strings.Index(s, ")."); i >= 0 {
			s = s[i+2:]
		}
	} else if i := strings.Index(s, "."); i >= 0 && !regexp.MustCompile(`^\.(func\d|gowrap|deferwrap|init)`).MatchString(s[i:]) {
		s = s[i+1:] // T.Method
	}
	base, rest, _ := strings.Cut(s, ".")
	if base == "init" || base == "glob" {
		return true, "a package initializer"
	}
	if rest != "" {
		return false, "" // a closure
	}
	if r, _ := utf8.DecodeRuneInString(base); unicode.IsUpper(r) {
		return true, "exported"
	}
	return false, ""
}

// TestSimdInstructionsGated proves that every vector instruction of the
// amd64 simd build runs only after a test of the CPU feature it needs.
//
// The tests it recognizes are branches on the package-level variables that
// simdGates derives from the sources (kernHasAVX2, dlyUseFma, ...): a
// CMPB var(SB), $0 followed by a conditional jump has one edge on which the
// variable is true. An instruction is gated for a feature when every path
// from the function entry to it takes such an edge, that is, when it is
// unreachable once those edges are removed. What a function still needs at
// its entry moves to the instructions that call it or take its address, and
// nothing may be left at a function that code outside the listing can reach.
// Unrecognized patterns count as ungated, so the test fails closed.
func TestSimdInstructionsGated(t *testing.T) {
	d := simdGuardSetup(t)
	l := d.listing
	if len(d.gates) < 10 {
		t.Fatalf("found only %d feature variables: %v", len(d.gates), d.gates)
	}

	type site struct {
		fn    string // function referenced at this instruction, or ""
		need  simdFeat
		guard simdFeat
		in    *asmInst
	}
	sites := map[string][]site{}
	nGateEdges, nWide := 0, 0
	for _, name := range l.names {
		f := l.funcs[name]
		n := len(f.insts)
		if n == 0 {
			continue
		}
		// gateOf[i] holds the features implied on the edge from i to its
		// branch target ([0]) and to the next instruction ([1]).
		type edges struct{ target, next int }
		es := make([]edges, n)
		gateOf := make([][2]simdFeat, n)
		for i := range f.insts {
			in := &f.insts[i]
			tg, nx, err := f.succ(i)
			if err != nil {
				t.Fatal(err)
			}
			es[i] = edges{tg, nx}
			if m := asmSBRe.FindAllStringSubmatch(in.args, -1); m != nil {
				for _, s := range m {
					g, ok := d.gates[s[1]]
					if !ok {
						continue
					}
					// A store to a feature variable outside the package
					// initializer would make the analysis unsound.
					if strings.HasSuffix(in.args, s[0]) && !strings.HasSuffix(name, ".init") {
						t.Errorf("%s: %s: store to a feature variable", d.short(name), in)
					}
					if in.op == "CMPB" && in.args == s[0]+", $0" && i+1 < n {
						switch f.insts[i+1].op {
						case "JNE":
							gateOf[i+1][0] = g
							nGateEdges++
						case "JEQ":
							gateOf[i+1][1] = g
							nGateEdges++
						}
					}
				}
			}
		}
		// reach[i] has the bit of a feature set when instruction i is
		// reachable without an edge that implies the feature.
		reach := make([]simdFeat, n)
		for _, feat := range []simdFeat{featAVX, featAVX2, featFMA} {
			stack := []int{0}
			reach[0] |= feat
			for len(stack) > 0 {
				i := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				for k, j := range [2]int{es[i].target, es[i].next} {
					if j < 0 || gateOf[i][k]&feat != 0 || reach[j]&feat != 0 {
						continue
					}
					reach[j] |= feat
					stack = append(stack, j)
				}
			}
		}
		for i := range f.insts {
			in := &f.insts[i]
			need, err := in.needs()
			if err != nil {
				t.Errorf("%s: %v", d.short(name), err)
				need = featAll
			}
			guard := featAll &^ reach[i]
			if need != 0 {
				if need&featAVX2 != 0 {
					nWide++
				}
				sites[name] = append(sites[name], site{need: need, guard: guard, in: in})
			}
			for _, s := range asmSBRe.FindAllStringSubmatch(in.args, -1) {
				ref := strings.TrimSuffix(s[1], "·f")
				if _, ok := l.funcs[ref]; ok && ref != name {
					sites[name] = append(sites[name], site{fn: ref, guard: guard, in: in})
				}
			}
		}
	}

	// need[f] is what f needs at its entry; why[f][feature] is one chain of
	// instructions that needs it.
	need := map[string]simdFeat{}
	why := map[string]map[simdFeat]string{}
	for changed := true; changed; {
		changed = false
		for _, name := range l.names {
			for _, s := range sites[name] {
				n, via := s.need, ""
				if s.fn != "" {
					n = need[s.fn]
				}
				n &^= s.guard
				for _, feat := range []simdFeat{featAVX, featAVX2, featFMA} {
					if n&feat == 0 || need[name]&feat != 0 {
						continue
					}
					if s.fn != "" {
						via = "\n      -> " + d.short(s.fn) + ": " + why[s.fn][feat]
					}
					if why[name] == nil {
						why[name] = map[simdFeat]string{}
					}
					why[name][feat] = s.in.String() + via
					need[name] |= feat
					changed = true
				}
			}
		}
	}

	var bad []string
	for _, name := range l.names {
		if need[name] == 0 {
			continue
		}
		root, reason := d.isRoot(name)
		if !root {
			continue
		}
		for _, feat := range []simdFeat{featAVX, featAVX2, featFMA} {
			if need[name]&feat != 0 {
				bad = append(bad, fmt.Sprintf("%s (%s) reaches an instruction that needs %v without a test of it:\n      %s",
					d.short(name), reason, feat, why[name][feat]))
			}
		}
	}
	sort.Strings(bad)
	for _, b := range bad {
		t.Error(b)
	}

	// The proof is vacuous if the listing or the gates were not recognized.
	nKernels := 0
	for _, n := range need {
		if n != 0 {
			nKernels++
		}
	}
	if nWide < 1000 || nGateEdges < 100 || nKernels < 50 {
		t.Errorf("too little recognized: %d AVX2 instructions, %d feature branches, %d functions that rely on their callers",
			nWide, nGateEdges, nKernels)
	}
	t.Logf("%d functions, %d feature variables, %d feature branches, %d AVX2 instructions, %d functions that rely on their callers",
		len(l.names), len(d.gates), nGateEdges, nWide, nKernels)

	// The purego build of the experiment has no vector instructions at all.
	out, err := simdGuardBuild(d.root, "purego")
	if err != nil {
		t.Fatal(err)
	}
	p := parseAsmListing(out)
	if len(p.names) < 100 {
		t.Fatalf("purego listing: %d functions", len(p.names))
	}
	for _, name := range p.names {
		for i := range p.funcs[name].insts {
			in := &p.funcs[name].insts[i]
			if n, err := in.needs(); n != 0 || err != nil {
				t.Errorf("purego: %s: %s: vector instruction", d.short(name), in)
			}
		}
	}
}

// TestNoLegacySSEWithDirtyUpperState proves that the amd64 simd build runs
// no legacy-SSE instruction while the upper halves of the Y registers are
// dirty. Go's scalar float code, its stack zeroing and its moves through X
// registers are legacy-SSE encoded (golang/go#80835), and each such
// instruction pays a state transition on Intel when a 256-bit instruction
// ran before it without a VZEROUPPER (archsimd.ClearAVXUpperBits) between.
//
// The state is followed along every path of every function: an instruction
// on a Y register makes it dirty, VZEROUPPER makes it clean. A call passes
// its state to the callee and continues with the state the callee returns
// in; a call out of the listing (the runtime, an indirect call) counts as
// legacy code. Functions that code outside the listing can reach start
// clean and must return clean.
func TestNoLegacySSEWithDirtyUpperState(t *testing.T) {
	d := simdGuardSetup(t)
	l := d.listing

	const (
		clean = 1 << iota
		dirty
	)
	// summary[f][e]: analysis of f entered in state e (0 clean, 1 dirty).
	type summary struct {
		exit uint8  // states it may return in
		hit  string // first legacy instruction that runs dirty, or ""
	}
	sum := map[string]*[2]summary{}
	for _, name := range l.names {
		sum[name] = &[2]summary{}
	}
	hits := map[string][]string{} // entered clean
	nDirty := 0
	analyze := func(name string, e int, report bool) summary {
		f := l.funcs[name]
		n := len(f.insts)
		in := make([]uint8, n)
		var res summary
		if n == 0 {
			return res
		}
		seen := map[int]bool{}
		hit := func(i int, s string) {
			if res.hit == "" {
				res.hit = s
			}
			if report && !seen[i] {
				seen[i] = true
				hits[name] = append(hits[name], s)
			}
		}
		in[0] = 1 << e
		work := []int{0}
		for len(work) > 0 {
			i := work[len(work)-1]
			work = work[:len(work)-1]
			inst := &f.insts[i]
			args := asmNameRe.ReplaceAllString(inst.args, "")
			var out uint8
			for _, s := range []uint8{clean, dirty} {
				if in[i]&s == 0 {
					continue
				}
				o := s
				callee := inst.callee()
				switch {
				case inst.op == "VZEROUPPER" || inst.op == "VZEROALL":
					o = clean
				case asmYRe.MatchString(args):
					o = dirty
				case callee != "":
					cs, known := sum[callee]
					switch {
					case strings.HasPrefix(callee, "runtime.morestack") || asmNoReturnRe.MatchString(callee):
						// The stack check runs before any vector
						// instruction; a failed bounds check does not
						// return.
					case known:
						c := cs[s>>1]
						if s == dirty && c.hit != "" {
							hit(i, fmt.Sprintf("%s\n      -> %s: %s", inst, d.short(callee), c.hit))
						}
						o = c.exit
					case s == dirty:
						hit(i, inst.String()+" (call out of the kernels)")
						o = clean
					}
					if inst.op == "JMP" {
						res.exit |= o
					}
				case inst.op == "RET":
					res.exit |= s
				case s == dirty && (inst.op == "DUFFZERO" || inst.op == "DUFFCOPY" ||
					!strings.HasPrefix(inst.op, "V") && asmXRe.MatchString(args)):
					hit(i, inst.String())
				}
				out |= o
			}
			tg, nx, err := f.succ(i)
			if err != nil {
				t.Fatal(err)
			}
			for _, j := range [2]int{tg, nx} {
				if j >= 0 && in[j]|out != in[j] {
					in[j] |= out
					work = append(work, j)
				}
			}
		}
		if report {
			for i := range in {
				if in[i]&dirty != 0 {
					nDirty++
				}
			}
		}
		return res
	}
	for changed := true; changed; {
		changed = false
		for _, name := range l.names {
			for e := 0; e < 2; e++ {
				if s := analyze(name, e, false); s != sum[name][e] {
					sum[name][e] = s
					changed = true
				}
			}
		}
	}
	for _, name := range l.names {
		analyze(name, 0, true)
	}

	nHits := 0
	for _, name := range l.names {
		for _, h := range hits[name] {
			nHits++
			t.Errorf("%s: legacy SSE with dirty upper state:\n      %s", d.short(name), h)
		}
		if root, reason := d.isRoot(name); root && sum[name][0].exit&dirty != 0 {
			nHits++
			t.Errorf("%s (%s) returns with dirty upper state: end the 256-bit section with archsimd.ClearAVXUpperBits",
				d.short(name), reason)
		}
	}
	if nDirty < 1000 {
		t.Errorf("only %d instructions with dirty upper state recognized", nDirty)
	}
	t.Logf("%d instructions run with dirty upper state, %d findings", nDirty, nHits)
}
