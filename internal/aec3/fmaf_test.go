package aec3

import (
	"math"
	"math/big"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"testing"
)

// fmafExactPrec makes a*b+c exact in big.Float: the product of two float32
// values has at most 48 significant bits, and the sum spans at most from
// 2^256 down to 2^-298.
const fmafExactPrec = 600

// fmafExact rounds a*b+c once to float32 with exact big.Float arithmetic.
// z and t are scratch values reused across calls.
func fmafExact(a, b, c float32, z, t *big.Float) float32 {
	fa, fb, fc := float64(a), float64(b), float64(c)
	if math.IsInf(fa, 0) || math.IsInf(fb, 0) || math.IsInf(fc, 0) || a != a || b != b || c != c {
		return float32(float64(fa*fb) + fc) // IEEE special-value rules
	}
	z.SetPrec(fmafExactPrec).SetFloat64(fa)
	t.SetPrec(fmafExactPrec).SetFloat64(fb)
	z.Mul(z, t)
	t.SetFloat64(fc)
	z.Add(z, t)
	if z.Sign() == 0 {
		return float32(float64(fa*fb) + fc) // sign of an exact zero
	}
	f, _ := z.Float32()
	return f
}

// fmafOperands returns case i of the fmaf test corpus: random bit patterns
// (NaN, Inf, subnormals included), values of moderate range, near
// cancellation, exact float32 rounding ties perturbed or not, subnormal
// results, the overflow boundary and special values, plus the operand
// families of the spectrum, matched-filter and adaptive-filter kernels
// (cases 8 to 11).
func fmafOperands(rng *rand.Rand, i int) (a, b, c float32) {
	pow := func(lo, hi int) float64 { return math.Ldexp(1, lo+rng.Intn(hi-lo+1)) }
	sign := func() float32 {
		if rng.Intn(2) == 0 {
			return -1
		}
		return 1
	}
	specials := [...]float32{0, float32(math.Copysign(0, -1)), 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
		-math.SmallestNonzeroFloat32, 0x1p-126, -0x1p-126, math.Float32frombits(0x007fffff), 0x1p64, 0x1p-75, 0.5}
	switch i % 12 {
	case 8:
		// a == b, both standard normal; c standard normal.
		a = float32(rng.NormFloat64())
		return a, a, float32(rng.NormFloat64())
	case 9:
		// Products in [1, 4) with c one float32 ulp step from -a*b.
		a = float32(1 + rng.Float64())
		b = float32(1 + rng.Float64())
		c = -float32(float64(a) * float64(b))
		return a, b, math.Float32frombits(math.Float32bits(c) + uint32(rng.Intn(3)) - 1)
	case 10:
		// Standard normal a, b with c = -a*b scaled by 1 + k*2^-24 or c tiny.
		a = float32(rng.NormFloat64())
		b = float32(rng.NormFloat64())
		if rng.Intn(2) == 0 {
			c = -float32(float64(a)*float64(b)) * (1 + float32(rng.Intn(5)-2)*0x1p-24)
		} else {
			c = float32(rng.NormFloat64() * 1e-8)
		}
		return a, b, c
	case 11:
		// Normal values scaled by 2^[-20, 20) (2^[-40, 40) for c).
		a = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-20)))
		b = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-20)))
		c = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(80)-40)))
		return a, b, c
	case 0:
		return math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32())
	case 1:
		a = float32(rng.NormFloat64() * pow(-40, 40))
		b = float32(rng.NormFloat64() * pow(-40, 40))
		c = float32(rng.NormFloat64() * pow(-80, 80))
	case 2:
		// Near cancellation: c within a few float32 ulps of -a*b.
		a = float32(rng.NormFloat64() * pow(-20, 20))
		b = float32(rng.NormFloat64() * pow(-20, 20))
		c = -float32(float64(a) * float64(b))
		c = math.Float32frombits(math.Float32bits(c) + uint32(rng.Intn(9)) - 4)
	case 3, 4:
		// Products of short odd integers: a*b has up to 26 significant bits,
		// so a*b+c often is, or is a tiny step away from, a float32
		// rounding tie.
		ea, eb := rng.Intn(60)-30, rng.Intn(60)-30
		a = sign() * float32(math.Ldexp(float64(rng.Intn(1<<13)|1), ea))
		b = sign() * float32(math.Ldexp(float64(rng.Intn(1<<13)|1), eb))
		switch rng.Intn(4) {
		case 0:
			c = 0
		case 1:
			// A perturbation far below the product's ulp.
			c = sign() * float32(math.Ldexp(1, ea+eb-rng.Intn(70)))
		case 2:
			// An addend on the product's grid: the sum stays short.
			c = sign() * float32(math.Ldexp(float64(rng.Intn(1<<24)), ea+eb))
		default:
			c = sign() * float32(math.Ldexp(float64(rng.Intn(1<<24)|1), ea+eb-rng.Intn(30)))
		}
	case 5:
		// Results in or near the float32 subnormal range.
		a = sign() * float32(rng.Float64()*pow(-80, -50))
		b = sign() * float32(rng.Float64()*pow(-80, -50))
		switch rng.Intn(3) {
		case 0:
			c = math.Float32frombits(rng.Uint32() & 0x80ffffff) // subnormal or tiny normal
		case 1:
			c = -float32(float64(a) * float64(b))
			c = math.Float32frombits(math.Float32bits(c) + uint32(rng.Intn(5)) - 2)
		default:
			c = 0
		}
	case 6:
		// Around the overflow boundary 2^128 - 2^103 = (31*601) * 1801 * 2^103.
		k := rng.Intn(104)
		switch rng.Intn(3) {
		case 0:
			a = sign() * float32(math.Ldexp(31*601, k))
			b = sign() * float32(math.Ldexp(1801, 103-k))
		case 1:
			a = sign() * float32(rng.Float64()*pow(60, 70))
			b = sign() * float32(rng.Float64()*pow(55, 64))
		default:
			a = sign() * float32(math.Ldexp(1, k))
			b = sign() * math.MaxFloat32 * float32(math.Ldexp(1, -k))
		}
		switch rng.Intn(4) {
		case 0:
			c = 0
		case 1:
			c = sign() * float32(math.Ldexp(1, rng.Intn(110)))
		case 2:
			c = sign() * math.MaxFloat32
		default:
			c = math.Float32frombits(rng.Uint32())
		}
	default:
		ops := [3]float32{}
		for j := range ops {
			if rng.Intn(3) == 0 {
				ops[j] = math.Float32frombits(rng.Uint32())
			} else {
				ops[j] = specials[rng.Intn(len(specials))]
			}
		}
		return ops[0], ops[1], ops[2]
	}
	return a, b, c
}

// fmafSpecials are the special values of the fmaf test corpus.
var fmafSpecials = [...]float32{0, float32(math.Copysign(0, -1)), 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)),
	float32(math.NaN()), math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
	-math.SmallestNonzeroFloat32, 0x1p-126, -0x1p-126, math.Float32frombits(0x007fffff), 0x1p64, 0x1p-75, 0.5,
	math.Float32frombits(0x7fa00000), 0x1p-149 * 3, -0x1p-75, 0x1p127, 0x1p-64}

// fmafBoundaryCases calls f with operands whose exact a*b + c lies on, just
// below and just above every kind of float32 rounding boundary:
//
//   - every triple of special values (zeros of both signs, infinities, NaNs,
//     the largest and smallest magnitudes);
//   - for a float32 value v of every exponent, subnormals included, and
//     mantissas of both parities, the midpoints next to v: c = v and
//     a*b = +-hulp exactly (a tie), and a*b = +-hulp*(1 - 2^-46) (short of
//     the tie by less than the float64 sum can hold, the double-rounding
//     case); with v's neighbour as c the same product comes from the other
//     side of the midpoint. hulp, half the spacing at v, is the product of
//     two powers of two, and (1 + 2^-23)*(1 - 2^-23) = 1 - 2^-46;
//   - the same around the overflow boundary 2^128 - 2^103, on products that
//     alone overflow float32, and on sums that cancel to zero.
func fmafBoundaryCases(f func(a, b, c float32)) {
	for _, a := range fmafSpecials {
		for _, b := range fmafSpecials {
			for _, c := range fmafSpecials {
				f(a, b, c)
			}
		}
	}
	const up, down = 1 + 0x1p-23, 1 - 0x1p-23 // up*down = 1 - 2^-46
	mantissas := [...]uint32{0, 1, 2, 3, 0x400000, 0x400001, 0x555555, 0x2aaaaa, 0x7ffffe, 0x7fffff}
	for exp := uint32(0); exp < 255; exp++ {
		for _, m := range mantissas {
			v := math.Float32frombits(exp<<23 | m)
			// Half the spacing at v as a product of two float32 powers of
			// two: 2^(e-24) for normal v, 2^-150 below.
			e := int(exp) - 127
			if exp == 0 {
				e = -126
			}
			h1 := float32(math.Ldexp(1, (e-24)/2))
			h2 := float32(math.Ldexp(1, (e-24)-(e-24)/2))
			if h1 == 0 || h2 == 0 || math.IsInf(float64(h1), 0) || math.IsInf(float64(h2), 0) {
				panic("fmafBoundaryCases: half spacing is not a product of float32 powers of two")
			}
			for _, sv := range [2]float32{v, -v} {
				for _, sh := range [2]float32{h1, -h1} {
					f(sh, h2, sv)           // the tie
					f(sh*up, h2*down, sv)   // just short of it
					f(sh*up, h2*up, sv)     // beyond it, by more than float64 loses
					f(sh*down, h2*down, sv) // short of it by 2^-23
					f(sh*3, h2, sv)         // three halves: the next tie but one
					f(sh*up, h2*down*3, sv)
					f(sh, h2*2, sv) // a full step: exact
					f(sh*up, h2*down*2, sv)
				}
			}
		}
	}
	// Products that overflow float32 on their own, and cancellation.
	for _, c := range [...]float32{0, math.MaxFloat32, -math.MaxFloat32, 0x1p103, -0x1p103, 0x1p104, -0x1p104,
		0x1p127, -0x1p127, float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, a := range [...]float32{0x1p64, -0x1p64, 0x1p64 * up, 0x1p64 * down, math.MaxFloat32, 0x1p100} {
			for _, b := range [...]float32{0x1p64, 0x1p64 * down, 0x1p63, 0x1p63 * up, -0x1p64, 2, 0x1p28} {
				f(a, b, c)
			}
		}
	}
	for _, a := range [...]float32{1, up, down, 3, 0x1p-75, 0x1p-75 * up, 0x1p60, math.MaxFloat32, 0x1p-149} {
		for _, b := range [...]float32{1, up, down, -1, 0x1p-74, 0x1p60 * down, 0x1p-149} {
			p := float32(float64(a) * float64(b))
			for _, c := range [...]float32{-p, p, math.Float32frombits(math.Float32bits(-p) + 1),
				math.Float32frombits(math.Float32bits(-p) - 1)} {
				f(a, b, c)
				f(-a, b, c)
			}
		}
	}
}

// TestFmafCorrectlyRounded checks fmaf, its slow path and the rounded
// float64 sum wherever fmafTie does not flag it against exact arithmetic:
// on the boundary cases of fmafBoundaryCases and on GOAEC3_FMAF_N random
// cases (default 3e6; 3e5 with -short), spread over GOMAXPROCS workers.
func TestFmafCorrectlyRounded(t *testing.T) {
	n := 3_000_000
	if testing.Short() {
		n = 300_000
	}
	if s := os.Getenv("GOAEC3_FMAF_N"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		n = v
	}
	// Fixed cases: (1+2^-12)^2 - 2^-11 = 1 + 2^-24 is a tie; a perturbation
	// below the float64 precision of the sum decides it.
	im := float32(4097.0 / 4096.0)
	fixed := [][3]float32{
		{im, im, -1.0 / (1 << 11)}, {im, im, 1e-30}, {im, im, -1e-30}, {im, im, 0},
		{31 * 601 * 0x1p50, 1801 * 0x1p53, 0}, {31 * 601 * 0x1p50, 1801 * 0x1p53, -0x1p-100},
		{0x1p-75, 0x1p-75, 0}, {0x1p-75, 0x1p-75, 0x1p-149}, {0x1p-75, 3 * 0x1p-75, -0x1p-149},
		{float32(math.Copysign(0, -1)), 1, float32(math.Copysign(0, -1))}, {float32(math.Copysign(0, -1)), 1, 0},
		{1, 1, -1}, {-1, 1, 1}, {0x1p-100, 0x1p-100, float32(math.Copysign(0, -1))},
	}
	var mu sync.Mutex
	var failure string
	var failures int
	check := func(a, b, c float32, z, t2 *big.Float) {
		want := fmafExact(a, b, c, z, t2)
		// The rounded float64 sum, where fmafTie does not flag it.
		sum := fmafSum(a, b, c)
		inline := float32(sum)
		if fmafTie(sum, float64(inline)) {
			inline = want
		}
		for _, got := range [3]float32{fmaf(a, b, c), fmafSlow(a, b, c), inline} {
			if !kernSameBits(got, want) {
				mu.Lock()
				if failures == 0 {
					failure = "fmaf(" + strconv.FormatFloat(float64(a), 'x', -1, 32) + ", " +
						strconv.FormatFloat(float64(b), 'x', -1, 32) + ", " +
						strconv.FormatFloat(float64(c), 'x', -1, 32) + ") = " +
						strconv.FormatFloat(float64(got), 'x', -1, 32) + ", want " +
						strconv.FormatFloat(float64(want), 'x', -1, 32)
				}
				failures++
				mu.Unlock()
			}
		}
	}
	var z, t2 big.Float
	for _, v := range fixed {
		check(v[0], v[1], v[2], &z, &t2)
	}
	fmafBoundaryCases(func(a, b, c float32) {
		check(a, b, c, &z, &t2)
		check(b, a, c, &z, &t2)
	})
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(1000 + w)))
			var z, t2 big.Float
			for i := w; i < n; i += workers {
				a, b, c := fmafOperands(rng, i)
				check(a, b, c, &z, &t2)
				// Commuted operands must give the same result.
				check(b, a, c, &z, &t2)
			}
		}(w)
	}
	wg.Wait()
	if failures > 0 {
		t.Fatalf("%d mismatches; first: %s", failures, failure)
	}
}

func BenchmarkFmaf(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n = 1024
	var a, x, c [n]float32
	for i := range a {
		a[i], x[i], c[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	var out [n]float32
	b.Run("fast", func(b *testing.B) {
		for b.Loop() {
			for i := range out {
				out[i] = fmaf(a[i], x[i], c[i])
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/fma")
	})
	b.Run("inline", func(b *testing.B) {
		for b.Loop() {
			for i := range out {
				q := fmafSum(a[i], x[i], c[i])
				r := float32(q)
				if fmafHard(q) {
					r = fmafSlow(a[i], x[i], c[i])
				}
				out[i] = r
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/fma")
	})
	b.Run("slow", func(b *testing.B) {
		for b.Loop() {
			for i := range out {
				out[i] = fmafSlow(a[i], x[i], c[i])
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/fma")
	})
}
