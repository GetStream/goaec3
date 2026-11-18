package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// kernEdgeVec is kernRandVec with extra ±0, ±Inf and NaN elements.
func kernEdgeVec(rng *rand.Rand, n int) []float32 {
	v := kernRandVec(rng, n)
	for i := range v {
		switch rng.Intn(24) {
		case 0:
			v[i] = 0
		case 1:
			v[i] = float32(math.Inf(1))
		case 2:
			v[i] = float32(math.Inf(-1))
		case 3:
			v[i] = float32(math.NaN())
		}
	}
	return v
}

// kernCheckExact is kernCheck without the NaN exemption on signs: NaN
// positions must agree, other elements bit for bit.
func kernCheckExact(t *testing.T, name string, got, want []float32) {
	t.Helper()
	kernCheck(t, name, got, want)
	for i := range want {
		if (got[i] != got[i]) != (want[i] != want[i]) {
			t.Fatalf("%s: idx %d NaN mismatch: got %v want %v", name, i, got[i], want[i])
		}
	}
}

func TestKernBinsElementwise(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for n := 0; n <= 70; n++ {
		for trial := 0; trial < 40; trial++ {
			x, y := kernEdgeVec(rng, n), kernEdgeVec(rng, n)
			a := kernEdgeVec(rng, 1)[0]
			got, want := make([]float32, n), make([]float32, n)
			kernScale(x, a, got)
			kernScaleGeneric(x, a, want)
			kernCheckExact(t, "Scale", got, want)
			kernAdd(x, y, got)
			kernAddGeneric(x, y, want)
			kernCheckExact(t, "Add", got, want)
			kernSub(x, y, got)
			kernSubGeneric(x, y, want)
			kernCheckExact(t, "Sub", got, want)
			for _, lim := range [][2]float32{{-32768, 32767}, {-1, 1}, {0, 1}, {-1, 0}, {a, a}} {
				copy(got, x)
				copy(want, x)
				kernClamp(got, lim[0], lim[1])
				kernClampGeneric(want, lim[0], lim[1])
				kernCheckExact(t, "Clamp", got, want)
				if lim[0] < lim[1] && lim[0] != 0 && lim[1] != 0 {
					for i, v := range x {
						if v == v && !kernSameBits(want[i], supSafeClamp(v, lim[0], lim[1])) {
							t.Fatalf("Clamp vs supSafeClamp(%v) = %v, %v", v, want[i], supSafeClamp(v, lim[0], lim[1]))
						}
					}
				}
			}
			c, d := kernEdgeVec(rng, n), kernEdgeVec(rng, n)
			if n == 0 {
				y = []float32{1}
			}
			kernMulAdd2(x, y, c, d, got)
			kernMulAdd2Generic(x, y, c, d, want)
			kernCheckExact(t, "MulAdd2", got, want)
			kernDiffPower(x, y, c, d, got)
			kernDiffPowerGeneric(x, y, c, d, want)
			kernCheckExact(t, "DiffPower", got, want)
			kernOneMinusSq(x, got)
			kernOneMinusSqGeneric(x, want)
			kernCheckExact(t, "OneMinusSq", got, want)
			kernMinGo(x, y, got)
			kernMinGoGeneric(x, y, want)
			kernCheckExact(t, "MinGo", got, want)
			kernMaxGo(x, y, got)
			kernMaxGoGeneric(x, y, want)
			kernCheckExact(t, "MaxGo", got, want)
			copy(got, c)
			copy(want, c)
			kernAxpy(x, a, got)
			kernAxpyGeneric(x, a, want)
			kernCheckExact(t, "Axpy", got, want)
			copy(got, c)
			copy(want, c)
			kernMaxScalarGo(got, a)
			kernMaxScalarGoGeneric(want, a)
			kernCheckExact(t, "MaxScalarGo", got, want)
			copy(got, c)
			copy(want, c)
			kernSmooth(got, x, a)
			kernSmoothGeneric(want, x, a)
			kernCheckExact(t, "Smooth", got, want)
			kernDiv(x, y, got)
			kernDivGeneric(x, y, want)
			kernCheckExact(t, "Div", got, want)
			kernStdMax(x, y, got)
			kernStdMaxGeneric(x, y, want)
			kernCheckExact(t, "StdMax", got, want)
			copy(got, c)
			copy(want, c)
			kernReverb(got, x, y, a)
			kernReverbGeneric(want, x, y, a)
			kernCheckExact(t, "Reverb", got, want)
			copy(got, c)
			copy(want, c)
			kernReverbScalar(got, x, a, y[0])
			kernReverbScalarGeneric(want, x, a, y[0])
			kernCheckExact(t, "ReverbScalar", got, want)
			// In place.
			copy(got, x)
			copy(want, x)
			kernScale(got, a, got)
			kernScaleGeneric(want, a, want)
			kernCheckExact(t, "Scale in place", got, want)
		}
	}
}

func TestKernMinMax(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	gen := []func(n int) []float32{
		func(n int) []float32 { return kernEdgeVec(rng, n) },
		func(n int) []float32 { return kernRandVec(rng, n) },
		func(n int) []float32 {
			// Zeros of both signs as extremes.
			v := make([]float32, n)
			for i := range v {
				switch rng.Intn(3) {
				case 0:
					v[i] = float32(math.Copysign(0, -1))
				case 1:
					v[i] = float32(rng.Float64())
				}
			}
			return v
		},
		func(n int) []float32 {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			return v
		},
	}
	fast := 0
	defer func() {
		if !kernMinMaxFast && fast != 0 || kernMinMaxFast && fast == 0 {
			t.Errorf("fast path taken %d times (kernMinMaxFast=%v)", fast, kernMinMaxFast)
		}
	}()
	for n := 1; n <= 70; n++ {
		for trial := 0; trial < 200; trial++ {
			x := gen[trial%len(gen)](n)
			gmn, gmx, ok := kernMinMax(x)
			if !ok {
				continue
			}
			wmn, wmx := kernMinMaxGeneric(x)
			kernCheckExact(t, "MinMax", []float32{gmn, gmx}, []float32{wmn, wmx})
			// The last-occurrence, non-strict formulation (std::minmax_element).
			lmn, lmx := x[0], x[0]
			for _, v := range x[1:] {
				if v < lmn {
					lmn = v
				}
				if !(v < lmx) {
					lmx = v
				}
			}
			kernCheckExact(t, "MinMax last", []float32{gmn, gmx}, []float32{lmn, lmx})
			fast++
		}
	}
}

func TestKernSumRows(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for nr := 0; nr <= 40; nr++ {
		for trial := 0; trial < 20; trial++ {
			rows := make([]*[FftLengthBy2Plus1]float32, nr)
			for i := range rows {
				rows[i] = new([FftLengthBy2Plus1]float32)
				copy(rows[i][:], kernEdgeVec(rng, FftLengthBy2Plus1))
			}
			var got, want [FftLengthBy2Plus1]float32
			copy(got[:], kernRandVec(rng, FftLengthBy2Plus1))
			kernSumRows(&got, rows)
			kernSumRowsGeneric(&want, rows)
			kernCheckExact(t, "SumRows", got[:], want[:])
		}
	}
}

func TestKernPackUnpack(t *testing.T) {
	rng := rand.New(rand.NewSource(14))
	for trial := 0; trial < 2000; trial++ {
		var v [FftLength]float32
		copy(v[:], kernEdgeVec(rng, FftLength))
		var gre, gim, wre, wim [FftLengthBy2Plus1]float32
		copy(gre[:], kernRandVec(rng, FftLengthBy2Plus1))
		copy(gim[:], kernRandVec(rng, FftLengthBy2Plus1))
		wre, wim = gre, gim
		kernUnpack(&v, &gre, &gim)
		kernUnpackGeneric(&v, &wre, &wim)
		kernCheckExact(t, "Unpack re", gre[:], wre[:])
		kernCheckExact(t, "Unpack im", gim[:], wim[:])

		copy(gre[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		copy(gim[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		var gv, wv [FftLength]float32
		copy(gv[:], kernRandVec(rng, FftLength))
		wv = gv
		kernPack(&gre, &gim, &gv)
		kernPackGeneric(&gre, &gim, &wv)
		for i := range gv {
			if math.Float32bits(gv[i]) != math.Float32bits(wv[i]) {
				t.Fatalf("Pack: idx %d got %#x want %#x", i, math.Float32bits(gv[i]), math.Float32bits(wv[i]))
			}
		}
	}
}

func TestKernBinsNoAllocs(t *testing.T) {
	x := make([]float32, FftLengthBy2Plus1)
	z := make([]float32, FftLengthBy2Plus1)
	var v [FftLength]float32
	var re, im [FftLengthBy2Plus1]float32
	rows := []*[FftLengthBy2Plus1]float32{&re, &im}
	n := testing.AllocsPerRun(100, func() {
		kernScale(x, 2, z)
		kernAdd(x, z, z)
		kernSub(x, z, z)
		kernMinMax(x)
		kernSumRows(&re, rows[1:])
		kernUnpack(&v, &re, &im)
		kernPack(&re, &im, &v)
	})
	if n != 0 {
		t.Fatalf("kernels allocate %v times", n)
	}
}

func BenchmarkKernBins(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, FftLengthBy2Plus1)
	y := make([]float32, FftLengthBy2Plus1)
	for i := range x {
		x[i], y[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	z := make([]float32, FftLengthBy2Plus1)
	var v [FftLength]float32
	var re, im [FftLengthBy2Plus1]float32
	rows := make([]*[FftLengthBy2Plus1]float32, 13)
	for i := range rows {
		rows[i] = new([FftLengthBy2Plus1]float32)
		copy(rows[i][:], x)
	}
	bench := func(name string, f, ref func()) {
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				f()
			}
		})
		b.Run(name+"Generic", func(b *testing.B) {
			for b.Loop() {
				ref()
			}
		})
	}
	bench("Scale", func() { kernScale(x, 3, z) }, func() { kernScaleGeneric(x, 3, z) })
	bench("Add", func() { kernAdd(x, y, z) }, func() { kernAddGeneric(x, y, z) })
	bench("MinMax64", func() { kernMinMax(x[:64]) }, func() { kernMinMaxGeneric(x[:64]) })
	bench("SumRows13", func() { kernSumRows(&re, rows) }, func() { kernSumRowsGeneric(&re, rows) })
	bench("Unpack", func() { kernUnpack(&v, &re, &im) }, func() { kernUnpackGeneric(&v, &re, &im) })
	bench("Pack", func() { kernPack(&re, &im, &v) }, func() { kernPackGeneric(&re, &im, &v) })
	bench("Sub", func() { kernSub(x, y, z) }, func() { kernSubGeneric(x, y, z) })
	bench("MulAdd2", func() { kernMulAdd2(x, y, x, y, z) }, func() { kernMulAdd2Generic(x, y, x, y, z) })
	bench("Axpy", func() { kernAxpy(x, 0.5, z) }, func() { kernAxpyGeneric(x, 0.5, z) })
	bench("OneMinusSq", func() { kernOneMinusSq(x, z) }, func() { kernOneMinusSqGeneric(x, z) })
	bench("DiffPower", func() { kernDiffPower(x, y, y, x, z) }, func() { kernDiffPowerGeneric(x, y, y, x, z) })
	bench("MinGo", func() { kernMinGo(x, y, z) }, func() { kernMinGoGeneric(x, y, z) })
	bench("MaxGo", func() { kernMaxGo(x, y, z) }, func() { kernMaxGoGeneric(x, y, z) })
	bench("MaxScalarGo", func() { kernMaxScalarGo(z, 0.1) }, func() { kernMaxScalarGoGeneric(z, 0.1) })
	bench("Clamp", func() { kernClamp(z, -1, 1) }, func() { kernClampGeneric(z, -1, 1) })
	bench("Smooth", func() { kernSmooth(z, x, 0.1) }, func() { kernSmoothGeneric(z, x, 0.1) })
	bench("Div", func() { kernDiv(x, y, z) }, func() { kernDivGeneric(x, y, z) })
	bench("Reverb", func() { kernReverb(z, x, y, 0.5) }, func() { kernReverbGeneric(z, x, y, 0.5) })
	bench("ReverbScalar", func() { kernReverbScalar(z, x, 2, 0.5) }, func() { kernReverbScalarGeneric(z, x, 2, 0.5) })
	bench("StdMax", func() { kernStdMax(x, y, z) }, func() { kernStdMaxGeneric(x, y, z) })

	a := func() *[FftLengthBy2Plus1]float32 {
		var p [FftLengthBy2Plus1]float32
		for i := range p {
			p[i] = float32(math.Abs(rng.NormFloat64()) * 1000)
		}
		return &p
	}
	ne, ec, ma, et, es, mt, out := a(), a(), a(), a(), a(), a(), a()
	bench("NoAudibleEcho", func() { kernNoAudibleEcho(ne, ec, ma, et, es, mt, out) },
		func() { kernNoAudibleEchoGeneric(ne, ec, ma, et, es, mt, out) })
	bench("MinGain", func() { kernMinGain(ec, 30, out) }, func() { kernMinGainGeneric(ec, 30, out) })
	bench("Weigh", func() { kernWeigh(800, 0.01, ec[7:], out[7:]) }, func() { kernWeighGeneric(800, 0.01, ec[7:], out[7:]) })
	n2, s2 := *ne, *ec
	bench("NoiseUpdate", func() { kernNoiseUpdate(&n2, &s2) }, func() { kernNoiseUpdateGeneric(&n2, &s2) })
	bench("NoiseInitialUpdate", func() { kernNoiseInitialUpdate(&s2, &n2) }, func() { kernNoiseInitialUpdateGeneric(&s2, &n2) })
	h := [3]float32{0.7929742, -0.36072128, -0.47047766}
	firX := make([]float32, 13*64)
	firOut := make([]float32, 13*64)
	bench("Fir3x64", func() { kernFir3(firX, &h, firOut, 128, 191) }, func() { kernFir3Generic(firX, &h, firOut, 128, 191) })
	bench("RefinedMu", func() { kernRefinedMu(ne, ec, ma, 10, 13, out) }, func() { kernRefinedMuGeneric(ne, ec, ma, 10, 13, out) })
	he := *ma
	bench("RefinedHError", func() { kernRefinedHError(ne, ec, &he) }, func() { kernRefinedHErrorGeneric(ne, ec, &he) })
	bench("RefinedLeak", func() { kernRefinedLeak(ne, ec, et[:], false, 0.001, 0.01, 0.001, 2, &he) },
		func() { kernRefinedLeakGeneric(ne, ec, et[:], false, 0.001, 0.01, 0.001, 2, &he) })
	bench("CoarseMu", func() { kernCoarseMu(ne, 10, 0.7, out) }, func() { kernCoarseMuGeneric(ne, 10, 0.7, out) })
}

func kernEdge65(rng *rand.Rand) *[FftLengthBy2Plus1]float32 {
	var a [FftLengthBy2Plus1]float32
	copy(a[:], kernEdgeVec(rng, FftLengthBy2Plus1))
	return &a
}

func kernNorm65(rng *rand.Rand, scale float64) *[FftLengthBy2Plus1]float32 {
	var a [FftLengthBy2Plus1]float32
	for i := range a {
		a[i] = float32(math.Abs(rng.NormFloat64()) * scale)
	}
	return &a
}

func TestKernBinsSites(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	pick := func() *[FftLengthBy2Plus1]float32 {
		if rng.Intn(2) == 0 {
			return kernEdge65(rng)
		}
		return kernNorm65(rng, math.Pow(10, float64(rng.Intn(8)-2)))
	}
	for trial := 0; trial < 3000; trial++ {
		ne, ec, ma, et, es, mt := pick(), pick(), pick(), pick(), pick(), pick()
		var got, want [FftLengthBy2Plus1]float32
		kernNoAudibleEcho(ne, ec, ma, et, es, mt, &got)
		kernNoAudibleEchoGeneric(ne, ec, ma, et, es, mt, &want)
		kernCheckExact(t, "NoAudibleEcho", got[:], want[:])

		mp := pick()[0]
		kernMinGain(ec, mp, &got)
		kernMinGainGeneric(ec, mp, &want)
		kernCheckExact(t, "MinGain", got[:], want[:])

		th, nz := pick()[1], pick()[2]
		from := rng.Intn(10)
		kernWeigh(th, nz, ec[from:], got[from:])
		kernWeighGeneric(th, nz, ec[from:], want[from:])
		kernCheckExact(t, "Weigh", got[from:], want[from:])

		g2, w2 := *ne, *ne
		kernNoiseUpdate(&g2, ec)
		kernNoiseUpdateGeneric(&w2, ec)
		kernCheckExact(t, "NoiseUpdate", g2[:], w2[:])
		g2, w2 = *ne, *ne
		kernNoiseInitialUpdate(ec, &g2)
		kernNoiseInitialUpdateGeneric(ec, &w2)
		kernCheckExact(t, "NoiseInitialUpdate", g2[:], w2[:])

		var gm, wm [FftLengthBy2Plus1]float32
		gate, sp := pick()[6], pick()[7]
		kernRefinedMu(ne, ec, ma, gate, sp, &gm)
		kernRefinedMuGeneric(ne, ec, ma, gate, sp, &wm)
		kernCheckExact(t, "RefinedMu", gm[:], wm[:])
		g2, w2 = *ma, *ma
		kernRefinedHError(ne, ec, &g2)
		kernRefinedHErrorGeneric(ne, ec, &w2)
		kernCheckExact(t, "RefinedHError", g2[:], w2[:])
		for _, dis := range []bool{false, true} {
			g2, w2 = *ma, *ma
			p := pick()
			kernRefinedLeak(ne, ec, et[:], dis, p[0], p[1], p[2], p[3], &g2)
			kernRefinedLeakGeneric(ne, ec, et[:], dis, p[0], p[1], p[2], p[3], &w2)
			kernCheckExact(t, "RefinedLeak", g2[:], w2[:])
		}
		kernCoarseMu(ne, gate, sp, &gm)
		kernCoarseMuGeneric(ne, gate, sp, &wm)
		kernCheckExact(t, "CoarseMu", gm[:], wm[:])

		n := 2 + rng.Intn(300)
		x := kernEdgeVec(rng, n)
		h := [3]float32{0.7929742, -0.36072128, -0.47047766}
		if trial%2 == 0 {
			h = [3]float32{pick()[3], pick()[4], pick()[5]}
		}
		lo := 2 + rng.Intn(n-1)
		hi := lo - 1 + rng.Intn(n-lo+1)
		go1, wo1 := kernEdgeVec(rng, n), make([]float32, n)
		copy(wo1, go1)
		kernFir3(x, &h, go1, lo, hi)
		kernFir3Generic(x, &h, wo1, lo, hi)
		kernCheckExact(t, "Fir3", go1, wo1)
	}
}
