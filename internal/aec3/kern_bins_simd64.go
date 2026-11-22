//go:build goexperiment.simd && (arm64 || amd64) && !purego

package aec3

import "simd/archsimd"

// 64-element forms of the slice helpers. A helper runs them for 64 and 65
// elements (kernIs64) and finishes the 65th with its scalar tail. The
// sixteen steps are closures called with constant offsets, so every access
// is base plus displacement and the loop over slices pays no bounds check.
// An element sees the operations of the loop of the helper.

// kernIs64 reports whether a helper over n elements runs its 64-element
// form.
func kernIs64(n int) bool { return n&^1 == 64 }

func kernLd64(a *[64]float32, k int) f32x4 {
	return archsimd.LoadFloat32x4Array((*[4]float32)(a[k : k+4]))
}

func kernSt64(a *[64]float32, k int, v f32x4) {
	v.StoreArray((*[4]float32)(a[k : k+4]))
}

// kernScale64 sets z = x*a.
func kernScale64(x *[64]float32, a float32, z *[64]float32) {
	av := kernSplat4(a)
	f := func(k int) {
		_ = av // a captured vector makes the compiler inline the closure
		kernSt64(z, k, kernLd64(x, k).Mul(av))
	}
	g := func(k int) {
		_ = av
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernAdd64 sets z = x+y.
func kernAdd64(x, y, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(z, k, kernLd64(x, k).Add(kernLd64(y, k)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernSub64 sets z = x-y.
func kernSub64(x, y, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(z, k, kernLd64(x, k).Sub(kernLd64(y, k)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernMul64 sets z = x*y.
func kernMul64(x, y, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(z, k, kernLd64(x, k).Mul(kernLd64(y, k)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernSqrt64 sets x = sqrt(x).
func kernSqrt64(x *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(x, k, kernLd64(x, k).Sqrt())
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernClamp64 sets x = std::min(std::max(x, lo), hi).
func kernClamp64(x *[64]float32, lo, hi float32) {
	lv, hv := kernSplat4(lo), kernSplat4(hi)
	f := func(k int) {
		_ = lv // a captured vector makes the compiler inline the closure
		kernSt64(x, k, vStdMin(vStdMax(kernLd64(x, k), lv), hv))
	}
	g := func(k int) {
		_ = lv
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernMulAdd2x64 sets z = a*b + c*d.
func kernMulAdd2x64(a, b, c, d, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		p := kernLd64(a, k).Mul(kernLd64(b, k))
		q := kernLd64(c, k).Mul(kernLd64(d, k))
		kernSt64(z, k, p.Add(q))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernAxpy64 sets z += x*a.
func kernAxpy64(x *[64]float32, a float32, z *[64]float32) {
	av := kernSplat4(a)
	f := func(k int) {
		_ = av // a captured vector makes the compiler inline the closure
		kernSt64(z, k, kernLd64(z, k).Add(kernLd64(x, k).Mul(av)))
	}
	g := func(k int) {
		_ = av
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernOneMinusSq64 sets z = 1 - x*x.
func kernOneMinusSq64(x, z *[64]float32) {
	one := kernSplat4(1)
	f := func(k int) {
		_ = one // a captured vector makes the compiler inline the closure
		v := kernLd64(x, k)
		kernSt64(z, k, one.Sub(v.Mul(v)))
	}
	g := func(k int) {
		_ = one
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernDiffPower64 sets z = dr*dr + di*di with dr = yr-er and di = yi-ei.
func kernDiffPower64(yr, er, yi, ei, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		dr := kernLd64(yr, k).Sub(kernLd64(er, k))
		di := kernLd64(yi, k).Sub(kernLd64(ei, k))
		kernSt64(z, k, dr.Mul(dr).Add(di.Mul(di)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernStdMin64 sets z = std::min(x, y).
func kernStdMin64(x, y, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(z, k, vStdMin(kernLd64(x, k), kernLd64(y, k)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernStdMax64 sets z = std::max(x, y).
func kernStdMax64(x, y, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(z, k, vStdMax(kernLd64(x, k), kernLd64(y, k)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernStdMaxScalar64 sets x = std::max(x, a).
func kernStdMaxScalar64(x *[64]float32, a float32) {
	av := kernSplat4(a)
	f := func(k int) {
		_ = av // a captured vector makes the compiler inline the closure
		kernSt64(x, k, vStdMax(kernLd64(x, k), av))
	}
	g := func(k int) {
		_ = av
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernSmooth64 sets s = s + c*(y-s).
func kernSmooth64(s, y *[64]float32, c float32) {
	cv := kernSplat4(c)
	f := func(k int) {
		_ = cv // a captured vector makes the compiler inline the closure
		a := kernLd64(s, k)
		kernSt64(s, k, a.Add(cv.Mul(kernLd64(y, k).Sub(a))))
	}
	g := func(k int) {
		_ = cv
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernDiv64 sets z = x/y.
func kernDiv64(x, y, z *[64]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64(z, k, kernLd64(x, k).Div(kernLd64(y, k)))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernReverb64 sets r = (r + p*s) * d.
func kernReverb64(r, p, s *[64]float32, d float32) {
	dv := kernSplat4(d)
	f := func(k int) {
		_ = dv // a captured vector makes the compiler inline the closure
		q := kernLd64(p, k).Mul(kernLd64(s, k))
		kernSt64(r, k, kernLd64(r, k).Add(q).Mul(dv))
	}
	g := func(k int) {
		_ = dv
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}

// kernReverbScalar64 sets r = (r + p*s) * d.
func kernReverbScalar64(r, p *[64]float32, s, d float32) {
	sv := kernSplat4(s)
	dv := kernSplat4(d)
	f := func(k int) {
		_ = sv // a captured vector makes the compiler inline the closure
		q := kernLd64(p, k).Mul(sv)
		kernSt64(r, k, kernLd64(r, k).Add(q).Mul(dv))
	}
	g := func(k int) {
		_ = sv
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
}
