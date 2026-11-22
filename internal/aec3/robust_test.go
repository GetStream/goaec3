package aec3

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func robustConfigLeaves(v reflect.Value, out []reflect.Value) []reflect.Value {
	if v.Kind() == reflect.Struct {
		for i := 0; i < v.NumField(); i++ {
			out = robustConfigLeaves(v.Field(i), out)
		}
		return out
	}
	return append(out, v)
}

// TestValidateIdempotent sets random leaves of the config to random values,
// NaN and ±Inf included. After Validate the config is valid: a second
// Validate reports true and changes nothing.
func TestValidateIdempotent(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	specialF := []float32{0, -0.0, 1, -1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32}
	specialI := []int{0, 1, -1, 64, 65, 250, 5000, 5001, math.MaxInt32, math.MinInt32, math.MaxInt, math.MinInt}
	for iter := 0; iter < 20000; iter++ {
		cfg := DefaultEchoCanceller3Config()
		leaves := robustConfigLeaves(reflect.ValueOf(&cfg).Elem(), nil)
		for range 1 + rng.Intn(20) {
			l := leaves[rng.Intn(len(leaves))]
			switch l.Kind() {
			case reflect.Bool:
				l.SetBool(rng.Intn(2) == 0)
			case reflect.Int:
				if rng.Intn(2) == 0 {
					l.SetInt(int64(specialI[rng.Intn(len(specialI))]))
				} else {
					l.SetInt(int64(rng.Intn(200) - 50))
				}
			case reflect.Float32:
				if rng.Intn(2) == 0 {
					l.SetFloat(float64(specialF[rng.Intn(len(specialF))]))
				} else {
					l.SetFloat(float64(math.Float32frombits(rng.Uint32())))
				}
			}
		}
		cfg.Validate()
		before := cfg
		if !cfg.Validate() {
			t.Fatalf("iteration %d: a validated config fails Validate:\n%+v\n%+v", iter, before, cfg)
		}
		if !reflect.DeepEqual(robustNaNFree(before), robustNaNFree(cfg)) {
			t.Fatalf("iteration %d: a second Validate changes the config", iter)
		}
	}
}

// robustNaNFree replaces the NaN leaves, which Validate leaves alone in the
// fields it does not check, so that DeepEqual compares the rest.
func robustNaNFree(cfg EchoCanceller3Config) EchoCanceller3Config {
	for _, l := range robustConfigLeaves(reflect.ValueOf(&cfg).Elem(), nil) {
		if l.Kind() == reflect.Float32 && math.IsNaN(l.Float()) {
			l.SetFloat(12345)
		}
	}
	return cfg
}

// TestRingOffsetIndexRange checks that OffsetIndex stays in the ring for any
// index in the ring and any offset.
func TestRingOffsetIndexRange(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	offsets := []int{0, 1, -1, math.MaxInt32, math.MinInt32, math.MaxInt / 3, -(math.MaxInt / 3)}
	for _, size := range []int{1, 2, 3, 17, 64, 250} {
		r := ringIndex{Size: size}
		for index := 0; index < size; index++ {
			for i := 0; i < 50; i++ {
				o := rng.Intn(10*size+1) - 5*size
				if i < len(offsets) {
					o = offsets[i]
				}
				p := r.OffsetIndex(index, o)
				if p < 0 || p >= size {
					t.Fatalf("size %d: OffsetIndex(%d, %d) = %d", size, index, o, p)
				}
				// Where the int arithmetic of the C++ is defined and not
				// negative, the result is (size + index + offset) % size.
				if size+index+o >= 0 && p != (size+index+o)%size {
					t.Fatalf("size %d: OffsetIndex(%d, %d) = %d", size, index, o, p)
				}
			}
		}
	}
}

// TestSetAudioBufferDelayIntRange checks the conversion of the external
// delay to a C int: values outside int32 wrap before the conversion to
// blocks, which rounds toward zero.
func TestSetAudioBufferDelayIntRange(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	cfg.Delay.UseExternalDelayEstimator = true
	for _, ms := range []int{0, 3, 4, -3, -4, -5, math.MaxInt32, math.MinInt32, math.MaxInt/2 + 1, math.MaxInt,
		math.MinInt, math.MaxInt/4 + 8} {
		b := NewRenderDelayBuffer(&cfg, 16000, 1)
		b.SetAudioBufferDelay(ms)
		if want := int(int32(ms) / 4); b.externalAudioBufferDelay != want {
			t.Errorf("SetAudioBufferDelay(%d): %d blocks, want %d", ms, b.externalAudioBufferDelay, want)
		}
		b.AlignFromExternalDelay()
		b.Reset()
		if d := b.Delay(); d < -b.blocks.Size || d > b.blocks.Size {
			t.Errorf("SetAudioBufferDelay(%d): delay %d after Reset", ms, d)
		}
	}
}
