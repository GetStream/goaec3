//go:build webrtc_oracle && webrtc_oracle_stock

package webrtcoracle

import (
	"fmt"
	"math"
	"testing"

	"github.com/GetStream/goaec3"
)

// TestEc3StockLibmDivergence measures how far goaec3 is from libwebrtc
// linked against the host libm, without the interposer that routes exp2f,
// __exp10f and log10f to the Go implementations (build with
// -tags 'webrtc_oracle webrtc_oracle_stock' and run this test alone; the
// bit-exact tests fail by design in that build). It logs, per scenario, the
// first frame whose output differs, the output SNR of Go against the C++
// and the largest sample difference, and fails only if the SNR drops below
// 60 dB.
func TestEc3StockLibmDivergence(t *testing.T) {
	const seconds = 60
	for _, rate := range []int{16000, 32000, 48000} {
		for _, ch := range [][2]int{{1, 1}, {2, 1}, {2, 2}} {
			for _, seed := range []int64{1, 2} {
				name := fmt.Sprintf("%dHz_r%d_c%d_seed%d", rate, ch[0], ch[1], seed)
				t.Run(name, func(t *testing.T) {
					cfg := goaec3.DefaultConfig()
					cfg.Filter.ExportLinearAecOutput = true
					cpp := NewEc3Canceller(&cfg, nil, rate, ch[0], ch[1], ec3Scalar)
					defer cpp.Free()
					e, err := goaec3.New(cfg, rate, ch[0], ch[1])
					if err != nil {
						t.Fatal(err)
					}
					n := rate / 100
					scene := newEc3Scene(rate, ch[0], ch[1], 60+20*int(seed), seed)
					render := ec3Frame(ch[0], n)
					capture := ec3Frame(ch[1], n)
					lin := ec3Frame(ch[1], goaec3.LinearOutputFrameSize)
					cppLin := make([]float32, ch[1]*160)
					var flat, cppCap []float32
					firstDiff, firstMetricDiff := -1, -1
					var sig, noise, maxDiff float64
					for frame := 0; frame < seconds*100; frame++ {
						scene.next(render, capture)
						flat = ec3Flat(render, flat)
						cpp.AnalyzeRenderFloat(flat)
						if err := e.AnalyzeRender(render); err != nil {
							t.Fatal(err)
						}
						cppCap = ec3Flat(capture, cppCap)
						cpp.ProcessCaptureFloat(cppCap, cppLin, false)
						if err := e.ProcessCaptureWithLinearOutput(capture, lin); err != nil {
							t.Fatal(err)
						}
						for c := range capture {
							for k, v := range capture[c] {
								w := cppCap[c*n+k]
								if math.Float32bits(v) != math.Float32bits(w) && firstDiff < 0 {
									firstDiff = frame
								}
								d := float64(v) - float64(w)
								sig += float64(w) * float64(w)
								noise += d * d
								maxDiff = max(maxDiff, math.Abs(d))
							}
						}
						gm, cm := e.Metrics(), cpp.Metrics()
						if firstMetricDiff < 0 && (gm.EchoReturnLoss != cm.EchoReturnLoss ||
							gm.EchoReturnLossEnhancement != cm.EchoReturnLossEnhancement || gm.DelayMs != cm.DelayMs) {
							firstMetricDiff = frame
						}
					}
					snr := math.Inf(1)
					if noise > 0 {
						snr = 10 * math.Log10(sig/noise)
					}
					t.Logf("first differing output frame %d, first differing metrics frame %d, output SNR %.1f dB, "+
						"max |diff| %.3g", firstDiff, firstMetricDiff, snr, maxDiff)
					if snr < 60 {
						t.Errorf("output SNR %.1f dB", snr)
					}
				})
			}
		}
	}
}
