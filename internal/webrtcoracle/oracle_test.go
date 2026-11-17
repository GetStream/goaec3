//go:build webrtc_oracle

package webrtcoracle

import "testing"

func TestLinks(t *testing.T) {
	var a [128]float32
	a[1] = 1
	OouraFft(&a, false)
	t.Log(a[:8])
}
