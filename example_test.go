package goaec3_test

import (
	"log"

	"github.com/GetStream/goaec3"
)

// Example cancels the echo of a mono loudspeaker signal from a mono
// microphone signal at 48 kHz.
func Example() {
	const sampleRate = 48000
	ec, err := goaec3.New(goaec3.DefaultConfig(), sampleRate, 1, 1)
	if err != nil {
		log.Fatal(err)
	}

	// One 10 ms frame per channel: sampleRate/100 samples in [-1, 1].
	render := [][]float32{make([]float32, ec.FrameSize())}
	capture := [][]float32{make([]float32, ec.FrameSize())}

	for i := 0; i < 100; i++ {
		// Fill render with the frame played by the loudspeaker and capture
		// with the frame recorded by the microphone, then:
		if err := ec.AnalyzeRender(render); err != nil {
			log.Fatal(err)
		}
		if err := ec.ProcessCapture(capture); err != nil { // echo removed in place
			log.Fatal(err)
		}
	}

	m := ec.Metrics()
	log.Printf("ERLE %.1f dB, delay %d ms", m.EchoReturnLossEnhancement, m.DelayMs)
}
