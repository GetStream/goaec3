//go:build !(goexperiment.simd && amd64 && !purego)

package ooura

func poisonScratch(*PairScratch) {}
