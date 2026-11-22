//go:build !amd64

package aec3

func hasAVX2() bool { return false }
