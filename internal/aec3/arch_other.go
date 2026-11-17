//go:build !amd64 && !arm64

package aec3

const (
	isAMD64 = false
	isARM64 = false
)

func hasAVX2() bool { return false }
