package aec3

import "runtime"

const (
	isAMD64 = runtime.GOARCH == "amd64"
	isARM64 = runtime.GOARCH == "arm64"
)
