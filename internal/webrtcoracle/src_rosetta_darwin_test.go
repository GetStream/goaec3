//go:build webrtc_src_oracle && !webrtc_oracle

package webrtcoracle

import "syscall"

// srcRosetta reports whether the process runs under Rosetta 2.
func srcRosetta() bool {
	v, err := syscall.SysctlUint32("sysctl.proc_translated")
	return err == nil && v == 1
}
