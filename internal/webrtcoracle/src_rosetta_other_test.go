//go:build webrtc_src_oracle && !webrtc_oracle && !darwin

package webrtcoracle

func srcRosetta() bool { return false }
