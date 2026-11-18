//go:build webrtc_src_oracle && !webrtc_oracle && webrtc_oracle_stock

package webrtcoracle

func init() { srcStockLibm = true }
