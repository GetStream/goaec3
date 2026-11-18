.PHONY: test test-purego test-simd test-amd64 test-all oracle oracle-src oracle-src-scalar
.PHONY: oracle-src-rosetta oracle-src-rosetta-avx2 oracle-x86-kernels fetch-webrtc gen-src-oracle lint bench

GO ?= go
# The C++ reference builds need clang: libwebrtc is built by clang with
# contraction off, which the sources request through a clang pragma.
CLANG_CC ?= clang
CLANG_CXX ?= clang++
ORACLE_TIMEOUT ?= 40m
SIMD := GOEXPERIMENT=simd
SRC_TAGS := webrtc_src_oracle
ROSETTA_ENV := CGO_ENABLED=1 GOARCH=amd64 CC="$(CLANG_CC) -arch x86_64" CXX="$(CLANG_CXX) -arch x86_64"

# Default lane, pure-Go lane references (no GOEXPERIMENT).
test:
	$(GO) test ./...

# Scalar lane.
test-purego:
	$(GO) test -tags purego ./...

# Default lane with the archsimd kernels.
test-simd:
	$(SIMD) $(GO) test ./...

# amd64 lanes (under Rosetta 2 on Apple silicon).
test-amd64:
	GOARCH=amd64 $(GO) test ./...
	GOARCH=amd64 $(SIMD) $(GO) test ./...

test-all: test test-purego test-simd test-amd64

# Sparse checkout of the pinned WebRTC m141 sources and abseil-cpp into
# tmp_check/webrtc-m141.
fetch-webrtc:
	bash scripts/fetch-webrtc.sh

# Regenerate the src_gen_* files of the webrtc_src_oracle build.
gen-src-oracle:
	bash scripts/gen-src-oracle.sh

# Parity against the prebuilt libwebrtc.a (darwin/arm64, needs
# tmp_check/libwebrtc): default and scalar C++ references, with and without
# the archsimd kernels.
oracle:
	$(GO) test -timeout $(ORACLE_TIMEOUT) -tags webrtc_oracle ./internal/webrtcoracle/
	$(SIMD) $(GO) test -timeout $(ORACLE_TIMEOUT) -tags webrtc_oracle ./internal/webrtcoracle/
	$(GO) test -timeout $(ORACLE_TIMEOUT) -tags 'webrtc_oracle purego' ./internal/webrtcoracle/
	$(SIMD) $(GO) test -timeout $(ORACLE_TIMEOUT) -tags 'webrtc_oracle purego' ./internal/webrtcoracle/

# End-to-end and kernel parity against the C++ reference built from the
# WebRTC sources, in this machine's native lane (NEON on arm64; AVX2 or SSE2
# on amd64 as libwebrtc's cpu_info would pick), with and without archsimd.
oracle-src: fetch-webrtc
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(GO) test -timeout $(ORACLE_TIMEOUT) -tags $(SRC_TAGS) ./internal/webrtcoracle/
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) test -timeout $(ORACLE_TIMEOUT) -tags $(SRC_TAGS) ./internal/webrtcoracle/

# The scalar lane (kNone, scalar Ooura) of the source build.
oracle-src-scalar: fetch-webrtc
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) GOAEC3_EXPECT_LANE=none $(GO) test -timeout $(ORACLE_TIMEOUT) -tags '$(SRC_TAGS) purego' ./internal/webrtcoracle/

# The x86_64 source build under Rosetta 2 on Apple silicon: the SSE2 lane
# (Go's AVX2 detection turned off) and the scalar lane.
oracle-src-rosetta: fetch-webrtc
	$(ROSETTA_ENV) $(SIMD) GODEBUG=cpu.avx2=off GOAEC3_EXPECT_LANE=sse2 $(GO) test -timeout $(ORACLE_TIMEOUT) -tags $(SRC_TAGS) ./internal/webrtcoracle/
	$(ROSETTA_ENV) GODEBUG=cpu.avx2=off GOAEC3_EXPECT_LANE=sse2 $(GO) test -timeout $(ORACLE_TIMEOUT) -tags $(SRC_TAGS) ./internal/webrtcoracle/
	$(ROSETTA_ENV) GOAEC3_EXPECT_LANE=none $(GO) test -timeout $(ORACLE_TIMEOUT) -tags '$(SRC_TAGS) purego' ./internal/webrtcoracle/

# The AVX2 lane under Rosetta 2, where Go selects it (Rosetta executes AVX2
# and FMA without CPUID reporting them, so the FMA kernels are forced). It
# fails on the lane check when Go's detection does not select AVX2 there.
oracle-src-rosetta-avx2: fetch-webrtc
	$(ROSETTA_ENV) $(SIMD) GOAEC3_FORCE_FMA=1 GOAEC3_FORCE_LANE=avx2 GOAEC3_EXPECT_LANE=avx2 $(GO) test -timeout $(ORACLE_TIMEOUT) -tags $(SRC_TAGS) ./internal/webrtcoracle/

# x86 SSE2/AVX2 Ooura FFT, spectrum and vector-math kernels against their C++
# sources (under Rosetta 2 on Apple silicon).
oracle-x86-kernels:
	$(ROSETTA_ENV) $(SIMD) $(GO) test -tags ooura_sse2_oracle ./internal/webrtcoracle/

lint:
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	$(GO) vet ./...
	$(SIMD) $(GO) vet ./...
	$(GO) vet -tags purego ./...
	$(GO) vet -tags $(SRC_TAGS) ./...
	$(SIMD) $(GO) vet -tags '$(SRC_TAGS) purego' ./...
	@if [ -d tmp_check/webrtc-m141/modules ]; then bash scripts/gen-src-oracle.sh -check; else echo "lint: tmp_check/webrtc-m141 missing, gen-src-oracle -check skipped"; fi

# Go public API against libwebrtc's EchoCanceller3 (ns per 10 ms frame); the
# prebuilt library on darwin/arm64, the source build elsewhere.
bench:
	@if [ -f tmp_check/libwebrtc/lib/libwebrtc.a ]; then \
		$(SIMD) $(GO) test -tags webrtc_oracle -run '^$$' -bench BenchmarkEc3VsLibwebrtc -benchtime 2000x -count 5 ./internal/webrtcoracle/; \
	else \
		CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) test -tags $(SRC_TAGS) -run '^$$' -bench BenchmarkEc3VsLibwebrtc -benchtime 2000x -count 5 ./internal/webrtcoracle/; \
	fi
