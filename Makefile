.PHONY: test test-purego test-simd test-amd64 test-all test-race oracle oracle-src oracle-src-scalar
.PHONY: oracle-stock oracle-src-stock oracle-src-rosetta oracle-src-rosetta-avx2 oracle-x86-kernels
.PHONY: fetch-webrtc gen-src-oracle lint bench vuln fuzz

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

# The race detector over the short test suite, without and with archsimd.
test-race:
	$(GO) test -race -short ./...
	$(SIMD) $(GO) test -race -short ./...

# Known vulnerabilities in the code the packages and their tests reach,
# without and with archsimd.
GOVULNCHECK := golang.org/x/vuln/cmd/govulncheck@v1.10.0
vuln:
	$(GO) run $(GOVULNCHECK) -test ./...
	$(SIMD) $(GO) run $(GOVULNCHECK) -test ./...

# The fuzz targets of the public API and the band splitting, FUZZTIME each.
FUZZTIME ?= 10m
fuzz:
	$(SIMD) $(GO) test -run '^$$' -fuzz FuzzEchoCanceller -fuzztime $(FUZZTIME) .
	$(SIMD) $(GO) test -run '^$$' -fuzz FuzzBandSplitter -fuzztime $(FUZZTIME) ./bands

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

# goaec3 against the prebuilt libwebrtc linked against Apple libm, without
# the interposer that substitutes the Go exp2f, exp10f and log10f: every
# frame of every scenario must still be bit-exact (darwin/arm64).
oracle-stock:
	$(SIMD) $(GO) test -timeout $(ORACLE_TIMEOUT) -tags 'webrtc_oracle webrtc_oracle_stock' -run '^TestEc3StockLibmDivergence$$' -v ./internal/webrtcoracle/

# The same against the source build on the host libm: bit-exact required on
# darwin (Apple libm); elsewhere (glibc) the divergence is logged, not failed.
oracle-src-stock: fetch-webrtc
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) test -timeout $(ORACLE_TIMEOUT) -tags '$(SRC_TAGS) webrtc_oracle_stock' -run '^TestEc3StockLibmDivergence$$' -v ./internal/webrtcoracle/

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

# The AVX2 lane under Rosetta 2 (Rosetta executes AVX2 and FMA without CPUID
# reporting them): GOAEC3_FORCE_LANE, read by the oracle's TestMain only,
# moves Go and the C++ build to AVX2, and GOAEC3_FORCE_FMA runs the FMA
# kernels.
oracle-src-rosetta-avx2: fetch-webrtc
	$(ROSETTA_ENV) $(SIMD) GOAEC3_FORCE_FMA=1 GOAEC3_FORCE_LANE=avx2 GOAEC3_EXPECT_LANE=avx2 $(GO) test -timeout $(ORACLE_TIMEOUT) -tags $(SRC_TAGS) ./internal/webrtcoracle/

# x86 SSE2/AVX2 Ooura FFT, spectrum and vector-math kernels against their C++
# sources: natively on amd64, under Rosetta 2 elsewhere (Apple silicon).
oracle-x86-kernels: fetch-webrtc
	@if [ "$$($(GO) env GOHOSTARCH)" = amd64 ]; then \
		echo CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) test -tags ooura_sse2_oracle ./internal/webrtcoracle/; \
		CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) test -tags ooura_sse2_oracle ./internal/webrtcoracle/; \
	else \
		echo $(ROSETTA_ENV) $(SIMD) $(GO) test -tags ooura_sse2_oracle ./internal/webrtcoracle/; \
		$(ROSETTA_ENV) $(SIMD) $(GO) test -tags ooura_sse2_oracle ./internal/webrtcoracle/; \
	fi

lint:
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	$(GO) vet ./...
	$(SIMD) $(GO) vet ./...
	$(GO) vet -tags purego ./...
	GOARCH=arm64 $(SIMD) $(GO) vet ./...
	GOARCH=amd64 $(SIMD) $(GO) vet ./...
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(GO) vet -tags $(SRC_TAGS) ./...
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) vet -tags '$(SRC_TAGS) purego' ./...
	CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) vet -tags '$(SRC_TAGS) webrtc_oracle_stock' ./...
	@if [ -d tmp_check/webrtc-m141/modules ]; then bash scripts/gen-src-oracle.sh -check; else echo "lint: tmp_check/webrtc-m141 missing, gen-src-oracle -check skipped"; fi

# Go public API against libwebrtc's EchoCanceller3 (ns per 10 ms frame); the
# prebuilt library on darwin/arm64, the source build elsewhere.
bench:
	@if [ -f tmp_check/libwebrtc/lib/libwebrtc.a ]; then \
		$(SIMD) $(GO) test -tags webrtc_oracle -run '^$$' -bench BenchmarkEc3VsLibwebrtc -benchtime 2000x -count 5 ./internal/webrtcoracle/; \
	else \
		CC=$(CLANG_CC) CXX=$(CLANG_CXX) $(SIMD) $(GO) test -tags $(SRC_TAGS) -run '^$$' -bench BenchmarkEc3VsLibwebrtc -benchtime 2000x -count 5 ./internal/webrtcoracle/; \
	fi
