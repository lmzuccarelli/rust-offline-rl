# for static build using x86_64-unknown-linux-gnu ensure glibc-static (rhel based) is installed

.PHONY: all clean-all build build-static verify fmt test

all: clean-all build

build-static:
	RUSTFLAGS="-C target-feature=+crt-static -Clink-arg=-fuse-ld=lld" cargo build --target x86_64-unknown-linux-gnu --release --verbose 

build:
	cargo build --release 

verify:
	cargo clippy --all-targets --all-features

fmt:
	rustfmt --check src/*.rs --edition 2024

test:
	cargo test -- --nocapture --test-threads=1

clean-all:
	rm -rf cargo-test*
	cargo clean
	rm -rf ./target/debug
