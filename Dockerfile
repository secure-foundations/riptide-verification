ARG MAKE_JOBS=1

ARG LLVM_VERSION=12.0.0
ARG LLVM_TARBALL_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-${LLVM_VERSION}.src.tar.xz
ARG CLANG_TARBALL_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/clang-${LLVM_VERSION}.src.tar.xz

ARG LIBDC_AARCH64_PATH=sdc/libDC.aarch64.so
ARG LIBDC_X86_64_PATH=sdc/libDC.x86_64.so

ARG VERUS_GIT=git@github.com:verus-lang/verus.git
ARG VERUS_COMMIT=5639d8008a61eb7545df4a6e7d2bc38be3d09fb0

FROM ubuntu:22.04 AS build

RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git wget

# Download and build LLVM
ARG MAKE_JOBS
ARG LLVM_VERSION
ARG LLVM_TARBALL_URL
ARG CLANG_TARBALL_URL
RUN mkdir -p build/llvm && \
    cd build/llvm && \
    wget -q ${LLVM_TARBALL_URL} && \
    wget -q ${CLANG_TARBALL_URL} && \
    tar xf llvm-${LLVM_VERSION}.src.tar.xz && \
    tar xf clang-${LLVM_VERSION}.src.tar.xz && \
    cd llvm-${LLVM_VERSION}.src && \
    mv ../clang-${LLVM_VERSION}.src tools/clang && \
    cmake -B build -DLLVM_ENABLE_ASSERTIONS=On -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD= && \
    cmake --build build --target opt clang -j ${MAKE_JOBS}

# Download and build Verus
# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

ARG VERUS_GIT
ARG VERUS_COMMIT
RUN git clone ${VERUS_GIT} build/verus && \
    cd build/verus && \
    git checkout ${VERUS_COMMIT} && \
    ./tools/get-z3.sh && \
    source ../tools/activate && \
    vargo build --release

FROM ubuntu:22.04 AS final

COPY --from=build /build/llvm/llvm-12.0.0.src/build/bin /build/llvm/bin

COPY --from=build /build/verus/source/target-verus/release/verus /build/verus/bin
COPY --from=build /build/verus/source/target-verus/release/verus-root /build/verus/bin
COPY --from=build /build/verus/source/target-verus/release/z3 /build/verus/bin
COPY --from=build /build/verus/source/target-verus/release/rust_verify /build/verus/bin

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Copy the recompiled binary depending on architecture
ARG LIBDC_AARCH64_PATH
ARG LIBDC_X86_64_PATH
COPY ${LIBDC_AARCH64_PATH} build/dc/libDC.aarch64.so
COPY ${LIBDC_X86_64_PATH} build/dc/libDC.x86_64.so

RUN arch=`uname -m` && \
    if [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; then \
        mv build/dc/libDC.x86_64.so build/dc/libDC.so && \
        rm build/dc/libDC.aarch64.so; \
    elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
        mv build/dc/libDC.aarch64.so build/dc/libDC.so && \
        rm build/dc/libDC.x86_64.so; \
    else \
        echo "unsupported architecture $arch"; \
        exit 1; \
    fi

WORKDIR /build/flowcert
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY evaluations evaluations
COPY semantics semantics
COPY tools tools
COPY utils utils

ENV LLVM_12_BIN=/build/llvm/bin
ENV LIBDC_PATH=/build/dc/libDC.so
