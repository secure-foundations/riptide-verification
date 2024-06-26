ARG MAKE_JOBS=1

ARG LLVM_VERSION=12.0.0
ARG LLVM_TARBALL_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-${LLVM_VERSION}.src.tar.xz
ARG CLANG_TARBALL_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/clang-${LLVM_VERSION}.src.tar.xz

ARG LIBDC_AARCH64_PATH=sdc/libDC.aarch64.so
ARG LIBDC_X86_64_PATH=sdc/libDC.x86_64.so

ARG Z3_GIT=https://github.com/Z3Prover/z3.git
ARG Z3_BRANCH=z3-4.12.2

ARG VERUS_GIT=https://github.com/zhengyao-lin/verus.git
ARG VERUS_BRANCH=oopsla2024-ae

FROM ubuntu:22.04 AS build

RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git wget zip curl

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
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="$PATH:/root/.cargo/bin"

ARG Z3_GIT
ARG Z3_BRANCH
RUN git clone --depth 1 --branch ${Z3_BRANCH} ${Z3_GIT} build/z3 && \
    cd build/z3 && \
    python3 scripts/mk_make.py && \
    cd build && \
    make
# NOTE: `make -j n` in docker build leaks a lot of memory

ARG VERUS_GIT
ARG VERUS_BRANCH
RUN git clone --depth 1 --branch ${VERUS_BRANCH} ${VERUS_GIT} build/verus && \
    cd build/verus/source && \
    cp /build/z3/build/z3 . && \
    bash -c ". ../tools/activate && vargo build --release"

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
ENV PATH="$PATH:/build/llvm/bin"
ENV PATH="$PATH:/build/verus/bin"
