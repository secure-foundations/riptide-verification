ARG MAKE_JOBS=1

ARG LLVM_VERSION=12.0.0
ARG LLVM_TARBALL_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-${LLVM_VERSION}.src.tar.xz
ARG CLANG_TARBALL_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/clang-${LLVM_VERSION}.src.tar.xz

ARG LIBDC_AARCH64_PATH=sdc/libDC.aarch64.so
ARG LIBDC_X86_64_PATH=sdc/libDC.x86_64.so

ARG Z3_GIT=https://github.com/Z3Prover/z3.git
ARG Z3_BRANCH=z3-4.12.2

# NOTE: the Rust version must match the version used by Verus
ARG RUST_VERSION=1.73.0
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

# Build Z3
ARG Z3_GIT
ARG Z3_BRANCH
RUN git clone --depth 1 --branch ${Z3_BRANCH} ${Z3_GIT} build/z3 && \
    cd build/z3 && \
    python3 scripts/mk_make.py && \
    cd build && \
    make
# NOTE: `make -j n` in docker build leaks a lot of memory

# Install Rust
ARG RUST_VERSION
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain=${RUST_VERSION}
ENV PATH="$PATH:/root/.cargo/bin"

# Build Verus
ARG VERUS_GIT
ARG VERUS_BRANCH
RUN git clone --depth 1 --branch ${VERUS_BRANCH} ${VERUS_GIT} build/verus && \
    cd build/verus/source && \
    cp /build/z3/build/z3 . && \
    bash -c ". ../tools/activate && vargo build --release"

FROM ubuntu:22.04 AS final

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Copy FlowCert and install dependencies
COPY requirements.txt /build/flowcert/requirements.txt
RUN python3 -m pip install -r /build/flowcert/requirements.txt

# Copy the precompiled RipTide compiler depending on architecture
ARG LIBDC_AARCH64_PATH
ARG LIBDC_X86_64_PATH
COPY ${LIBDC_AARCH64_PATH} /build/dc/libDC.aarch64.so
COPY ${LIBDC_X86_64_PATH} /build/dc/libDC.x86_64.so
RUN arch=`uname -m` && \
    if [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; then \
        mv /build/dc/libDC.x86_64.so /build/dc/libDC.so && \
        rm /build/dc/libDC.aarch64.so; \
    elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
        mv /build/dc/libDC.aarch64.so /build/dc/libDC.so && \
        rm /build/dc/libDC.x86_64.so; \
    else \
        echo "unsupported architecture $arch"; \
        exit 1; \
    fi

# Copy LLVM binaries
COPY --from=build /build/llvm/llvm-12.0.0.src/build/bin /build/llvm/bin

# Copy Rust binaries
COPY --from=build /root/.rustup /root/.rustup
COPY --from=build /root/.cargo /root/.cargo
RUN rm -r /root/.rustup/toolchains/*/bin && \
    rm -r /root/.rustup/toolchains/*/etc && \
    rm -r /root/.rustup/toolchains/*/libexec && \
    rm -r /root/.rustup/toolchains/*/share && \
    rm -r /root/.rustup/toolchains/*/lib/rustlib/*-linux-gnu/bin && \
    rm -r /root/.cargo/registry

# Copy Verus binaries
COPY --from=build /build/verus/source/target-verus/release /build/verus/bin

# Copy FlowCert files
COPY confluence /build/flowcert/confluence
COPY evaluations /build/flowcert/evaluations
COPY semantics /build/flowcert/semantics
COPY tools /build/flowcert/tools
COPY utils /build/flowcert/utils
COPY README.md /build/flowcert/README.md

ADD image/paper.pdf /build/flowcert/paper.pdf
ADD image/init.sh /root/init.sh
RUN echo ". ~/init.sh" >> /root/.bashrc

# Collapse all layers
FROM scratch
COPY --from=final / /
WORKDIR /build/flowcert
ENV LLVM_12_BIN=/build/llvm/bin \
    LIBDC_PATH=/build/dc/libDC.so \
    PATH="$PATH:/root/.cargo/bin:/build/llvm/bin:/build/verus/bin"
CMD ["/bin/bash"]
