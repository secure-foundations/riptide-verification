WaveCert: Translation Validation for RipTide
---

This branch is for OOPSLA 2024 artifact evaluation.
Please see the main branch for more information.

To build the artifact image, run
```
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t flowcert -m 32g --build-arg MAKE_JOBS=8 --build-arg BUILDKIT_INLINE_CACHE=1 .
```

```
DOCKER_BUILDKIT=0 docker build -f Dockerfile --build-arg MAKE_JOBS=8 .
```
