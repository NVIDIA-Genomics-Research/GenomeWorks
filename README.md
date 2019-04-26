# GenomeWorks

## Clone GenomeWorks
```bash
git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/genomics/GenomeWorks.git
```

## Build GenomeWorks
To build GenomeWorks -

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install
make -j install
```

## Enable Unit Tests
To enable unit tests, add `-Dgw_enable_tests=ON` to the `cmake` command in the build step.

