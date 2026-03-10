cmake .. \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_C_FLAGS="-O3 -flto -fpass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so" \
  -DCMAKE_CXX_FLAGS="-O3 -flto -fpass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -flto -Wl,--load-pass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -flto -Wl,--load-pass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so" \
  -DWITH_TESTS=OFF \
  -DWITH_TOOLS=OFF \
  -DWITH_BENCHMARK_TOOLS=ON


make -j 32 db_bench

