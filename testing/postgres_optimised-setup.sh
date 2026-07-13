export CFLAGS="-O3 -fpass-plugin=/home/adrianj/IOOptCompilerPass/build/llvm-src/IOOpt.so"
export CC=clang
export CXX=clang++
./configure --without-icu --without-readline --without-zlib --without-lz4 --prefix=/home/adrianj/IOOptCompilerPass/testing/postgresql-optimised-install

