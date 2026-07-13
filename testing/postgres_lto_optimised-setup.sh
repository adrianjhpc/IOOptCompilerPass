export CC=clang
export AR=llvm-ar
export NM=llvm-nm
export RANLIB=llvm-ranlib

export CFLAGS="-O3 -flto=full -fpass-plugin=/home/adrianj/IOOptCompilerPass/build/llvm-src/IOOpt.so"

export LDFLAGS="-flto=full -fuse-ld=lld -Wl,--load-pass-plugin=/home/adrianj/IOOptCompilerPass/build/llvm-src/IOOpt.so"

./configure --without-icu --without-readline --without-zlib --without-lz4 --prefix=/home/adrianj/IOOptCompilerPass/testing/postgresql-lto-optimised-install

