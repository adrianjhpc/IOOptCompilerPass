export CC=clang-20
export AR=llvm-ar-20
export NM=llvm-nm-20
export RANLIB=llvm-ranlib-20

export CFLAGS="-O3 -flto=full -fpass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so"

export LDFLAGS="-flto=full -fuse-ld=lld -Wl,--load-pass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so"

./configure --without-icu --without-readline --without-zlib --without-lz4 --prefix=/home/postgres/postgres_lto_optimised_install

