export CC=clang-20
export AR=llvm-ar-20
export NM=llvm-nm-20
export RANLIB=llvm-ranlib-20

export CFLAGS="-O3"

export LDFLAGS="-fuse-ld=lld"

./configure --without-icu --without-readline --without-zlib --without-lz4 --prefix=/home/postgres/postgres_lto_install

