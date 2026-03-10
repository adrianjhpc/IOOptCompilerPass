export CFLAGS="-O3 -fpass-plugin=/mnt/pvc/IOCompiler/build/src/libIOOpt.so"
export CC=clang-20
export CXX=clang++-20
./configure --without-icu --without-readline --without-zlib --without-lz4 --prefix=/home/postgres/postgres_optimised_install

