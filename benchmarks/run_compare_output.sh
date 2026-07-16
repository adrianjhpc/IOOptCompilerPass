python3 compare_io.py --out-file streaming_output.log benchmarks/media_standard benchmarks/media_fast 
python3 compare_io.py --out-file wal_output.log benchmarks/wal_standard benchmarks/wal_fast 
python3 compare_io.py --out-file http_output.log benchmarks/http_standard benchmarks/http_fast 
python3 compare_io.py --out-file lto_benchmark.log benchmarks/lto_standard benchmarks/lto_fast 
python3 compare_io.py --out-file output.ppm benchmarks/pixel_write_standard benchmarks/pixel_write_fast 
python3 compare_io.py --out-file output.dat benchmarks/loop_standard benchmarks/loop_fast
python3 compare_mpi_io.py --out-file mpi_ioopt_benchmark.dat ./benchmarks/mpi_write_at_standard benchmarks/mpi_write_at_fast

