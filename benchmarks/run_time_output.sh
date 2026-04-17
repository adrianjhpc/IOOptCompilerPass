python3 repeat_time_io.py --iterations 10 --out-file streaming_output.log benchmarks/media_standard benchmarks/media_fast 
python3 repeat_time_io.py --iterations 10 --out-file wal_output.log benchmarks/wal_standard benchmarks/wal_fast 
python3 repeat_time_io.py --iterations 10 --out-file http_output.log benchmarks/http_standard benchmarks/http_fast 
python3 repeat_time_io.py --iterations 10 --out-file lto_benchmark.log benchmarks/lto_standard benchmarks/lto_fast 
python3 repeat_time_io.py --iterations 10 --out-file output.ppm benchmarks/pixel_write_standard benchmarks/pixel_write_fast 
python3 repeat_time_mpi_io.py --iterations 10 --out-file mpi_ioopt_benchmark.dat ./benchmarks/mpi_write_at_standard benchmarks/mpi_write_at_fast

