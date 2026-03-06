export LD_LIBRARY_PATH=/home/postgres/postgres_original_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/postgres/postgres_lto_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/postgres/postgres_optimised_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/postgres/postgres_lto_optimised_install/lib:$LD_LIBRARY_PATH
rm -rf ./test_db
./bin/initdb -D ./test_db
sleep 1
./bin/pg_ctl -D ./test_db -l logfile start
sleep 1
./bin/createdb test_perf
./bin/pgbench -i -s 10 test_perf
echo " "
echo "Write test"
./bin/pgbench -c 10 -j 2 -T 60 test_perf
echo " "
./bin/pg_ctl -D ./test_db stop
rm -rf ./test_db
./bin/initdb -D ./test_db
sleep 1
./bin/pg_ctl -D ./test_db -l logfile start
sleep 1
./bin/createdb test_perf
./bin/pgbench -i -s 10 test_perf
echo " "
echo "Read test"
./bin/pgbench -S -c 20 -j 4 -T 60 test_perf
echo " "
./bin/pg_ctl -D ./test_db stop
