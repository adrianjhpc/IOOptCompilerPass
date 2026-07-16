# File write benchmark

set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pixel_write_benchmark.cpp)
set(BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/pixel_write_benchmark.bc)
set(OPT_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/pixel_write_benchmark.opt.bc)
set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/pixel_write_benchmark.opt.o)

# Standard -O3 Baseline
add_executable(pixel_write_standard ${SRC_FILE})
target_compile_options(pixel_write_standard PRIVATE -O3)

# Custom Pass -O3 (Optimised)
add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${SRC_FILE} -o ${BC_FILE} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${SRC_FILE}
)

add_custom_command(
    OUTPUT ${OPT_BC_FILE}
    COMMAND opt -load-pass-plugin=$<TARGET_FILE:IOOpt> -passes="io-opt" ${BC_FILE} -o ${OPT_BC_FILE}
    DEPENDS ${BC_FILE} IOOpt
)

add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${OPT_BC_FILE} -o ${OBJ_FILE}
    DEPENDS ${OPT_BC_FILE}
)

add_executable(pixel_write_fast ${OBJ_FILE})
set_target_properties(pixel_write_fast PROPERTIES LINKER_LANGUAGE CXX)


# HTTP Server benchmark

set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/http_benchmark.cpp)
set(BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/http_benchmark.bc)
set(OPT_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/http_benchmark.opt.bc)
set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/http_benchmark.opt.o)


#Standard -O3 Baseline
add_executable(http_standard ${SRC_FILE})
target_compile_options(http_standard PRIVATE -O3 -fno-inline)

# Custom Pass -O3 (Optimised)
add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${SRC_FILE} -o ${BC_FILE} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${SRC_FILE}
)

add_custom_command(
    OUTPUT ${OPT_BC_FILE}
    COMMAND opt -load-pass-plugin=$<TARGET_FILE:IOOpt> -passes="io-opt" ${BC_FILE} -o ${OPT_BC_FILE}
    DEPENDS ${BC_FILE} IOOpt
)

add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${OPT_BC_FILE} -o ${OBJ_FILE}
    DEPENDS ${OPT_BC_FILE}
)

add_executable(http_fast ${OBJ_FILE})
set_target_properties(http_fast PROPERTIES LINKER_LANGUAGE CXX)

# Database WAL benchmark

set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/database_wal_benchmark.cpp)
set(BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/database_wal_benchmark.bc)
set(OPT_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/database_wal_benchmark.opt.bc)
set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/database_wal_benchmark.opt.o)

# Baseline
add_executable(wal_standard ${SRC_FILE})
target_compile_options(wal_standard PRIVATE -O3 -fno-inline)

# Optimised
add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${SRC_FILE} -o ${BC_FILE} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${SRC_FILE}
)

add_custom_command(
    OUTPUT ${OPT_BC_FILE}
    COMMAND opt -load-pass-plugin=$<TARGET_FILE:IOOpt> -passes="io-opt" ${BC_FILE} -o ${OPT_BC_FILE}
    DEPENDS ${BC_FILE} IOOpt
)

add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${OPT_BC_FILE} -o ${OBJ_FILE}
    DEPENDS ${OPT_BC_FILE}
)

add_executable(wal_fast ${OBJ_FILE})
set_target_properties(wal_fast PROPERTIES LINKER_LANGUAGE CXX)

# Multimedia streaming benchmark

set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/media_benchmark.cpp)
set(BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/media_benchmark.bc)
set(OPT_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/media_benchmark.opt.bc)
set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/media_benchmark.opt.o)


# Baseline
add_executable(media_standard ${SRC_FILE})
target_compile_options(media_standard PRIVATE -O3 -fno-inline)

# Optimised
add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${SRC_FILE} -o ${BC_FILE} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${SRC_FILE}
)

add_custom_command(
    OUTPUT ${OPT_BC_FILE}
    COMMAND opt -load-pass-plugin=$<TARGET_FILE:IOOpt> -passes="io-opt" ${BC_FILE} -o ${OPT_BC_FILE}
    DEPENDS ${BC_FILE} IOOpt
)

add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${OPT_BC_FILE} -o ${OBJ_FILE}
    DEPENDS ${OPT_BC_FILE}
)

add_executable(media_fast ${OBJ_FILE})
set_target_properties(media_fast PROPERTIES LINKER_LANGUAGE CXX)

# LTO cross-file benchmark

set(LTO_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/lto_bench_main.cpp)
set(LTO_LOG_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/lto_bench_log.cpp)

set(LTO_MAIN_BC  ${CMAKE_CURRENT_BINARY_DIR}/lto_bench_main.bc)
set(LTO_LOG_BC   ${CMAKE_CURRENT_BINARY_DIR}/lto_bench_log.bc)

set(LTO_MERGED_BC ${CMAKE_CURRENT_BINARY_DIR}/lto_merged.bc)
set(LTO_OPT_BC    ${CMAKE_CURRENT_BINARY_DIR}/lto_merged.opt.bc)
set(LTO_OBJ       ${CMAKE_CURRENT_BINARY_DIR}/lto_merged.opt.o)

# Baseline
add_executable(lto_standard
    ${LTO_MAIN_SRC}
    ${LTO_LOG_SRC}
)
target_compile_options(lto_standard PRIVATE -O3)


# Optimised: LTO enabled and Custom Pass Plugin
add_custom_command(
    OUTPUT ${LTO_MAIN_BC}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${LTO_MAIN_SRC} -o ${LTO_MAIN_BC} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${LTO_MAIN_SRC}
)

add_custom_command(
    OUTPUT ${LTO_LOG_BC}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${LTO_LOG_SRC} -o ${LTO_LOG_BC} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${LTO_LOG_SRC}
)

add_custom_command(
    OUTPUT ${LTO_MERGED_BC}
    COMMAND llvm-link ${LTO_MAIN_BC} ${LTO_LOG_BC} -o ${LTO_MERGED_BC}
    DEPENDS ${LTO_MAIN_BC} ${LTO_LOG_BC}
)

add_custom_command(
    OUTPUT ${LTO_OPT_BC}
    COMMAND opt -load-pass-plugin=$<TARGET_FILE:IOOpt> "-passes=cgscc(inline),function(instcombine,io-opt)" ${LTO_MERGED_BC} -o ${LTO_OPT_BC}
    DEPENDS ${LTO_MERGED_BC} IOOpt
    VERBATIM
)

add_custom_command(
    OUTPUT ${LTO_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${LTO_OPT_BC} -o ${LTO_OBJ}
    DEPENDS ${LTO_OPT_BC}
)

add_executable(lto_fast ${LTO_OBJ})
set_target_properties(lto_fast PROPERTIES LINKER_LANGUAGE CXX)

# Loop Collapsing benchmark

set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/loop_benchmark.cpp)
set(BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/loop_benchmark.bc)
set(OPT_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/loop_benchmark.opt.bc)
set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/loop_benchmark.opt.o)

# Baseline
add_executable(loop_standard ${SRC_FILE})
target_compile_options(loop_standard PRIVATE -O3 -fno-inline)

# Optimised
add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${SRC_FILE} -o ${BC_FILE} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
    DEPENDS ${SRC_FILE}
)

add_custom_command(
    OUTPUT ${OPT_BC_FILE}
    COMMAND opt -load-pass-plugin=$<TARGET_FILE:IOOpt> -passes="io-opt" ${BC_FILE} -o ${OPT_BC_FILE}
    DEPENDS ${BC_FILE} IOOpt
)

add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${OPT_BC_FILE} -o ${OBJ_FILE}
    DEPENDS ${OPT_BC_FILE}
)

add_executable(loop_fast ${OBJ_FILE})
set_target_properties(loop_fast PROPERTIES LINKER_LANGUAGE CXX)


# MPI-IO Benchmarks (HPC Evaluation)
# MPI-IO Benchmark
if(MPI_CXX_FOUND)
    message(STATUS "MPI Found: Generating MPI benchmarks")

    # Baseline
    add_executable(mpi_write_at_standard mpi_write_at_benchmark.cpp)
    target_link_libraries(mpi_write_at_standard PRIVATE MPI::MPI_CXX)
    target_compile_options(mpi_write_at_standard PRIVATE -O3)

    # Optimised
    set(MPI_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mpi_write_at_benchmark.cpp)
    set(MPI_BC  ${CMAKE_CURRENT_BINARY_DIR}/mpi_write_at_benchmark.bc)
    set(MPI_OPT ${CMAKE_CURRENT_BINARY_DIR}/mpi_write_at_benchmark.opt.bc)
    set(MPI_OBJ ${CMAKE_CURRENT_BINARY_DIR}/mpi_write_at_benchmark.opt.o)

    # Extract MPI include directories and compile options so Clang can find <mpi.h>
    set(MPI_INC_FLAGS "")
    foreach(DIR IN LISTS MPI_CXX_INCLUDE_DIRS)
        list(APPEND MPI_INC_FLAGS "-I${DIR}")
    endforeach()

    # Compile C++ to LLVM IR (.bc)
    add_custom_command(
        OUTPUT ${MPI_BC}
        COMMAND ${CMAKE_CXX_COMPILER} -O3 -emit-llvm -c ${MPI_SRC} -o ${MPI_BC}
                -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include
                ${MPI_INC_FLAGS} ${MPI_CXX_COMPILE_OPTIONS}
        DEPENDS ${MPI_SRC}
        VERBATIM
    )

    # Run IOOpt via 'opt' (Injecting the shadow buffer environment variable here!)
    # We use CMake's built-in cross-platform '-E env' tool to set the variable.
    add_custom_command(
        OUTPUT ${MPI_OPT}
        COMMAND ${CMAKE_COMMAND} -E env "IO_SHADOW_BUFFER_MAX=65536"
                opt -load-pass-plugin=$<TARGET_FILE:IOOpt> 
                "-passes=cgscc(inline),function(instcombine,io-opt)" 
                ${MPI_BC} -o ${MPI_OPT}
        DEPENDS ${MPI_BC} IOOpt
        VERBATIM
    )

    # Compile optimised IR to an Object file (.o)
    add_custom_command(
        OUTPUT ${MPI_OBJ}
        COMMAND ${CMAKE_CXX_COMPILER} -O3 -c ${MPI_OPT} -o ${MPI_OBJ}
        DEPENDS ${MPI_OPT}
        VERBATIM
    )

    # Link into the final executable
    # target_link_libraries ensures all theMPI linker flags are applied properly.
    add_executable(mpi_write_at_fast ${MPI_OBJ})
    target_link_libraries(mpi_write_at_fast PRIVATE MPI::MPI_CXX)
    set_target_properties(mpi_write_at_fast PROPERTIES LINKER_LANGUAGE CXX)

else()
    message(WARNING "MPI CXX compiler not found. Skipping MPI benchmarks")
endif()

