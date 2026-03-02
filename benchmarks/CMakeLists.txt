# Standard -O3 Baseline
add_executable(bench_standard pixel_write_benchmark.cpp)
target_compile_options(bench_standard PRIVATE -O3)

# Custom Pass -O3 (Optimized)
add_executable(bench_fast pixel_write_benchmark.cpp)

# Use the generator expression $<TARGET_FILE:IOOpt> to get the absolute path
# to your library automatically. This also creates a build dependency
target_compile_options(bench_fast PRIVATE 
    -O3 
    "-fpass-plugin=$<TARGET_FILE:IOOpt>"
)

# Ensure bench_fast only builds after the plugin is ready
add_dependencies(bench_fast IOOpt)
