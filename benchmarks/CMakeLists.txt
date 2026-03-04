# Standard -O3 Baseline
add_executable(pixel_write_standard pixel_write_benchmark.cpp)
target_compile_options(pixel_write_standard PRIVATE -O3)

# Custom Pass -O3 (Optimized)
add_executable(pixel_write_fast pixel_write_benchmark.cpp)

# Use the generator expression $<TARGET_FILE:IOOpt> to get the absolute path
# to your library automatically. This also creates a build dependency
target_compile_options(pixel_write_fast PRIVATE 
    -O3 
    "-fpass-plugin=$<TARGET_FILE:IOOpt>"
)

# Ensure bench_fast only builds after the plugin is ready
add_dependencies(pixel_write_fast IOOpt)

#Standard -O3 Baseline
add_executable(http_standard http_benchmark.cpp)
target_compile_options(http_standard PRIVATE -O3 -fno-inline)

# Custom Pass -O3 (Optimized)
add_executable(http_fast http_benchmark.cpp)
# Load your custom IOOpt pass
target_compile_options(http_fast PRIVATE 
    -O3 
    -fno-inline 
    -fpass-plugin=$<TARGET_FILE:IOOpt>
)
# Ensure the pass is built before compiling this benchmark
add_dependencies(http_fast IOOpt)
