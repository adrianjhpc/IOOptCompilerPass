set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

# Run TableGen to generate Dialect C++ code from IODialect.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/IODialect.td)

include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${CLANG_IR_SRC_DIR}/include
  ${CLANG_IR_BUILD_DIR}/tools/clang/include
  ${CLANG_IR_BUILD_DIR}/tools/clang/include/clang/CIR/Interfaces  # This is usually where the .inc lives
)

mlir_tablegen(IODialect.h.inc -gen-op-decls)
mlir_tablegen(IODialect.cpp.inc -gen-op-defs)
mlir_tablegen(IODialectDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IODialectDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(IODialectIncGen)

# Build the MLIR Passes library
add_library(MLIRIOOptimiser
  IODialect.cpp 
  IOLoopBatching.cpp
  IOLoweringToLLVM.cpp
  IORecognition.cpp)

# Ensure TableGen runs BEFORE trying to compile the C++ files
add_dependencies(MLIRIOOptimiser IODialectIncGen)

# Link against all required MLIR Dialects and Core Libraries
target_link_libraries(MLIRIOOptimiser PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRRewrite
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
)

# Tell the compiler exactly where to look for headers and generated .inc files
target_include_directories(MLIRIOOptimiser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}             # For IODialect.h in mlir-src/
    ${CMAKE_CURRENT_BINARY_DIR}             # For the generated .inc files
    ${CMAKE_BINARY_DIR}                     # Fallback for global generated files
)

add_executable(io-opt io-opt.cpp)

target_link_libraries(io-opt PRIVATE
  MLIRIOOptimiser
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRLLVMDialect
)
