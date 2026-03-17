set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

# Run TableGen to generate Dialect C++ code from IODialect.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/IODialect.td)

include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${CLANG_IR_SRC_DIR}/include
  ${CLANG_IR_BUILD_DIR}/tools/clang/include
  ${CLANG_IR_BUILD_DIR}/tools/clang/include/clang/CIR/Interfaces  
)


find_library(MLIRCIR_LIB 
    NAMES MLIRCIR 
    PATHS ${LLVM_LIBRARY_DIR} 
    NO_DEFAULT_PATH
)

find_library(MLIRCIR_INTERFACES_LIB 
    NAMES MLIRCIRInterfaces 
    PATHS ${LLVM_LIBRARY_DIR} 
    NO_DEFAULT_PATH
)

find_library(MLIRCIR_TRANSFORMS_LIB
    NAMES MLIRCIRTransforms
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

find_library(TARGET_LOWERING_LIB
    NAMES TargetLowering
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)


if(NOT MLIRCIR_LIB OR NOT MLIRCIR_INTERFACES_LIB)
    message(FATAL_ERROR "Could not find ClangIR libraries in ${LLVM_LIBRARY_DIR}!")
endif()

find_library(CLANGAST_LIB 
    NAMES clangAST 
    PATHS ${LLVM_LIBRARY_DIR} 
    NO_DEFAULT_PATH
)

find_library(CLANGLEX_LIB 
    NAMES clangLex 
    PATHS ${LLVM_LIBRARY_DIR} 
    NO_DEFAULT_PATH
)

find_library(CLANGBASIC_LIB 
    NAMES clangBasic 
    PATHS ${LLVM_LIBRARY_DIR} 
    NO_DEFAULT_PATH
)


find_library(CLANGCIRLOWERINGDIRECTTOLLVM_LIB
    NAMES clangCIRLoweringDirectToLLVM
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

find_library(CLANGCIRLOWERINHELPERS_LIB
    NAMES clangCIRLoweringHelpers
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

find_library(CLANGCIR_LIB
    NAMES clangCIR
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

find_library(CLANGCPP_LIB
    NAMES clang-cpp
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

if(NOT CLANGAST_LIB OR NOT CLANGLEX_LIB OR NOT CLANGBASIC_LIB)
    message(FATAL_ERROR "Could not find Clang AST/Lex/Basic libraries in ${LLVM_LIBRARY_DIR}!")
endif()

llvm_map_components_to_libnames(LLVM_CORE_DEPS
    FrontendOpenMP
    FrontendHLSL
    Analysis
    TransformUtils
    Target
    Core
    Support
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
  IORecognition.cpp
  TargetUtils.cpp)

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
  MLIRRegisterAllPasses
  ${MLIRCIR_LIB}
  ${MLIRCIR_INTERFACES_LIB}
  ${CLANGAST_LIB}
  ${CLANGLEX_LIB}
  ${CLANGBASIC_LIB}
  ${LLVM_CORE_DEPS}
)

# Tell the compiler where to look for headers and generated .inc files
target_include_directories(MLIRIOOptimiser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}             # For IODialect.h in mlir-src/
    ${CMAKE_CURRENT_BINARY_DIR}             # For the generated .inc files
    ${CMAKE_BINARY_DIR}                     # Fallback for global generated files
)

add_executable(io-opt io-opt.cpp)

target_link_libraries(io-opt PRIVATE
    MLIRIOOptimiser
    ${CLANGCIRLOWERINGDIRECTTOLLVM_LIB}
    ${CLANGCIRLOWERINHELPERS_LIB}
    ${TARGET_LOWERING_LIB}
    ${MLIRCIR_TRANSFORMS_LIB}
    ${MLIRCIR_LIB}
    ${CLANGCIR_LIB}
    ${MLIRCIR_INTERFACES_LIB}
    ${MLIRCIR_TRANSFORMS_LIB}
    ${CLANGAST_LIB}
    ${CLANGLEX_LIB}
    ${CLANGBASIC_LIB}
    ${CLANGCPP_LIB}
    MLIROptLib
    MLIRRegisterAllPasses
    MLIRSCFDialect
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRTransforms
    MLIRPass
    MLIRParser
    MLIRIR
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIROpenMPToLLVMIRTranslation
    ${LLVM_CORE_DEPS}
)
