#include "IODialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::io;

// Include the generated Dialect definitions
#include "IODialectDialect.cpp.inc"

// Initialise the dialect (registering our custom ops)
void IODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IODialect.cpp.inc"
      >();
}

// Include the generated Operation definitions
#define GET_OP_CLASSES
#include "IODialect.cpp.inc"
