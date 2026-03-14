// include/IODialect.h
#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

#include "IODialectDialect.h.inc"

#define GET_OP_CLASSES
#include "IODialect.h.inc"
