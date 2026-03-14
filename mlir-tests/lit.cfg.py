import os
import lit.formats

config.name = 'IOOpt-MLIR'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']

# Get the directory where this lit.cfg.py file lives
config.test_source_root = os.path.dirname(__file__)

# Safely get the tools directories. If they weren't injected by CMake,
# we manually point to your build directory and system paths.
fallback_io_tools = os.path.abspath(os.path.join(config.test_source_root, '..', 'build', 'mlir-src'))

io_tools_dir = getattr(config, 'io_tools_dir', fallback_io_tools)
llvm_tools_dir = getattr(config, 'llvm_tools_dir', '')

# Build the execution PATH
path_list = [io_tools_dir]
if llvm_tools_dir:
    path_list.append(llvm_tools_dir)
path_list.append(config.environment.get('PATH', ''))

config.environment['PATH'] = os.path.pathsep.join(path_list)
