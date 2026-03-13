import lit.formats
import os

config.name = "IOOptimisationPass Suite"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.c', '.cpp']

# Pass through environment variables
config.environment['PATH'] = os.environ.get('PATH', '')
if 'CPATH' in os.environ:
    config.environment['CPATH'] = os.environ.get('CPATH')

# --- Shared Library Substitutions ---
if hasattr(config, 'shlibdir'):
    config.substitutions.append(('%shlibdir', config.shlibdir))
if hasattr(config, 'shlibext'):
    config.substitutions.append(('%shlibext', config.shlibext))

# --- Tool Substitutions ---
# We use 'getattr' to provide a fallback name if CMake didn't find them
clang = getattr(config, 'clang_bin', 'clang')
clangpp = getattr(config, 'clangpp_bin', 'clang++')
opt = getattr(config, 'opt_bin', 'opt')
filecheck = getattr(config, 'filecheck_bin', 'FileCheck')
llvm_link = getattr(config, 'llvm_link_bin', 'llvm-link')

config.substitutions.append(('%clang', clang))
config.substitutions.append(('%ppclang', clangpp))
config.substitutions.append(('%opt', opt))
config.substitutions.append(('%FileCheck', filecheck))
config.substitutions.append(('%llvmlink', llvm_link))
