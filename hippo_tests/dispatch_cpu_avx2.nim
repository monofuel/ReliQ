## CPU AVX2 Dispatch Test Entrypoint
## Tests CPU threading with AVX2 SIMD acceleration and SIMPLE hippo runtime
##
## Usage: nim r dispatch_cpu_avx2.nim

import ../src/device/dispatch

when isMainModule:
  echo "=== CPU AVX2 Dispatch Tests ==="
  echo "Configuration: CPU threading + AVX2 SIMD, SIMPLE runtime, vectorWidth=8"
  echo ""

  runDispatchTests()
