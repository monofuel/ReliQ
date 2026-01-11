## GPU Dispatch Test Entrypoint
## Tests basic GPU with hippo runtime
## Uses actual GPU hardware via HIP or CUDA
##
## IMPORTANT: compile with cpp and appropriate GPU compiler
## Usage: nim cpp -r dispatch_hippo.nim
##
## TODO can test both 32 and 64 vectorWidth
## DO NOT TOUCH

import ../src/device/dispatch

when isMainModule:
  echo "=== GPU Dispatch Tests ==="
  echo "Configuration: GPU, hippo runtime, vectorWidth=32"
  echo ""

  runDispatchTests()
