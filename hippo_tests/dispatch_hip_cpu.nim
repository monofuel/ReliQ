## AMD HIP_CPU Dispatch Test Entrypoint
## Tests basic AMD with HIP_CPU hippo runtime
## test the `hip` backend with HIP_CPU runtime for easier testing before fully migrating to GPU.
##
## IMPORTANT: compile with cpp
## Usage: nim cpp -r dispatch_hip_cpu.nim
##
## DO NOT TOUCH

import ../src/device/dispatch

when isMainModule:
  echo "=== CPU Dispatch Tests ==="
  echo "Configuration: AMD, HIP_CPU runtime, vectorWidth=32"
  echo ""

  runDispatchTests()
