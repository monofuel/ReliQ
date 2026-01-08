## CPU Dispatch Test Entrypoint
## Tests basic CPU threading with SIMPLE hippo runtime
##
## Usage: nim r dispatch_cpu.nim
##
## DO NOT TOUCH

import ../src/device/dispatch

when isMainModule:
  echo "=== CPU Dispatch Tests ==="
  echo "Configuration: CPU threading, SIMPLE runtime, vectorWidth=8"
  echo ""

  runDispatchTests()
