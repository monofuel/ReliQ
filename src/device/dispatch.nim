# monofuel note: working on migrating dispatch to hippo.
# I haven't used macros with hippo so this is very fun (tm)
# example launchers: ../../ReliQ/hippo_tests
# relying on compile time constants to set target platform and settings
# hippo should work across all

#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/device/dispatch.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
]#

# NB. hippo and macros are required but for some reason nim linter thinks it's unused.
import
  std/[macros],
  hippo,
  ./platforms

cpu:
  import simd/simdtypes
  
# TODO (monofuel) could we handle vectorWidth more automatically?
# cpu: 4/8/16 automatic depending on avx instruction and register size
# nvidia: warp, 32
# amd: wavefront, 32 or 64

# NB. Hippo's simple CPU backend automatically handles threads with OMP_NUM_THREADS or countProcessors().

# TODO (monofuel) max thread size of a block is like 65k I think? should have fancier logic for larger for loops.

macro each*(x: ForLoopStmt): untyped =
  ## Threaded + vectorized for loop construct that launches hippo kernels
  ##
  ## Used in regular code to automatically launch hippo kernels for parallel execution.
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in each 0..<10: <body>
  ## ```
  ## into automatic kernel creation and launching with vectorization.
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])

  if $rng != "..<":
    error("Only half-open ranges with '..<' are supported")

  # Generate unique kernel name to avoid conflicts
  let kernelName = genSym(nskProc, "eachKernel")

  result = quote do:
    # Create hippo kernel that uses the loop body
    proc `kernelName`(){.hippoGlobal.} =
      cpu:
        # Calculate flat thread index across all blocks and threads
        let threadIdxFlat = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x)

        # Each thread processes vectorWidth elements sequentially
        let startIdx = `lo` + threadIdxFlat * vectorWidth
        let endIdx = min(startIdx + vectorWidth, `hi`)

        # Process vectorWidth elements per thread
        # TODO (monofuel) SIMD work will need more thought
        var `idnt` = startIdx
        while `idnt` < endIdx:
          `body`
          inc `idnt`
      gpu:
        # TODO
        # can test with HIP_CPU
        # TODO (monofuel) GPU mode will only be operating over 1 element, needs more thought
        # should operate over $vectorWidth elements in a warp
        discard

    # Launch the kernel
    let totalWork = `hi` - `lo`
    let numChunks = (totalWork + vectorWidth - 1) div vectorWidth
    # Use different launch configurations for CPU vs GPU
    when defined(cpu):
      # CPU: one block with many threads (one thread per vectorWidth elements chunk)
      hippoLaunchKernel(
        `kernelName`,
        gridDim = newDim3(1, 1, 1),
        blockDim = newDim3(numChunks.uint32, 1, 1),
        args = hippoArgs()
      )
    else:
      # GPU: use 256 threads per block for better performance
      # TODO (monofuel) should be operating over warps of $vectorWidth elements
      const blockSize = 256'u32
      let gridSize = ((numChunks + int(blockSize) - 1) div int(blockSize)).uint32
      hippoLaunchKernel(
        `kernelName`,
        gridDim = newDim3(gridSize, 1, 1),
        blockDim = newDim3(blockSize, 1, 1),
        args = hippoArgs()
      )

macro all*(x: ForLoopStmt): untyped =
  ## Threaded for loop construct that launches hippo kernels
  ##
  ## Used in regular code to automatically launch hippo kernels for parallel execution.
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in all 0..<10: <body>
  ## ```
  ## into automatic kernel creation and launching (one thread per element).
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])

  if $rng != "..<":
    error("Only half-open ranges with '..<' are supported")

  # monofuel note: this is very clever, I should implement this for helper macros in hippo
  # Generate unique kernel name to avoid conflicts
  let kernelName = genSym(nskProc, "allKernel")

  result = quote do:
    # Create hippo kernel that uses the loop body
    proc `kernelName`(){.hippoGlobal.} =
      # Calculate flat thread index across all blocks and threads
      let idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x)
      let elementIdx = `lo` + idx
      if elementIdx < `hi`:
        let `idnt` = elementIdx
        `body`

    # Launch the kernel (one thread per element)
    let totalWork = `hi` - `lo`
    # Use different launch configurations for CPU vs GPU
    when defined(cpu):
      # CPU: one block with many threads (one thread per element)
      hippoLaunchKernel(
        `kernelName`,
        gridDim = newDim3(1, 1, 1),
        blockDim = newDim3(totalWork.uint32, 1, 1),
        args = hippoArgs()
      )
    else:
      # GPU: use 256 threads per block for better performance
      const blockSize = 256'u32
      let gridSize = ((totalWork + int(blockSize) - 1) div int(blockSize)).uint32
      hippoLaunchKernel(
        `kernelName`,
        gridDim = newDim3(gridSize, 1, 1),
        blockDim = newDim3(blockSize, 1, 1),
        args = hippoArgs()
      )

# TODO (monofuel) how to handle memory easily with macros?
#                 this will be very important, but possibly also tricky.
# Global device memory for testing (allocated once)
var testAllResults = hippoMalloc(sizeof(int) * 80)
var testEachResults = hippoMalloc(sizeof(int) * 80)
const TestSize = 80

proc runDispatchTests*(testSize = 80) =
  ## Run comprehensive tests for the dispatch macros by launching hippo kernels
  ## Can be called from external test entrypoints

  # Use the module-level constant for now
  let actualTestSize = TestSize

  block:
    echo "Testing 'all' macro via hippo kernel..."

    # Use macro directly - it will create and launch its own kernel
    for i in all 0..<actualTestSize:
      let arr = cast[ptr UncheckedArray[int]](testAllResults.p)
      arr[i] = i * 2

    # Copy back results
    var hostAllResults: array[TestSize, int]
    hippoMemcpy(addr hostAllResults[0], testAllResults, sizeof(int) * actualTestSize, HippoMemcpyDeviceToHost)

    # Synchronize to catch any gpu errors before verifying results
    hippoSynchronize()

    # Verify results
    var allPassed = true
    var allSum = 0
    var allExpectedSum = 0
    for i in 0..<actualTestSize:
      allSum += hostAllResults[i]
      allExpectedSum += i * 2
      if hostAllResults[i] != i * 2:
        echo "FAIL: allResults[", i, "] = ", hostAllResults[i], ", expected ", i * 2
        allPassed = false
        break

    echo "  Sum of results: ", allSum
    echo "  Expected sum: ", allExpectedSum
    echo "  Sample values: allResults[0]=", hostAllResults[0], ", allResults[10]=", hostAllResults[10], ", allResults[", actualTestSize-1, "]=", hostAllResults[actualTestSize-1]

    if allPassed and allSum == allExpectedSum:
      echo "✓ 'all' macro test PASSED: All ", actualTestSize, " elements processed correctly"
    else:
      echo "✗ 'all' macro test FAILED"

  block:
    echo ""
    echo "Testing 'each' macro via hippo kernel..."

    # Use macro directly - it will create and launch its own kernel
    for i in each 0..<actualTestSize:
      let arr = cast[ptr UncheckedArray[int]](testEachResults.p)
      arr[i] = i * 3

    # Copy back results
    var hostEachResults: array[TestSize, int]
    hippoMemcpy(addr hostEachResults[0], testEachResults, sizeof(int) * actualTestSize, HippoMemcpyDeviceToHost)

    # Synchronize to catch any gpu errors before verifying results
    hippoSynchronize()

    # Verify results
    var eachPassed = true
    var processedCount = 0
    var eachSum = 0
    var eachExpectedSum = 0
    for i in 0..<actualTestSize:
      eachSum += hostEachResults[i]
      eachExpectedSum += i * 3
      if hostEachResults[i] == i * 3:
        inc processedCount

    echo "  Processed ", processedCount, " out of ", actualTestSize, " elements"
    echo "  Sum of results: ", eachSum
    echo "  Expected sum: ", eachExpectedSum
    echo "  Sample values: eachResults[0]=", hostEachResults[0], ", eachResults[10]=", hostEachResults[10], ", eachResults[", actualTestSize-1, "]=", hostEachResults[actualTestSize-1]
    echo "  (Note: 'each' processes in vectorWidth=", vectorWidth, " chunks)"

    if eachPassed and processedCount == actualTestSize and eachSum == eachExpectedSum:
      echo "✓ 'each' macro test PASSED: Vectorized processing working correctly"
    else:
      echo "✗ 'each' macro test FAILED"
      if processedCount != actualTestSize:
        echo "  Error: Only ", processedCount, " elements processed, expected ", actualTestSize
      if eachSum != eachExpectedSum:
        echo "  Error: Sum mismatch - got ", eachSum, ", expected ", eachExpectedSum

  # TODO (monofuel) we should check for errors with hippoCheckLastError()

  echo ""
  echo "Summary:"
  echo "  vectorWidth: ", vectorWidth
  echo "  testSize: ", actualTestSize

when isMainModule:
  runDispatchTests()
