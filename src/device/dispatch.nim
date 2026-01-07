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

import std/[macros, os, strutils, cpuinfo, typedthreads]

import platforms

nvidia: import cuda/[cudawrap]
amd: import hip/[hipwrap]
cpu: 
  import simd/[simdtypes, x86wrap]
  when defined(avx2):
    import simd/avx2wrap
  when defined(sse):
    import simd/ssewrap

var numThreads*: int = 1
let envThreads = getEnv("OMP_NUM_THREADS")
if envThreads.len > 0:
  try: numThreads = parseInt(envThreads)
  except ValueError: numThreads = countProcessors()

macro each*(x: ForLoopStmt): untyped =
  ## Threaded + vectorized for loop consturct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in every 0..10: <body>
  ## ```
  ## into a threaded + vectorized loop on CPU/GPU. Behaves like `Grid`'s
  ## `accelerator_for` construct. 
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  
  if $rng != "..<":
    error("Only half-open ranges with '..<' are supported in 'all' loops")
  
  result = quote do:
    nvidia: discard
    amd: discard
    cpu:
      let totalWork = `hi` - `lo`
      let baseChunkSize = (totalWork + numThreads - 1) div numThreads
      let chunkSize = ((baseChunkSize + vectorWidth - 1) div vectorWidth) * vectorWidth

      proc workerAll(threadId: int) {.thread.} =
        let startIdx = `lo` + threadId * chunkSize
        let endIdx = min(`lo` + (threadId + 1) * chunkSize, `hi`)
        
        # Process vectorWidth-aligned chunks using SIMD for index generation
        var `idnt` = startIdx
        let vectorEnd = startIdx + ((endIdx - startIdx) div vectorWidth) * vectorWidth
        
        # Vectorized section: use SIMD to generate indices efficiently
        when vectorWidth > 1 and defined(avx2):
          # Generate base index vector [0, 1, 2, 3, 4, 5, 6, 7] using AVX2
          when vectorWidth == 8:
            let baseIndices = mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
            let stepVector = mm256_set1_epi32(vectorWidth.cint)
            var currentBase = mm256_set1_epi32(`idnt`.cint)
            
            while `idnt` < vectorEnd:
              # Generate indices for this chunk: [idnt, idnt+1, ..., idnt+7]
              let indices = mm256_add_epi32(currentBase, baseIndices)
              
              # Extract indices and process each element
              var indexArray: array[8, int32]
              mm256_storeu_si256(cast[ptr m256i](addr indexArray[0]), indices)
              
              for lane in 0..<vectorWidth:
                block:
                  var `idnt` = indexArray[lane].int
                  `body`
              
              # Advance to next chunk
              currentBase = mm256_add_epi32(currentBase, stepVector)
              `idnt` += vectorWidth
          elif vectorWidth == 4 and defined(sse):
            # Use SSE for 4-wide vectors
            let baseIndices = mm_setr_epi32(0, 1, 2, 3)
            let stepVector = mm_set1_epi32(vectorWidth.cint)
            var currentBase = mm_set1_epi32(`idnt`.cint)
            
            while `idnt` < vectorEnd:
              let indices = mm_add_epi32(currentBase, baseIndices)
              
              var indexArray: array[4, int32]
              mm_storeu_si128(cast[ptr m128i](addr indexArray[0]), indices)
              
              for lane in 0..<vectorWidth:
                block:
                  var `idnt` = indexArray[lane].int
                  `body`
              
              currentBase = mm_add_epi32(currentBase, stepVector)
              `idnt` += vectorWidth
          else:
            # Fallback: sequential processing
            while `idnt` < vectorEnd:
              for lane in 0..<vectorWidth:
                block:
                  var `idnt` = `idnt` + lane
                  `body`
              `idnt` += vectorWidth
        else:
          # No SIMD: sequential processing
          while `idnt` < vectorEnd:
            for lane in 0..<vectorWidth:
              block:
                var `idnt` = `idnt` + lane
                `body`
            `idnt` += vectorWidth
        
        # Remainder: process remaining elements one by one
        while `idnt` < endIdx:
          `body`
          inc `idnt`
      
      # Use standard Nim threads instead of malebolgia
      var threads: array[32, Thread[int]]  # Support up to 32 threads
      for threadId in 0..<numThreads:
        createThread(threads[threadId], workerAll, threadId)
      for threadId in 0..<numThreads:
        joinThread(threads[threadId])

macro all*(x: ForLoopStmt): untyped =
  ## Threaded for loop construct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in every 0..10: <body>
  ## ```
  ## into a simple loop that iterates over the specified range.
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  
  if $rng != "..<":
    error("Only half-open ranges with '..<' are supported in 'every' loops")
  
  result = quote do:
    let totalWork = `hi` - `lo`
    let baseChunkSize = (totalWork + numThreads - 1) div numThreads

    proc workerEvery(threadId: int) {.thread.} =
      let startIdx = `lo` + threadId * baseChunkSize
      let endIdx = min(`lo` + (threadId + 1) * baseChunkSize, `hi`)
      
      for `idnt` in startIdx..<endIdx:
        `body`
      
    # Use standard Nim threads instead of malebolgia
    var threads: array[32, Thread[int]]  # Support up to 32 threads
    for threadId in 0..<numThreads:
      createThread(threads[threadId], workerEvery, threadId)
    for threadId in 0..<numThreads:
      joinThread(threads[threadId])

when isMainModule:
  const testSize = 80
  
  block:
    echo "Testing 'all' macro (threaded, non-vectorized)..."
    var allResults = cast[ptr UncheckedArray[int]](allocShared0(testSize * sizeof(int)))
    
    for n in all 0..<testSize:
      allResults[n] = n * 2
    
    # Verify all elements were processed
    var allPassed = true
    var allSum = 0
    var allExpectedSum = 0
    for i in 0..<testSize:
      allSum += allResults[i]
      allExpectedSum += i * 2
      if allResults[i] != i * 2:
        echo "FAIL: allResults[", i, "] = ", allResults[i], ", expected ", i * 2
        allPassed = false
        break
    
    echo "  Sum of results: ", allSum
    echo "  Expected sum: ", allExpectedSum
    echo "  Sample values: allResults[0]=", allResults[0], ", allResults[10]=", allResults[10], ", allResults[", testSize-1, "]=", allResults[testSize-1]
    
    if allPassed and allSum == allExpectedSum:
      echo "✓ 'all' macro test PASSED: All ", testSize, " elements processed correctly"
    else:
      echo "✗ 'all' macro test FAILED"
    
    deallocShared(allResults)
  
  block:
    echo ""
    echo "Testing 'each' macro (threaded + vectorized)..."
    var eachResults = cast[ptr UncheckedArray[int]](allocShared0(testSize * sizeof(int)))
    
    for n in each 0..<testSize:
      eachResults[n] = n * 3
    
    # Verify elements were processed (note: each processes in vectorWidth chunks)
    var eachPassed = true
    var processedCount = 0
    var eachSum = 0
    var eachExpectedSum = 0
    for i in 0..<testSize:
      eachSum += eachResults[i]
      eachExpectedSum += i * 3
      if eachResults[i] == i * 3:
        inc processedCount
    
    echo "  Processed ", processedCount, " out of ", testSize, " elements"
    echo "  Sum of results: ", eachSum
    echo "  Expected sum: ", eachExpectedSum
    echo "  Sample values: eachResults[0]=", eachResults[0], ", eachResults[10]=", eachResults[10], ", eachResults[", testSize-1, "]=", eachResults[testSize-1]
    echo "  (Note: 'each' processes in vectorWidth=", vectorWidth, " chunks)"
    
    if eachPassed and processedCount == testSize and eachSum == eachExpectedSum:
      echo "✓ 'each' macro test PASSED: Vectorized processing working correctly"
    else:
      echo "✗ 'each' macro test FAILED"
      if processedCount != testSize:
        echo "  Error: Only ", processedCount, " elements processed, expected ", testSize
      if eachSum != eachExpectedSum:
        echo "  Error: Sum mismatch - got ", eachSum, ", expected ", eachExpectedSum
    
    deallocShared(eachResults)
  
  echo ""
  echo "Summary:"
  echo "  numThreads: ", numThreads
  echo "  vectorWidth: ", vectorWidth
  echo "  testSize: ", testSize
