# Writeup

## Part 1

### Step 1

1. Execution time in micro seconds of `python run_benchmark.py --kernel tiled -n 25600` was 37318 μs.

```
(aws_neuronx_venv_pytorch_2_8) ubuntu@ip-172-31-33-95:~/asst4-trainium2/part1$ python run_benchmark.py --kernel tiled -n 25600

Running vector_add_tiled with shape (25600,)

Correctness passed? True

Benchmarking performance.........

file                                                                            
+---+----+---------+---------+---------+-------+--------+-------+-------+-------+--------+---------+---------+-------+
  B   NC   NC USED   WEIGHTS   MODE      INF/S   IRES/S   L(1)    L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  
  1   1    1         dynamic   LIBMODE   26.71   26.71    37342   37388   37394   37298    37308     37318     N/A    
+---+----+---------+---------+---------+-------+--------+-------+-------+-------+--------+---------+---------+-------+
```

2. Now with `ROW_CHUNK=128`, the execution time is 378 μs. Thus, the code is close to 100x faster.

The code is written as a loop, where `ROW_CHUNK` entries are loaded and processes in parallel. Since the NC has capacity for 128 partitions, it makes sense that if we use all 128 partitions we get around a 100x speed up.

```
(aws_neuronx_venv_pytorch_2_8) ubuntu@ip-172-31-33-95:~/asst4-trainium2/part1$ python run_benchmark.py --kernel tiled -n 25600

Running vector_add_tiled with shape (25600,)

Correctness passed? True

Benchmarking performance.........

file                                                                            
+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+
  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  
  1   1    1         dynamic   LIBMODE   1992.91   1992.91   435    439     457     378      378       378       N/A    
+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+
```

3. 

With `ROW_CHUNK=256`, there is an error because the partition dimension aka tensor shape 0 can be max 128, but in the code it's set to be `ROW_CHUNK`, leading to error if `ROW_CHUNK > 128`.

```
(aws_neuronx_venv_pytorch_2_8) ubuntu@ip-172-31-33-95:~/asst4-trainium2/part1$ python run_benchmark.py --kernel tiled -n 25600

Running vector_add_tiled with shape (25600,)
Traceback (most recent call last):
  File "/home/ubuntu/asst4-trainium2/part1/run_benchmark.py", line 106, in <module>
    main()
  File "/home/ubuntu/asst4-trainium2/part1/run_benchmark.py", line 102, in main
    benchmark_kernel(kernel, *kernel_args, profile_name=args.profile_name)
  File "/home/ubuntu/asst4-trainium2/part1/run_benchmark.py", line 51, in benchmark_kernel
    out = nki.baremetal(kernel)(*args)
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 235, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.Kernel.__call__
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 236, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.Kernel.__call__
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 322, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.TraceKernel.call_impl
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 337, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.TraceKernel.specialize_and_call
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 339, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.TraceKernel.specialize_and_call
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 347, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.TraceKernel.expand_kernel_with_ctx
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 367, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.TraceKernel.expand_kernel_with_ctx
  File "neuronxcc/nki/compiler/backends/neuron/TraceKernel.py", line 352, in neuronxcc.nki.compiler.backends.neuron.TraceKernel.TraceKernel.expand_kernel_with_ctx
  File "/home/ubuntu/asst4-trainium2/part1/kernels.py", line 76, in vector_add_tiled
    a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
ValueError: number of partitions 256 exceed architecture limitation of 128. Info on how to fix: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.errors.html#err-num-partition-exceed-arch-limit
```

### Step 2a

1. 186 μs. This is around 2x faster.

```
Running vector_add_stream with shape (25600,)

Correctness passed? True

Benchmarking performance.........

file                                                                            
+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+
  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  
  1   1    1         dynamic   LIBMODE   2957.50   2957.50   270    272     297     186      186       186       N/A    
+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+
```

2. TODO: reduce DMA transfers

### Step 2b

## Part 2