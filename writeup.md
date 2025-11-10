# Writeup

## Part 1

### Step 1

1. Execution time in micro seconds of `python run_benchmark.py --kernel tiled -n 25600` was 37318 μs.

(aws_neuronx_venv_pytorch_2_8) ubuntu@ip-172-31-33-95:~/asst4-trainium2/part1$ python run_benchmark.py --kernel tiled -n 25600

Running vector_add_tiled with shape (25600,)

Correctness passed? True

Benchmarking performance.........

file                                                                            
+---+----+---------+---------+---------+-------+--------+-------+-------+-------+--------+---------+---------+-------+
  B   NC   NC USED   WEIGHTS   MODE      INF/S   IRES/S   L(1)    L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  
  1   1    1         dynamic   LIBMODE   26.71   26.71    37342   37388   37394   37298    37308     37318     N/A    
+---+----+---------+---------+---------+-------+--------+-------+-------+-------+--------+---------+---------+-------+


## Part 2