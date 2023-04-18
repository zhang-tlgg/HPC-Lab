# 奇偶排序 实验报告

张天乐 2018011038

## sort 函数源代码



```cpp
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void merge(float *mine, int n_mine, float *neighbor, int n_neighbor, float *result, bool is_left) {
  if(is_left){
    int pp = 0, qq = 0;
    for (int i = 0; i < n_mine; i++){
      if (qq == n_neighbor) 
        result[i] = mine[pp++];
      else if (pp == n_mine)
        result[i] = neighbor[qq++];
      else
        result[i] = mine[pp] < neighbor[qq] ? mine[pp++] : neighbor[qq++];
    }
  }
  else{
    int pp = n_mine -1, qq = n_neighbor - 1;
    for (int i = n_mine - 1; i >= 0; i--){
      if (qq == -1) 
        result[i] = mine[pp--];
      else if (pp == -1)
        result[i] = neighbor[qq--];
      else
        result[i] = mine[pp] > neighbor[qq] ? mine[pp--] : neighbor[qq--];
    }
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  if (out_of_range)
    return;

  size_t block_size = ceiling(n, nprocs);
  float *data_neighbor = new float[block_size];
  float *data_merged = new float[block_len];
  MPI_Status status;
  MPI_Request request[2];

  // 进程内排序
  std::sort(data, data + block_len);

  // 奇偶排序
  int round = -1;
  while (++round < nprocs){
    int neighbor = ((rank % 2) == (round % 2)) ? (rank + 1) : (rank - 1);
    int neighbor_len;
    if (neighbor < 0 || (neighbor > rank && last_rank))
      continue;
    
    // 收发消息
    MPI_Isend(data, block_len, MPI_FLOAT, neighbor, round, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(data_neighbor, block_size, MPI_FLOAT, neighbor, round, MPI_COMM_WORLD, &request[1]);
    MPI_Wait(&request[1], &status);
    MPI_Get_count(&status, MPI_FLOAT, &neighbor_len);

    // 计算
    bool is_left = rank < neighbor;
    if ((is_left && data[block_len - 1] > data_neighbor[0]) || (!is_left && data_neighbor[neighbor_len - 1] > data[0])) {
      merge(data, block_len, data_neighbor, neighbor_len, data_merged, is_left);
      std::swap(data, data_merged);
    }
    MPI_Wait(&request[0], nullptr);
  }

  delete[] data_neighbor;
  delete[] data_merged;
}

```

## 优化方式

- 使用非阻塞通信，将计算时间和通信时间尽可能地重叠。

- 相邻进程归并后，每个进程承担一般的计算任务，只计算出分配到自己的元素。

- 减少内存的开销与移动。

## 性能测试

| 进程数  | 运行时间（毫秒）     | 加速比         |
| ---- | ------------ | ----------- |
| 1×1  | 12489.233000 |             |
| 1×2  | 6618.923000  | 1.88689806  |
| 1×4  | 3543.656000  | 3.52439204  |
| 1×8  | 1984.241000  | 6.294211741 |
| 1×16 | 1211.695000  | 10.30724151 |
| 2×16 | 788.153000   | 15.84620372 |
