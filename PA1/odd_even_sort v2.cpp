"""
将 wile 循环中的 wait 提到进程最后。
效果：数量 n 大时效果好， n 小时效果反而变差。
"""
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
  MPI_Request request_recv;
  MPI_Request *request_send = new MPI_Request[nprocs];

  // 进程内排序
  std::sort(data, data + block_len);

  // 奇偶排序
  int round = -1;
  int r = 0;
  while (++round < nprocs){
    int neighbor = ((rank % 2) == (round % 2)) ? (rank + 1) : (rank - 1);
    int neighbor_len;
    if (neighbor < 0 || (neighbor > rank && last_rank))
      continue;
    
    // 收发消息
    MPI_Isend(data, block_len, MPI_FLOAT, neighbor, round, MPI_COMM_WORLD, &request_send[r++]);
		MPI_Irecv(data_neighbor, block_size, MPI_FLOAT, neighbor, round, MPI_COMM_WORLD, &request_recv);
		MPI_Wait(&request_recv, &status);
    MPI_Get_count(&status, MPI_FLOAT, &neighbor_len);

    // 计算
    bool is_left = rank < neighbor;
    if ((is_left && data[block_len - 1] > data_neighbor[0]) || (!is_left && data_neighbor[neighbor_len - 1] > data[0])) {
      merge(data, block_len, data_neighbor, neighbor_len, data_merged, is_left);
      std::swap(data, data_merged);
    }
		// MPI_Wait(&request[0], nullptr);
  }

  MPI_Waitall(r, request_send, nullptr);
  delete[] request_send;
  delete[] data_neighbor;
	delete[] data_merged;
}
