/*
朴素实现
n = 1000,  Time: 1.794193 ms
n = 10000, Time: 1163.289596 ms
*/

#include "apsp.h"

#define max_dist 100001

__device__ int get_block_elem(int n, int *graph, int block_i, int block_j) {
	int i = block_i + threadIdx.y, j = block_j + threadIdx.x;
	return (i < n && j < n) ? graph[i * n + j] : max_dist;
}

__device__ void write_block_elem(int n, int *graph, int block_i, int block_j, int ele) {
	int i = block_i + threadIdx.y, j = block_j + threadIdx.x;
	if (i < n && j < n)
		graph[i * n + j] = ele;
}

__global__ void first_step(int n, int *graph, int r) {
	__shared__ int block[32][32];
	int block_i = r * 32, block_j = r * 32;
	int dist = max_dist;
	block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, block_i, block_j);
	__syncthreads();
	for (int k = 0; k < 32; k++)
		dist = min(dist, block[threadIdx.y][k] + block[k][threadIdx.x]);
	write_block_elem(n, graph, block_i, block_j, dist);
}

__global__ void second_step(int n, int *graph, int r) {
	__shared__ int centre_block[32][32], block[32][32];
	int centre_i = r * 32, centre_j = r * 32;
	int dist = max_dist;
	if(blockIdx.y == 0){
		// 水平
		int block_j = blockIdx.x < r ? blockIdx.x * 32 : (blockIdx.x + 1) * 32;
		centre_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, centre_j);
		block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, block_j);
		__syncthreads();
		for (int k = 0; k < 32; k++)
			dist = min(dist, centre_block[threadIdx.y][k] + block[k][threadIdx.x]);
		write_block_elem(n, graph, centre_i, block_j, dist);
	}
	else{
		// 竖直
		int block_i = blockIdx.x < r ? blockIdx.x * 32 : (blockIdx.x + 1) * 32;
		centre_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, centre_j);
		block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, block_i, centre_j);
		__syncthreads();
		for (int k = 0; k < 32; k++)
			dist = min(dist, block[threadIdx.y][k] + centre_block[k][threadIdx.x]);
		write_block_elem(n, graph, block_i, centre_j, dist);
	}
}

__global__ void third_step(int n, int *graph, int r) {
	__shared__ int v_block[32][32], h_block[32][32];
	int centre_i = r * 32, centre_j = r * 32;
	int block_i = blockIdx.y < r ? blockIdx.y * 32 : (blockIdx.y + 1) * 32;
	int block_j = blockIdx.x < r ? blockIdx.x * 32 : (blockIdx.x + 1) * 32;
	int dist = max_dist;
	h_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, block_j);
	v_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, block_i, centre_j);
	__syncthreads();
	dist = get_block_elem(n, graph, block_i, block_j);
	for (int k = 0; k < 32; k++)
		dist = min(dist, v_block[threadIdx.y][k] + h_block[k][threadIdx.x]);
	write_block_elem(n, graph, block_i, block_j, dist);
}

void apsp(int n, /* device */ int *graph) {
	int rounds = (n - 1) / 32 + 1;
	dim3 thr(32, 32);
	dim3 blk_2((n - 1) / 32, 2);
	dim3 blk_3((n - 1) / 32, (n - 1) / 32);
	for (int r = 0; r < rounds; r++)
	{
		first_step<<<1, thr>>>(n, graph, r);
		second_step<<<blk_2, thr>>>(n, graph, r);
		third_step<<<blk_3, thr>>>(n, graph, r);
	}
}
