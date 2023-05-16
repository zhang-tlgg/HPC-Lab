/*
两级分块
n = 1000,  Time: 1.716608 ms
n = 10000, Time: 660.978576 ms
*/

#include "apsp.h"

#define max_dist 100001
#define batch_size_1 6
#define batch_size_2 6

__device__ int get_block_elem(int n, int *graph, int block_i, int block_j) {
	int i = block_i + threadIdx.y, j = block_j + threadIdx.x;
	return (i < n && j < n) ? graph[i * n + j] : max_dist;
}

__device__ int get_block_elem_in(int n, int *graph, int block_i, int block_j) {
	int i = block_i + threadIdx.y, j = block_j + threadIdx.x;
	return graph[i * n + j];
}

__device__ void write_block_elem(int n, int *graph, int block_i, int block_j, int ele) {
	int i = block_i + threadIdx.y, j = block_j + threadIdx.x;
	if (i < n && j < n)
		graph[i * n + j] = ele;
}

__device__ void write_block_elem_in(int n, int *graph, int block_i, int block_j, int ele) {
	int i = block_i + threadIdx.y, j = block_j + threadIdx.x;
	graph[i * n + j] = ele;
}

__global__ void first_step(int n, int *graph, int r) {
	__shared__ int centre_block[32][32];
	int centre_i = r * 32, centre_j = r * 32;
	int dist = max_dist;
	centre_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, centre_j);
	__syncthreads();
	for (int k = 0; k < 32; k++)
		dist = min(dist, centre_block[threadIdx.y][k] + centre_block[k][threadIdx.x]);
	write_block_elem(n, graph, centre_i, centre_j, dist);
}

__global__ void second_step(int n, int *graph, int r) {
	__shared__ int centre_block[32][32], block[batch_size_1][32][32];
	int centre_i = r * 32, centre_j = r * 32;
	int dist = max_dist;
	if(blockIdx.y == 0){
		// 水平
		int block_j = blockIdx.x * batch_size_1 * 32;
		centre_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, centre_j);
		for (int p = 0, j = block_j; p < batch_size_1; p++, j += 32)
			block[p][threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, j);
		__syncthreads();
		for (int p = 0, j = block_j; p < batch_size_1; p++, j += 32){
			dist = max_dist;
			for (int k = 0; k < 32; k++)
				dist = min(dist, centre_block[threadIdx.y][k] + block[p][k][threadIdx.x]);
			write_block_elem(n, graph, centre_i, j, dist);
		}
	}
	else{
		// 竖直
		int block_i = blockIdx.x * batch_size_1 * 32;
		centre_block[threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, centre_j);
		for (int p = 0, i = block_i; p < batch_size_1; p++, i += 32)
			block[p][threadIdx.y][threadIdx.x] = get_block_elem(n, graph, i, centre_j);
		__syncthreads();
		for (int p = 0, i = block_i; p < batch_size_1; p++, i += 32) {
			dist = max_dist;
			for (int k = 0; k < 32; k++)
				dist = min(dist, block[p][threadIdx.y][k] + centre_block[k][threadIdx.x]);
			write_block_elem(n, graph, i, centre_j, dist);
		}
	}
}

__global__ void third_step(int n, int *graph, int r) {
	__shared__ int v_block[batch_size_2][32][32], h_block[batch_size_2][32][32];
	int centre_i = r * 32, centre_j = r * 32;
	int block_i = blockIdx.y * batch_size_2 * 32;
	int block_j = blockIdx.x * batch_size_2 * 32;
	int dist = max_dist;
	for (int k = 0, j = block_j; k < batch_size_2; k++, j += 32)
		h_block[k][threadIdx.y][threadIdx.x] = get_block_elem(n, graph, centre_i, j);
	for (int k = 0, i = block_i; k < batch_size_2; k++, i += 32)
		v_block[k][threadIdx.y][threadIdx.x] = get_block_elem(n, graph, i, centre_j);
	__syncthreads();
	if (block_i + batch_size_2 * 32 <= n && block_j + batch_size_2 * 32 <= n) {
		for (int p = 0, i = block_i; p < batch_size_2; p++, i += 32) {
			for (int q = 0, j = block_j; q < batch_size_2; q++, j += 32) {
				dist = get_block_elem_in(n, graph, i, j);
				for (int k = 0; k < 32; k++)
					dist = min(dist, v_block[p][threadIdx.y][k] + h_block[q][k][threadIdx.x]);
				write_block_elem_in(n, graph, i, j, dist);
			}
		}
	}
	else {
		for (int p = 0, i = block_i; p < batch_size_2; p++, i += 32) {
			for (int q = 0, j = block_j; q < batch_size_2; q++, j += 32) {
				dist = get_block_elem(n, graph, i, j);
				for (int k = 0; k < 32; k++)
					dist = min(dist, v_block[p][threadIdx.y][k] + h_block[q][k][threadIdx.x]);
				write_block_elem(n, graph, i, j, dist);
			}
		}
	}
}

void apsp(int n, /* device */ int *graph) {
	int rounds = (n - 1) / 32 + 1;
	dim3 thr(32, 32);
	dim3 blk_2((n - 1) / (batch_size_1 * 32) + 1, 2);
	int blk_3_x = (n - 1) / (batch_size_2 * 32) + 1;
	dim3 blk_3(blk_3_x, blk_3_x);
	for (int r = 0; r < rounds; r++)
	{
		first_step<<<1, thr>>>(n, graph, r);
		second_step<<<blk_2, thr>>>(n, graph, r);
		third_step<<<blk_3, thr>>>(n, graph, r);
	}
}
