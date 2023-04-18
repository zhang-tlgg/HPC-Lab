#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define EPS 1e-8

namespace ch = std::chrono;

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    //TODO
    memcpy(recvbuf, sendbuf, n * sizeof(float));
    int block_size = n / comm_sz;
    float* recv_data = new float[block_size];
    float* databuf = (float*)recvbuf;
    MPI_Request request[2];

    // 第一阶段 comm_sz - 1 步
    for (int k = 0; k < comm_sz - 1; k++)
    {
        int offset_send = ((my_rank - k + comm_sz) % comm_sz) * block_size;
        int offset_recv = ((my_rank - k - 1 + comm_sz) % comm_sz) * block_size;
        // 发送数据块
        MPI_Isend(databuf + offset_send, block_size, MPI_FLOAT, (my_rank + 1 + comm_sz) % comm_sz, k, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(recv_data, block_size, MPI_FLOAT, (my_rank - 1 + comm_sz) % comm_sz, k, MPI_COMM_WORLD, &request[1]);
		MPI_Wait(&request[1], nullptr);
        // 将收到的数据与自身相加
        for (int i = 0; i < block_size; i++)
        {
            databuf[offset_recv + i] += recv_data[i];
        }
        MPI_Wait(&request[0], nullptr);
    }

    // 第二阶段 comm_sz - 1 步
    for (int k = 0; k < comm_sz - 1; k++){
        int offset_send = ((my_rank + 1 - k + comm_sz) % comm_sz) * block_size;
        int offset_recv = ((my_rank - k + comm_sz) % comm_sz) * block_size;
        // 发送数据块
        MPI_Isend(databuf + offset_send, block_size, MPI_FLOAT, (my_rank + 1 + comm_sz) % comm_sz, k, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(databuf + offset_recv, block_size, MPI_FLOAT, (my_rank - 1 + comm_sz) % comm_sz, k, MPI_COMM_WORLD, &request[1]);
		MPI_Wait(&request[1], nullptr);
        MPI_Wait(&request[0], nullptr);
    }
}


// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            correct = false;
            break;
        }

    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
