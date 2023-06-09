# MPI 异步通信小作业 实验报告

张天乐 2018011038

## 任务一

| 编号  | 消息长度     | 计算量 | 总耗时      |
| --- | -------- | --- | -------- |
| 1   | 16384    | 0   | 0.336995 |
| 2   | 32768    | 0   | 0.484066 |
| 3   | 65536    | 0   | 0.768876 |
| 4   | 131072   | 0   | 1.26643  |
| 5   | 262144   | 0   | 2.10037  |
| 6   | 524288   | 0   | 3.79261  |
| 7   | 1048576  | 0   | 7.24227  |
| 8   | 2097152  | 0   | 14.2029  |
| 9   | 4194304  | 0   | 28.102   |
| 10  | 8388608  | 0   | 58.8824  |
| 11  | 16777216 | 0   | 120.654  |
| 12  | 33554432 | 0   | 259.283  |

- 每次消息长度是倍增的，总耗时的变化趋势是如何的？

消息长度较小时，总耗时增长倍数小于2，但不断增大。消息长度较大时，总耗时基本是成倍增的。

- 为什么会有这样的趋势？

消息本身在网络上传输的时间和消息长度成正比，而缓存预热、握手等启动开销仅需单次，在消息较短时占用时间比例显著。

## 任务二

| 编号  | 消息长度      | 计算量 | mpi_sync 总耗时 | mpi_async  总耗时 |
| --- | --------- | --- | ------------ | -------------- |
| 1   | 100000000 | 10  | 842.053      | 743.821        |
| 2   | 100000000 | 20  | 944.572      | 743.17         |
| 3   | 100000000 | 40  | 1145.67      | 842.328        |
| 4   | 100000000 | 80  | 1642.69      | 800.226        |
| 5   | 100000000 | 160 | 2451.11      | 1600.37        |

- 通信时间和计算时间满足什么关系时，非阻塞通信程序能完美掩盖通信时间？

通信时间小于计算时间时。

- 简述两份代码的不同之处。

`mpi_sync`是阻塞通信，消息发完后才进行计算；`mpi_async`是非阻塞通信，发消息和计算同时进行。
