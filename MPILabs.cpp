#include <stdio.h>
#include <malloc.h>
#include "mpi.h"
#include <windows.h>
#include <math.h>
#include <omp.h>

void lab_1(int argc, char* argv[]) {
    int rank, size, resultlen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &resultlen);
    double elapsed_time = MPI_Wtime() - start_time;
    printf("Hello world from process %d of %d at %s. Elapsed time: %f\n", 
                                    rank, size, name, elapsed_time);
    MPI_Finalize();
}

void lab_2(int argc, char* argv[]) {
    int rank, size, resultlen, number = 1;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &resultlen);
    while (number > 0) {
        MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("Process %d of %d received number %d from process 0\n", rank, size, number);
        fflush(stdout);   
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Process %d of %d asks for number input (0 or less = exit): ", rank, size);
            fflush(stdout);
            scanf_s("%d", &number);
        }
    } 
    MPI_Finalize();
}

void lab_3(int argc, char* argv[]) { 
    int rank, size, resultlen, i = 0, n = 1;
    int diff, claster_size, claster_start, claster_end;
    bool is_flat;
    double x, sum, total_sum, start_time, elapsed_time, total_time = 0;
    const double PI_STANDARD = 3.141592653589793238462643;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &resultlen);
    while (n != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Enter number of intervals n (0 for exit): ");
            fflush(stdout);
            scanf_s("%d", &n);
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n != 0) {
            start_time = MPI_Wtime();
            is_flat = rank >= n % size;
            diff = rank - n % size;
            claster_size = is_flat ? n / size : n / size + 1;
            claster_start = is_flat ? diff * claster_size + (rank - diff) * (claster_size + 1) : rank * claster_size;
            claster_end = claster_start + claster_size;
            sum = 0;
            i = claster_start + 1;
            while (i <= claster_end) {
                x = (2.0 * i - 1) / (2.0 * n);
                sum += 4.0 / (1 + x * x);
                i++;
            }
            elapsed_time = MPI_Wtime() - start_time;
            printf("Compute time in a cluster %d-%d taken by process %d of %d: %f s\n",
                claster_start, claster_end, rank, size, elapsed_time);
            fflush(stdout);
            MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                total_sum *= 1.0 / n;
                printf("Total compute time: %f s\nResult: %.15f\nDeviation from PI_STANDARD: %.15f\n",
                    total_time, total_sum, fabs(total_sum - PI_STANDARD));
                fflush(stdout);
            }
        }
    }
    MPI_Finalize();
}

void ring(char* sbuf, char* rbuf, int msg_size, int size, int rank, int from, int to) {
    double start_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank % 2 == 0) {
        MPI_Send(sbuf, msg_size, MPI_CHAR, to, 0, MPI_COMM_WORLD);
        MPI_Recv(rbuf, msg_size, MPI_CHAR, from, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }
    else {
        MPI_Recv(rbuf, msg_size, MPI_CHAR, from, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Send(sbuf, msg_size, MPI_CHAR, to, 0, MPI_COMM_WORLD);
    }
    printf("[Process %d, RING mode] %d B message received from process %d and sent to process %d\n",
                                                                               rank, msg_size, from, to);
    printf("[Process %d, RING mode] Total messaging time for %d B: %f s\n", rank, msg_size, MPI_Wtime() - start_time);
    fflush(stdout);
}

void broadcast(char* sbuf, char* rbuf, int msg_size, int size, int rank) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(sbuf, msg_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            printf("[Process %d, BROADCAST mode] %d B message sent to process %d\n", rank, msg_size, i);
        }
    }
    else {
        MPI_Recv(rbuf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        printf("[Process %d, BROADCAST mode] %d B message received from process %d\n", rank, msg_size, 0);
    }
    printf("[Process %d, BROADCAST mode] Messaging of %d B finished at %f s\n", rank, msg_size, MPI_Wtime());
    fflush(stdout);
}

void gather(char* sbuf, char* rbuf, int msg_size, int size, int rank) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Recv(rbuf, msg_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            printf("[Process %d, GATHER mode] %d B message received from process %d\n", rank, msg_size, i);
        }
    }
    else {
        MPI_Send(sbuf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        printf("[Process %d, GATHER mode] %d B message sent to process %d\n", rank, msg_size, 0);
    }
    printf("[Process %d, GATHER mode] Messaging of %d B finished at %f s\n", rank, msg_size, MPI_Wtime());
    fflush(stdout);
}

void all_to_all(char* sbuf, char* rbuf, int msg_size, int size, int rank, 
                                MPI_Request* requests, MPI_Status* statuses) {
    for (int i = 0; i < size; i++) {
        if (rank != i) {
            MPI_Isend(sbuf, msg_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, requests);
            MPI_Request_free(&requests[0]);
            MPI_Waitall(size, requests, statuses);
            MPI_Irecv(rbuf, msg_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, requests);
            printf("[Process %d, ALL-TO-ALL mode] %d B message received from process %d, sent to process %d\n", 
                                                                                        rank, msg_size, i, i);
        }
    }
    fflush(stdout);
}

void lab_4(int argc, char* argv[]) {
    int rank, size, resultlen, number = 1;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &resultlen);
    const int B = 1, KB = 1024, MB = KB * KB;
    int from = rank == 0 ? size - 1 : rank - 1;
    int to = rank == size - 1 ? 0 : rank + 1;
    char* sbuf = (char*)malloc(MB * sizeof(char));
    char* rbuf = (char*)malloc(MB * sizeof(char));
    char* gather_rbuf = (char*)malloc(MB * sizeof(char) * size);
    MPI_Request* requests = (MPI_Request*)malloc(size * sizeof(MPI_Request));
    MPI_Status* statuses = (MPI_Status*)malloc(size * sizeof(MPI_Status));
    ring(sbuf, rbuf, B, size, rank, from, to);
    ring(sbuf, rbuf, KB, size, rank, from, to);
    ring(sbuf, rbuf, MB, size, rank, from, to);
    broadcast(sbuf, rbuf, KB, size, rank);
    broadcast(sbuf, rbuf, MB, size, rank);
    gather(sbuf, gather_rbuf, KB, size, rank);
    gather(sbuf, gather_rbuf, MB, size, rank);
    all_to_all(sbuf, rbuf, KB, size, rank, requests, statuses);
    MPI_Finalize();
    free(sbuf);
    free(rbuf);
    free(gather_rbuf);
    free(requests);
    free(statuses);
}

void fill_a_b(float* a, float* b, int chunk_m, int n, int chunk_shift) {
    for (int i = 0; i < chunk_m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + chunk_shift + 1;
        }
    }
    for (int j = 0; j < n; j++) {
        b[j] = j + chunk_shift + 1; 
    }
}

void sgemv(float *a, float *b, float *chunk_c, int chunk_m, int n) {
    for (int i = 0; i < chunk_m; i++) {
        chunk_c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            chunk_c[i] += a[i * n + j] * b[j];
        }
    }
}

void lab_5(int argc, char* argv[]) { 
    int m, chunk_m, chunk_shift, n, rank, size, resultlen;
    float *a, *b, *c, *chunk_c;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &resultlen);
    if (rank == 0) {
        printf("Enter matrix size [m] and [n] in 2 lines below:\n");
        fflush(stdout);
        scanf_s("%d x %d\n", &m, &n);
        chunk_m = m / size;
        chunk_shift = chunk_m * n * rank;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_shift, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    b = (float*) malloc(sizeof(*b) * n);
    c = (float*) malloc(sizeof(*c) * m);
    chunk_c = (float*)malloc(sizeof(*c) * chunk_m);
    a = (float*)malloc(sizeof(*a) * chunk_m * n);
    fill_a_b(a, b, chunk_m, n, chunk_shift);
    sgemv(a, b, chunk_c, chunk_m, n);
    double elapsed_time = MPI_Wtime() - start_time;
    double average_time;
    MPI_Allgather(&chunk_c[0], chunk_m, MPI_FLOAT, &c[chunk_shift], chunk_m, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &average_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    printf("[PROCESS %d] Total calculation time: %.10f s\n", rank, elapsed_time);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);
    free(a);
    free(b);
    free(c);
    free(chunk_c);
    MPI_Finalize();
}

void lab_6(int argc, char* argv[]) {
    const int DEFAULT_THREADS = 8;
    const double PI_STANDARD = 3.141592653589793238462643;
    int threads, n = 1;
    double total_sum = 0, total_time = 0;

    printf("Enter number of threads to deploy to: ");
    scanf_s("%d", &threads);
    fflush(stdout);
    threads = (threads <= 0 || threads > 64) ? DEFAULT_THREADS : threads;

    while (n != 0) {
        printf("\nEnter number of intervals n (0 for exit): ");
        scanf_s("%d", &n);
        fflush(stdout);
#pragma omp parallel num_threads(threads)
        {
            int rank = omp_get_thread_num(), size = omp_get_num_threads();
            int diff, claster_size, claster_start, claster_end;
            double x, sum, start_time, elapsed_time;
            bool is_flat;

            if (n != 0) {
                start_time = omp_get_wtime();
                is_flat = rank >= n % size;
                diff = rank - n % size;
                claster_size = is_flat ? n / size : n / size + 1;
                claster_start = is_flat 
                    ? diff * claster_size + (rank - diff) * (claster_size + 1) 
                    : rank * claster_size;
                claster_end = claster_start + claster_size;
                sum = 0;
#pragma omp parallel for private(x) shared(sum)
                for (int i = claster_start + 1; i <= claster_end; i++) {
                    x = (2.0 * i - 1) / (2.0 * n);
                    sum += 4.0 / (1 + x * x);
                }
                
                elapsed_time = omp_get_wtime() - start_time;
#pragma omp critical (results_out)
                {
                printf("Compute time in a cluster %d-%d taken by process %d of %d: %f s\n",
                    claster_start, claster_end, rank, size, elapsed_time);
                fflush(stdout);
                }

                total_sum += sum;
                total_time += elapsed_time;
            }
        }

        if (n != 0) {
            total_sum *= 1.0 / n;
            printf("Total compute time: %f s\nResult: %.15f\nDeviation from PI_STANDARD: %.15f\n",
                total_time, total_sum, fabs(total_sum - PI_STANDARD));
            fflush(stdout);
        }
    }
}

int main(int argc, char* argv[])
{
    lab_4(argc, argv);
    return 0;
}
