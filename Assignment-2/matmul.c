/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float

void init(int M, int N, REAL A[][N]) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (REAL) drand48();
        }
    }
}

double maxerror(int M, int N, REAL A[][N], REAL B[][N]) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i][j] - B[i][j]) / A[i][j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void matmul_base(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N]);
void matmul_base_sub(int i_start, int j_start, int Mt, int Nt, int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N]);
void matmul_pthread(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks);
void matmul_openmp(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed_base, elapsed_pthread, elapsed_openmp; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*5); /* we use 5 matrix in this example */
    /* below is a cast from memory buffer to a 2-d row-major array */
    REAL (*A)[N] = (REAL(*)[N])heap_buffer;
    REAL (*B)[N] = (REAL(*)[N])&heap_buffer[N*N];
    REAL (*C_base)[N] = (REAL(*)[N])&heap_buffer[2*N*N];
    REAL (*C_pthread)[N] = (REAL(*)[N])&heap_buffer[3*N*N];
    REAL (*C_openmp)[N] = (REAL(*)[N])&heap_buffer[4*N*N];

    srand48((1 << 12));
    init(N, N, A);
    init(N, N, B);

    /* example run */
    elapsed_base = read_timer();
    matmul_base(N, N, N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);

    elapsed_pthread = read_timer();
    matmul_pthread(N, N, N, A, B, C_pthread, num_tasks);
    elapsed_pthread = (read_timer() - elapsed_pthread);

    elapsed_openmp = read_timer();
    matmul_openmp(N, N, N, A, B, C_openmp, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    printf("======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_pthread:\t\t%4f\t%4f \t\t%g\n", elapsed_pthread * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_pthread)), maxerror(N, N, C_base, C_pthread));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));
    free(heap_buffer);
    return 0;
}

void matmul_base(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N]) {
    int i, j, k;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < K; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
}

void matmul_base_sub(int i_start, int j_start, int Mt, int Nt, int M, int K, int N,
        REAL A[][K], REAL B[][N], REAL C[][N]) {
    int i, j, k;
    for (i = i_start; i < i_start + Mt; i++) {
        for (j = j_start; j < j_start + Nt; j++) {
            REAL temp = 0.0;
            for (k = 0; k < K; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
}

void dist(int tid, int N, int num_tasks, int *Nt, int *start) {
    int remain = N % num_tasks;
    int esize = N / num_tasks;
    if (tid < remain) { /* each of the first remain task has one more element */
        *Nt = esize + 1;
        *start = *Nt * tid;
    } else {
        *Nt = esize;
        *start = esize * tid + remain;
    }
}

/* the thread arguments in this struct are all read-only access, so we will not need to
create thread-specific copy of it, i.e. all thread share
 */
struct matmul_pthread_data {
    int M;
    int K;
    int N;
    REAL **A;
    REAL **B;
    REAL **C;
    int num_tasks;
} pthread_data;

void * matmul_thread_func(void * threadid) {
    int tid = (int)threadid;
    int M = pthread_data.M;
    int K = pthread_data.K;
    int N = pthread_data.N;
    REAL (*A)[K]  = (REAL(*)[K])pthread_data.A;
    REAL (*B)[N]  = (REAL(*)[N])pthread_data.B;
    REAL (*C)[N]  = (REAL(*)[N])pthread_data.C;
    int num_tasks = pthread_data.num_tasks;

    int Mt, i_start;
    dist(tid, M, num_tasks, &Mt, &i_start);
    matmul_base_sub(i_start, 0, Mt, N, M, K, N, A, B, C);

    pthread_exit(NULL);
}

void matmul_pthread(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks) {
    pthread_data.M = M;
    pthread_data.K = K;
    pthread_data.N = N;
    pthread_data.A = A;
    pthread_data.B = B;
    pthread_data.C = C;
    pthread_data.num_tasks = num_tasks;

    pthread_t task_threads[num_tasks];
    int tid;
    for (tid = 0; tid < num_tasks; tid++) {
        pthread_create(&task_threads[tid], NULL, matmul_thread_func, (void*)tid);
    }

    for (tid = 0; tid < num_tasks; tid++) {
        pthread_join(task_threads[tid], NULL);
    }
}

void matmul_openmp(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(M,K,N,A,B,C) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < K; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
}