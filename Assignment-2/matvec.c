/*
 * matrix vector multiplication: Y[] = A[][] * B[]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include <omp.h>
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

void zero(REAL A[], int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i * n + j] = 0.0;
        }
    }
}

void init(int N, REAL A[]) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (REAL) drand48();
    }
}

double check(REAL A[], REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}

void matvec_base(int M, int N, REAL Y[], REAL A[][N], REAL B[]);

void matvec_base_sub(int i_start, int Mt, int M, int N, REAL Y[], REAL A[][N], REAL B[]);

void matvec_pthread(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks);
void matvec_openmp(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed; /* for timing */
    double elapsed_pthread;
    double elapsed_openmp;
    if (argc < 2) {
        fprintf(stderr, "Usage: matvec <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL A[N][N];
    REAL B[N];
    REAL Y_base[N];
    REAL Y_pthread[N];
    REAL Y_openmp[N];
    /* more C matrix needed */

    srand48((1 << 12));
    init(N * N, (REAL *) A);
    init(N, B);

    /* example run */
    elapsed = read_timer();
    matvec_base(N, N, Y_base, A, B);
    elapsed = (read_timer() - elapsed);

    elapsed_pthread = read_timer();
    matvec_pthread(N, N, Y_pthread, A, B, num_tasks);
    elapsed_pthread = (read_timer() - elapsed_pthread);

    elapsed_openmp = read_timer();
    matvec_pthread(N, N, Y_openmp, A, B, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tMatrix Vector Multiplication: Y[N] = A[N][N] * B[N], N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matvec_base:\t\t%4f\t%4f \t\t%g\n", elapsed * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed), check(Y_base,Y_base, N));
    printf("matvec_pthread:\t\t%4f\t%4f \t\t%g\n", elapsed_pthread * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed_pthread), check(Y_base, Y_pthread, N));
    printf("matvec_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed_openmp), check(Y_base, Y_openmp, N));
    return 0;

}

void matvec_base(int M, int N, REAL Y[], REAL A[][N], REAL B[]) {
    int i, j;
    for (i = 0; i < M; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i][j] * B[j];
        }
        Y[i] = temp;
    }
}

void matvec_base_sub(int i_start, int Mt, int M, int N, REAL Y[], REAL A[][N], REAL B[]) {
    int i, j;
    for (i = i_start; i < i_start + Mt; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i][j] * B[j];
        }
        Y[i] = temp;
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
struct matvec_pthread_data {
    int M;
    int N;
    REAL *Y;
    REAL **A;
    REAL *B;
    int num_tasks;
} pthread_data;

void * matvec_pthread_func(void * thread_id) {
    int tid = (int)thread_id;
    int M = pthread_data.M;
    int N = pthread_data.N;
    REAL * Y = pthread_data.Y;
    REAL (*A)[N]  = (REAL(*)[N])pthread_data.A;
    REAL * B = pthread_data.B;
    int num_tasks = pthread_data.num_tasks;

    int Mt, i_start;
    dist(tid, M, num_tasks, &Mt, &i_start);
    matvec_base_sub(i_start, Mt, M, N, Y, A, B);
    pthread_exit(NULL);
}

void matvec_pthread(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks) {
    pthread_data.M = M;
    pthread_data.N = N;
    pthread_data.Y = Y;
    pthread_data.A = A;
    pthread_data.B = B;
    pthread_data.num_tasks = num_tasks;

    pthread_t task_pthreads[num_tasks];
    int tid;
    for (tid = 0; tid < num_tasks; tid++) {
        pthread_create(&task_pthreads[tid], NULL, matvec_pthread_func, (void*)tid);
    }

    for (tid = 0; tid < num_tasks; tid++) {
        pthread_join(task_pthreads[tid], NULL);
    }
}

void matvec_openmp(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks) {
    int i, j;
    #pragma omp parallel for shared(M,N,Y,A,B) private(i,j) num_threads(num_tasks)
    for (i = 0; i < M; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i][j] * B[j];
        }
        Y[i] = temp;
    }
}
