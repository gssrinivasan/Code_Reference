/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 * Dense matrix multiplication for studyinng various data-decomposition strategies
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>

/* read timer in second */
double read_timer()
{
    struct timeb tm; 
    ftime(&tm);
    return (double)tm.time + (double)tm.millitm/1000.0;
}

/* read timer in ms */
double read_timer_ms()
{
    struct timeb tm; 
    ftime(&tm);
    return (double)tm.time * 1000.0 + (double)tm.millitm;
}

#define REAL float
void init(int M, int N, REAL A[][N])
{
        int i, j;

        for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (REAL)drand48();
        }
    }
}

double maxerror(int M, int N, REAL A[][N], REAL B[][N])
{
        int i, j;
        double error = 0.0;

        for (i = 0; i < M; i++) 
            {
             for (j = 0; j < N; j++) 
                 {
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
void matmul_base_1(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N]);
void matmul_base_sub(int i_start, int j_start, int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N]);
void matmul_row1D(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks);
void matmul_col1D(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks);
void matmul_rowcol2D(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks);

int main(int argc,char *argv[])
{

  int N, i, j;
  int num_tasks = 5; /* 5 is default number of tasks */
  double elapsed_base, elapsed_base_1, elapsed_row1d, elapsed_col1d, elapsed_rowcol2d; /* for timing */
  if (argc < 2) {
    fprintf(stderr,"correct format: N for (NxN) and # of tasks as: matmul <n> [<#tasks(%d)>]\n", num_tasks);
    exit(1);
  }
  N = atoi(argv[1]);
  if (argc > 2) num_tasks = atoi(argv[2]);
  REAL A[N][N];
  REAL B[N][N];
  REAL C1[N][N]; // matmul_base
  REAL C2[N][N]; // matmul_base_1
  REAL C3[N][N]; // matmul_row1D
  REAL C4[N][N]; // matmul_column1D
  REAL C5[N][N]; // matmul_rowcol2D

  // generating random numbers for the A and B matrix
  srand48((1 << 12));
  init(N, N, A);
  init(N, N, B);

  /* output run */
  elapsed_base = read_timer();
  matmul_base(N,N,N,A,B,C1);
  elapsed_base = (read_timer() - elapsed_base);

  elapsed_base_1 = read_timer();
  matmul_base_1(N,N,N,A,B,C2);
  elapsed_base_1 = (read_timer() - elapsed_base_1);

  elapsed_row1d = read_timer();
  matmul_row1D(N,N,N,A,B,C3,num_tasks);
  elapsed_row1d = (read_timer() - elapsed_row1d);

  elapsed_col1d = read_timer();
  matmul_col1D(N,N,N,A,B,C4,num_tasks);
  elapsed_col1d = (read_timer() - elapsed_col1d);

  elapsed_rowcol2d = read_timer();
  matmul_rowcol2D(N,N,N,A,B,C5,num_tasks);
  elapsed_rowcol2d = (read_timer() - elapsed_rowcol2d);

  /*displaying output*/
  printf ("\n\n");
  printf("-==============================================================================-\n");
  printf("\t \t  \t \tmatmul(%dx%d)  \t \t \t \t\n", N, N);
  printf("===============================================================================\n");
  printf("-------------------------------------------------------------------------------\n");
  printf("\t\tA[M][K] * B[K][N] = C[M][N], M=N=K=%d\t\t\n",N);
  printf("-------------------------------------------------------------------------------\n");
  printf("\t \t Performance:\t\tRuntime (ms)\t MFLOPS\t \tERROR (Compared to Base)\t\t\n");
  printf("\t \t matmul_base:\t\t%4f\t%4f\t%g\n",elapsed_base*1.0e3,((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)),maxerror(N,N,C1,C1));
  printf("\t \t matmul_base_1:\t\t%4f\t%4f\t%g\t\n",elapsed_base_1*1.0e3,((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base_1)),maxerror(N,N,C1,C2));
  printf("\t \t matmul_row1D:\t\t%4f\t%4f\t%g\t\n",elapsed_row1d*1.0e3,((((2.0 * N) * N) * N) / (1.0e6 * elapsed_row1d)),maxerror(N,N,C1,C3));
  printf("\t \t matmul_col1D:\t\t%4f\t%4f\t%g\t\n",elapsed_col1d*1.0e3,((((2.0 * N) * N) * N) / (1.0e6 * elapsed_col1d)),maxerror(N,N,C1,C4));
  printf("\t \t matmul_rowcol2D:\t%4f\t%4f\t%g\t\n",elapsed_rowcol2d*1.0e3,((((2.0 * N) * N) * N)/(1.0e6 * elapsed_rowcol2d)),maxerror(N,N,C1,C5));
  printf("---------------------------------------------------------------------------------\n\n");
return 0;
}


void matmul_base(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N])
{
	// This is to compute the row-wise elements in the matrix C
        int i,j,k;

	for(i = 0; i < M; i++)
           {
            for(j = 0; j < N; j++) 
               { 
                 C[i][j] = 0;
                 for(k = 0; k < K; k++)
                    C[i][j] += A[i][k]*B[k][j];
               }
           }
}

void matmul_base_1(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N])
{
	// This is to compute the column-wise elements in the matrix C
        int i,j,k;

	for(j=0;j<M;j++)
           {
            for(i=0;i<N;i++) 
               { 
                 C[i][j] = 0;
                 for(k=0;k<K;k++)
                    C[i][j]+=A[i][k]*B[k][j]
               }
           }


}

void matmul_base_sub(int i_init, int j_init,int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N])
{
        /*
         Assuming i_init, j_init for C matrix
         M, K, N general value from A and B
        */
                 int i=i_init,j=j_init,k;

                 C[i][j] = 0;
                 for(k = 0; k < N; k++)
                    C[i][j] += A[i][k]*B[k][j];

}

void matmul_row1D (int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks)
{
// consider 1 col from B and A matrix to generate 1 row in C matrix

        int i,j,k,z;
        int initiali =0, initialj=0;
        int num_of_work_in_one_task = N / num_tasks;
        int num_of_matdata_left = N % num_tasks; 
// calculated for N not being divisible by #tasks
        int residue=0,residuer;
        int not_in_cycle = N-num_of_matdata_left; 
        if (num_of_matdata_left == 0)//for N divisible by num_tasks
           {
            for (i=0 ; i<num_tasks; i++)
                {
                 for (j=0; j<num_of_work_in_one_task; ++j)
                     {
                       for (initiali=0;initiali<N;initiali++)
	                  {    
		            int i_init = initiali;
		            int j_init = initialj;
 matmul_base_sub(i_init,j_init,M,K,N,A,B,C);
                          }
                       initialj = initialj + 1;
                     }
                }
            }
        else //for N non divisible by num_tasks
        {
         for (i=0 ; i<num_tasks; i++)
             {
              for (j=0; j<num_of_work_in_one_task; ++j)
                  {   
                   for (initialj=0; initialj<N;++initialj)
                       {
                        int i_init = initiali;
		        int j_init = initialj;
                        matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
                       }
                   initiali++;    
                   if (initiali == not_in_cycle-2)  
                   residue = not_in_cycle; 
                       if (residue == not_in_cycle)
                          {
                              for (z=0;z<num_of_matdata_left;z++)
                                  {
                                    for (initialj=0; initialj<N;++initialj)
                                        {
                                         int i_init = initiali;
		                         int j_init = initialj;
                                         matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
                                        }
                                    initiali++;    
                                   }
                          }
                  }
             }
        }
}

void matmul_col1D (int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks)
{
	// consider 1 col from B and A matrix to generate 1 col in C matrix

        int i,j,k,z;
        int initiali =0, initialj=0;
        int num_of_work_in_one_task = N / num_tasks;
        int num_of_matdata_left = N % num_tasks; 
// calculated for N not being divisible by #tasks
        int residue=N,residuer;
        int not_in_cycle = N*num_of_work_in_one_task; 
        if (num_of_matdata_left == 0)//for N divisible by num_tasks
           {
            for (i=0 ; i<num_tasks; i++)
                {
                 for (j=0; j<num_of_work_in_one_task; ++j)
                     {
                       for (initialj=0;initialj<N;initialj++)
	                  {    
		            int i_init = initiali;
		            int j_init = initialj;
matmul_base_sub(i_init,j_init,M,K,N,A,B,C);
                          }
                       initiali = initiali + 1;
                     }
                }
            }
        else//for N non divisible by num_tasks
        {
         for (i=0 ; i<num_tasks; i++)
             {

              for (j=0; j<num_of_work_in_one_task; ++j)
                  {    
                   for (initiali=0; initiali<N;++initiali)
                       {
                        int i_init = initiali;
		        int j_init = initialj;
                        matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
                       }
                   initialj++;
                   if (initialj == not_in_cycle)  
                   residue = not_in_cycle;
                   if    (initialj+1 == N-num_of_matdata_left)
                          {   
                              for (z=0;z<num_of_matdata_left;z++)
                                  {
                                    for (initiali=0; initiali<N;++initiali)
                                        {
                                         int i_init = initiali;
		                         int j_init = initialj;
                                         matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
                                        }
                                    initialj++;    
                                   } 
                          }    

                  }
             }
        }
}

void matmul_rowcol2D(int M, int K, int N, REAL A[][K], REAL B[][N], REAL C[][N], int num_tasks)
{
int i,j,k;
int initiali=0,initialj=0;
int i_init, j_init;
double b = N/sqrt(num_tasks);
int no_of_rows_in_block = b;
int no_of_blocks = N/no_of_rows_in_block;
int completed_rows = no_of_rows_in_block*no_of_blocks;
int leftdata = N-completed_rows;
int residue = 0;
int rownum_completed = 0;
/*
For the computation of C matrix for N being non-divisible by num_tasks
*/
if (N%no_of_rows_in_block !=0)
   {
    for (residue=0;residue<N;)//forout
        {
         /*
          the first "if" for the completed row condition, is to check when we need to move to the lower half of the C matrix computation. Completed_rows variable gives the N value. Once the N elements are completed, we need to move to the next block computation, which will follow the row below the already computed one.
         */
         if (residue != completed_rows )//ifmain
            {
             for (k=0;k<=no_of_blocks;)//formain
	 {
                 /*
                  The "if" here for the "no_of_blocks" is used for the computation of the left-over block data to be considered,
                   eg., for ./matmul 32 5
no_of_rows_in_block = 14
so, two 14X14 block will be computed and 4 elements will be left,
hence,there will be one 14X4 (left-most) and one 4X14(down) blocks 
will be left unattended,which is computed below                                           
                 */
	  if (k == no_of_blocks)//if
	     {
	      for(i=0;i<no_of_rows_in_block;i++)//for1
	         {
                          for(j=0; j < leftdata ; j++)
	             {
i_init = initiali;
	              j_init = initialj;
	              matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
	              initialj++;
                             }
                          initialj = initialj-leftdata;
                          initiali++;
                         }//for1
                      residue = residue + no_of_rows_in_block;
                      initialj =0;
                      k=k+1;
	     }//if
                  else
                     {
	      for(i=0; i<no_of_rows_in_block;i++)//for2
	         {
	          for(j=0; j<no_of_rows_in_block; j++)//for3
	             {
                              i_init = initiali;
	              j_init = initialj;
	              matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
                              initialj++;
                             }//for3
                          initialj = initialj-no_of_rows_in_block;
	          initiali++;
                         }//for2
                      k=k+1;
                      initiali = initiali-no_of_rows_in_block;
                      initialj = initialj+no_of_rows_in_block;
	     }//else
                 }//formain part

            }//ifmain

         else
            {
             rownum_completed = rownum_completed + no_of_rows_in_block;
             for(i=0;i<leftdata;i++)//for1
	{
                 for(j=0; j<no_of_rows_in_block; j++)
	    {
                     i_init = initiali;
	     j_init = initialj;
	     matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
	     initialj++;
                    }
	 initialj = initialj-no_of_rows_in_block;
                 initiali++;
                }//for1
             initiali = initiali - leftdata;
             initialj = initialj+no_of_rows_in_block;
             if (rownum_completed == completed_rows )
                {
                 residue = residue + no_of_rows_in_block;
                }
            }
        }//forout
     /* 
     This is to compute the left-over part of C matrix 
     after the block decomposition is done, 
     say, in our eg. of ./matmul 32
     there will be 4X4 matrix to be computed, done here.
     */
    for (i=0 ; i<leftdata;i++ )
        {
         for (j=0 ; j<leftdata; j++)
             {
              i_init = initiali;
              j_init = initialj;
              matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
              initialj++;
             }
         initialj = initialj-leftdata;
         initiali++;
        }

   }
/*
For the computation of C matrix for N being non-divisible by num_tasks

This part of the program will compute the C matrix block-wise, 
for N being exactly divisible by squareroot of num_tasks.
the code computes the C matrix block by block using the counter values
initiali and initialj values.
*/
else 
   {                        
    for (completed_rows =0; completed_rows<2;completed_rows++)                       
        { 
         for (k=0;k<no_of_blocks;)
             {
              for(i=0; i<no_of_rows_in_block;i++)//for2
	 {
	  for(j=0; j<no_of_rows_in_block; j++)//for3
	     {
                      i_init = initiali;
	      j_init = initialj;
	      matmul_base_sub(i_init,j_init,M,K,N,A, B, C);
                      initialj++;
                     }//for3
                  initialj = initialj-no_of_rows_in_block;
	  initiali++;
                 }//for2
              k=k+1;
              initiali = initiali-no_of_rows_in_block;
              initialj = initialj+no_of_rows_in_block;
             }
         initiali = initiali+no_of_rows_in_block;
         initialj = 0;
        }

    }
}







