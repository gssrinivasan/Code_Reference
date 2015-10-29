# include <stdio.h>
# include <omp.h>
int main ()
{
int i,j,k,n = 0;
#pragma omp parallel num_threads(2) private(i,j,k)
{ 
#pragma omp for nowait
for(i=0; i<3;++i)
    
   {
    # pragma omp parallel for private(j,k) 
    for(j=0;j<3;++j)
       { for(k=0;k<3;++k)
printf("%d,%d,%d \n",i,j,k);
        }
   }
}}
