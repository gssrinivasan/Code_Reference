#include <stdio.h>
__global__ void hellogpu (void)
{
printf ("Hello world from GPU!! from %d \n", threadIdx.x);
}

int main(void)
{
printf("Hello world from CPU!! \n");
hellogpu <<<2, 10>>>();
cudaDeviceReset ();
return 0;
}