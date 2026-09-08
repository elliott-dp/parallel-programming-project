/* roofline.c -- measures the node-level roofline roofs and a library ceiling.
 *
 *   make roofline            (needs OpenBLAS: make roofline BLAS=1)
 *   ./bench/roofline > results/roofs.csv
 *
 * Emits CSV consumed by scripts/plot_results.py --kind roofline.
 *
 * Measures the two roofline roofs plus an OpenBLAS ceiling.
 *   1. STREAM triad  -> achievable DRAM bandwidth (the sloped roof)
 *   2. FMA chain     -> achievable peak flop rate  (the flat roof)
 *   3. OpenBLAS dgemm at the SUMMA panel shape -> a library reference line
 */
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef USE_BLAS
#include <cblas.h>
#endif
static double wt(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
                       return t.tv_sec+1e-9*t.tv_nsec;}
static double *al(size_t n){void*p;if(posix_memalign(&p,64,n*sizeof(double)))exit(2);
                            return (double*)p;}

static double stream_triad(void){
    size_t n = 40u<<20;                 /* 320 MB per array, far beyond LLC */
    double *a=al(n),*b=al(n),*c=al(n),best=1e30;
    for(size_t i=0;i<n;i++){a[i]=1.0;b[i]=2.0;c[i]=3.0;}
    for(int r=0;r<5;r++){
        double t=wt();
        for(size_t i=0;i<n;i++) a[i]=b[i]+3.0*c[i];
        t=wt()-t; if(t<best)best=t;
    }
    free(a);free(b);free(c);
    return 3.0*n*sizeof(double)/best/1e9;   /* 2 reads + 1 write */
}

static double peak_fma(void){
    /* 16 independent accumulators: enough to hide FMA latency and let the
     * compiler use the full vector width. Nothing here touches memory. */
    enum {NA=16}; double a[NA], bb=1.0000001, cc=0.9999999, best=1e30;
    for(int i=0;i<NA;i++) a[i]=1.0+i*1e-9;
    long iters=20000000L;
    for(int r=0;r<3;r++){
        double t=wt();
        for(long it=0;it<iters;it++)
            for(int i=0;i<NA;i++) a[i]=a[i]*bb+cc;
        asm volatile("" : : "r"(a) : "memory");
        t=wt()-t; if(t<best)best=t;
    }
    double sink=0; for(int i=0;i<NA;i++) sink+=a[i];
    /* The result MUST be consumed, or the whole loop is dead code and the
     * measured rate is meaningless (this bit me: it reported 3e7 Gflop/s). */
    fprintf(stderr, "# fma sink %.3e\n", sink);
    return 2.0*NA*iters/best/1e9;        /* 2 flops per FMA */
}

int main(void){
    printf("metric,value,unit\n");
    printf("stream_triad,%.2f,GB/s\n", stream_triad());
    /* Reported for reference only. A compiler barrier is needed to stop the
     * loop being eliminated, and that barrier itself then limits the rate, so
     * this UNDER-estimates peak and must not be used as the roof. The roof
     * below is the measured library rate instead. */
    printf("fma_microbench,%.2f,Gflop/s\n", peak_fma());
#ifdef USE_BLAS
    openblas_set_num_threads(1);
    int m=1024,n=1024,k=256; double best=1e30;
    double *A=al((size_t)m*k),*B=al((size_t)k*n),*C=al((size_t)m*n);
    for(size_t i=0;i<(size_t)m*k;i++)A[i]=1.0/(i+1);
    for(size_t i=0;i<(size_t)k*n;i++)B[i]=1.0/(i+2);
    memset(C,0,(size_t)m*n*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,k,B,n,1.0,C,n);
    for(int r=0;r<10;r++){
        double t=wt();
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,k,B,n,1.0,C,n);
        t=wt()-t; if(t<best)best=t;
    }
    printf("openblas_dgemm,%.2f,Gflop/s\n", 2.0*m*n*k/best/1e9);
#endif
    return 0;
}
