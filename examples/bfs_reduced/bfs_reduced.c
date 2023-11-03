void bfs_reduced(int * restrict A, int * restrict B, int len)
{
    for(int i = 0; i < len; i++) {
        if (!A[B[i]]) {
            A[B[i]] = 1;
        }
    }
}
