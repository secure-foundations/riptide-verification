void bfs_reduced2(int * A, int len)
{
    for(int i = 0; i < len; i++) {
        if (!A[i]) {
            A[i] = 1;
        }
    }
}
