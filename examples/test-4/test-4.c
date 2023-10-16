void test(int *A, int *B, int len)
{
    for (int i = 0; i < len; i++) {
        A[i] = 0;
        for (int j = i; j < len; j++) {
            A[i] += B[j];
        }
    }
}
