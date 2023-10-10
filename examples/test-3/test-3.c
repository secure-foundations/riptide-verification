void test(int *A, int *B, int len)
{
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            A[i] = B[j];
        }
    }
}
