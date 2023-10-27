void test(int *A, int *B, int len)
{
    for (int i = 0; i < len; i++) {
        if (A[i] < B[i]) {
            A[i] = B[i];
        } else {
            B[i] = A[i];
        }
    }
}
