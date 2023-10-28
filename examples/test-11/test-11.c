void test(int *A, int *B, int lenA, int lenB)
{
    for (int i = 0; i < lenA; i++) {
        for (int j = 0; j < lenB; j++) {
            A[i] = B[j];
        }
    }
}
