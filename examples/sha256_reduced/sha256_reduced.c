void sha256_reduced(int * restrict out, int m, int n)
{
	int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n;) {
            j = i;
            i = i * i;
        }
    }

    out[0] = i;
    out[1] = j;
}
