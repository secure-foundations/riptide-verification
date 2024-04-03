// BYTE data[64];
// WORD datalen;
// unsigned long long bitlen;
// WORD state[8];

typedef unsigned char BYTE;
typedef unsigned int  WORD;

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

// static const WORD k[64] = {
// 	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
// 	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
// 	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
// 	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
// 	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
// 	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
// 	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
// 	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
// };

void sha256_transform(WORD * restrict ctx_state, const WORD * restrict data, const WORD * restrict k, WORD * restrict m)
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2;

	// for (i = 0, j = 0; i < 16; ++i, j += 4)
	// 	m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (i = 0; i < 16; ++i)
        m[i] = data[i];

	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx_state[0];
	b = ctx_state[1];
	c = ctx_state[2];
	d = ctx_state[3];
	e = ctx_state[4];
	f = ctx_state[5];
	g = ctx_state[6];
	h = ctx_state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	// TODO: This is done due to a limitation in our heuristics
	// In the LLVM version, we have temporary definitions p_i = ctx_state + i
	// And p_i is used after the loop here.
	// But right now we cannot infer the equality of of p_i = ctx_state + i
	// at cut point 3 (i.e. the last loop)
	ctx_state += a; ctx_state++;
	ctx_state += b; ctx_state++;
	ctx_state += c; ctx_state++;
	ctx_state += d; ctx_state++;
	ctx_state += e; ctx_state++;
	ctx_state += f; ctx_state++;
	ctx_state += g; ctx_state++;
	ctx_state += h;
}
