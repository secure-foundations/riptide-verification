void bfs_reduced3(
	int * restrict A,
	int * restrict indices,
	int len
)
{
	for (int j = 0; j < len; j++) {
		int next = *A;
		for(int i = indices[next]; i < indices[next + 1]; i++) {
			*A = i;
		}
	}
}

// void bfs_reduced3(
// 	const int * restrict rows,
// 	const int * restrict cols,
// 	int count,
// 	int * restrict queue,
// 	int * restrict visited,
// 	int * restrict walk,
// 	int len
// )
// {

// 	int queue_front = 0;
// 	int queue_back = 1;
// 	int walk_pos = 0;

// 	// Walk in BFS order
// 	for (int j = 0; j < len; j++) {

// 		// Fetch the next vertex and add it to @walk
// 		int next = queue[0];
// 		// queue_front++;
// 		// walk[walk_pos] = next;
// 		// walk_pos++;

// 		// Add unvisited neighbors to the queue
// 		for(int i = rows[next] ; i < rows[next + 1]; i++) {
// 			queue[0] = i;
// 		}
// 	}
// }
