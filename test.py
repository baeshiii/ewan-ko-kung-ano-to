import multiprocessing as mp

# ---------- COMMON CHECK ----------

def check_dimensions(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication")

# ---------- UMA IMPLEMENTATION ----------

def multiply_row(args):
    A, B, row_index = args
    cols_B = len(B[0])
    cols_A = len(A[0])

    result_row = [0] * cols_B

    for j in range(cols_B):
        for k in range(cols_A):
            result_row[j] += A[row_index][k] * B[k][j]

    return row_index, result_row


def matrix_multiply_uma(A, B, workers=4):
    check_dimensions(A, B)

    rows_A = len(A)
    result = [None] * rows_A

    with mp.Pool(workers) as pool:
        tasks = [(A, B, i) for i in range(rows_A)]
        results = pool.map(multiply_row, tasks)

    for row_index, row_value in results:
        result[row_index] = row_value

    return result


# ---------- NUMA IMPLEMENTATION ----------

def numa_worker(start_row, A_slice, B, output):
    cols_B = len(B[0])
    cols_A = len(A_slice[0])

    for i, row in enumerate(A_slice):
        result_row = [0] * cols_B
        for j in range(cols_B):
            for k in range(cols_A):
                result_row[j] += row[k] * B[k][j]
        output[start_row + i] = result_row


def matrix_multiply_numa(A, B, nodes=4):
    check_dimensions(A, B)

    rows_A = len(A)
    nodes = min(nodes, rows_A) 
    chunk_size = (rows_A + nodes - 1) // nodes

    manager = mp.Manager()
    output = manager.list([None] * rows_A)
    processes = []

    for i in range(nodes):
        start = i * chunk_size
        end = min(start + chunk_size, rows_A)

        A_local = A[start:end]
        p = mp.Process(target=numa_worker, args=(start, A_local, B, output))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return list(output)


# ---------- MAIN ----------

if __name__ == "__main__":
    A = [
        [1, 2],
        [5, 6],
        [9, 1]
    ]  

    B = [
        [1, 0, 2],
        [0, 1, 3]
    ]  

    print("Matrix A:")
    for row in A:
        print(row)

    print("\nMatrix B:")
    for row in B:
        print(row)

    print("\nUMA Result:")
    for row in matrix_multiply_uma(A, B):
        print(row)

    print("\nNUMA Result:")
    for row in matrix_multiply_numa(A, B):
        print(row)
