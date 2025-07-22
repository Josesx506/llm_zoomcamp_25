from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from tqdm import tqdm


def multithread_func(seq, func, max_workers=None):
    """
    Make api calls or run any other function in parallel using multiple threads
    """
    if max_workers is None:
        max_workers = max(1, int(cpu_count() * 0.75))

    results = [None] * len(seq)  # Preallocate to preserve input order
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, el): i for i, el in enumerate(seq)}

        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            results[idx] = future.result()

    return results
