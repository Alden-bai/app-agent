from scripts.benchmark_liger_sft import sample_indices, select_batches


def test_large_embedding_gradient_sampling_is_in_bounds():
    size = 248320 * 2048
    indices = sample_indices(size)
    assert len(indices) == 512
    assert indices[0] == 0
    assert indices[-1] == size - 1
    assert all(0 <= i < size for i in indices)
    assert indices == sorted(set(indices))
    assert sample_indices(1) == [0]
    assert sample_indices(3) == [0, 1, 2]


def test_benchmark_selection_retains_complete_real_rows_and_longest():
    rows = [{"input_ids": list(range(n)), "labels": [-100] * (n - 1) + [1]} for n in range(1, 101)]
    batches = select_batches(rows)
    assert len(batches["median_batch16"]) == 16
    assert len(batches["p95_batch8"]) == 8
    assert batches["longest_batch1"] == [(99, rows[99])]
    for batch in batches.values():
        assert len({i for i, _ in batch}) == len(batch)
        assert all(row is rows[i] for i, row in batch)
