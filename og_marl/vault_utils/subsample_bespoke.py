import numpy as np
import jax.numpy as jnp


# given bin edges and a sorted array of values, get the bin number per value
def get_bin_numbers(sorted_values, bin_edges):
    bin_numbers = np.zeros_like(sorted_values)

    def get_bin_number(bin_num, value):
        is_overflowing = value > bin_edges[bin_num]

        if is_overflowing:
            bin_num += 1
            is_doubly_overflowing = value > bin_edges[bin_num]
            if is_doubly_overflowing:
                bin_num = get_bin_number(bin_num, value)

        return bin_num

    bin_bookmark = 0

    for i, val in enumerate(sorted_values):
        bin_bookmark = get_bin_number(bin_bookmark, val)
        bin_numbers[i] = bin_bookmark

    return bin_numbers


def bin_processed_data(all_sorted_return_start_end, n_bins=500):
    # get bin edges, including final endpoint
    bin_edges = jnp.linspace(
        start=min(min(all_sorted_return_start_end[:, 0]), 0),
        stop=max(all_sorted_return_start_end[:, 0]),
        num=n_bins,
        endpoint=True,
    )
    print(all_sorted_return_start_end.shape[0])

    # get bin numbers
    bin_numbers = get_bin_numbers(all_sorted_return_start_end[:, 0], bin_edges)
    print(bin_numbers.shape[0])

    bar_labels, bar_heights = np.unique(bin_numbers, return_counts=True)

    padded_heights = np.zeros(n_bins)
    for bar_l, bar_h in zip(bar_labels, bar_heights):
        padded_heights[int(bar_l)] = bar_h

    return bar_labels, bar_heights, padded_heights.astype(int), bin_edges, bin_numbers


# sample from pdf according to heights
# BIG NOTE: CHECK THE DISPARITY, OTHERWISE YOUR DISTRIBUTION WILL BE TOO MUCH
def episode_idxes_sampled_from_pdf(pdf, bar_heights):
    num_to_sample = np.round(pdf).astype(int)
    sample_range_edges = np.concatenate([[0], np.cumsum(bar_heights)])

    assert num_to_sample.shape == bar_heights.shape

    target_sample_idxes = []
    for i, n_sample in enumerate(num_to_sample):
        sample_base = np.arange(sample_range_edges[i], sample_range_edges[i + 1])
        if n_sample <= 0:  # we don't have any to sample
            pass

        else:
            if n_sample > bar_heights[i]:  # if we sample more than all in the bar
                sample_rest = np.random.choice(sample_base, n_sample - bar_heights[i], replace=True)
                sample = np.concatenate([sample_base, sample_rest])
            else:
                sample = np.random.choice(
                    sample_base, n_sample, replace=False
                )  # make false for no replace
            target_sample_idxes = target_sample_idxes + list(np.sort(sample))
    return target_sample_idxes
