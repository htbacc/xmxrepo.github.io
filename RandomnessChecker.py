import numpy as np
import scipy.stats as stats
import zlib

def string_to_bitstream(s):
    """Convert a string to a bitstream."""
    return [int(bit) for char in s for bit in format(ord(char), '08b')]

def entropy_test(bitstream):
    """Calculate the entropy of the bitstream."""
    _, counts = np.unique(bitstream, return_counts=True)
    probabilities = counts / len(bitstream)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def chi_square_test(bitstream):
    """Perform the Chi-Square test on the bitstream."""
    _, counts = np.unique(bitstream, return_counts=True)
    expected_counts = len(bitstream) / len(counts)
    chi_square_stat, p_value = stats.chisquare(counts, f_exp=expected_counts)
    return chi_square_stat, p_value

def frequency_test(bitstream):
    """Perform the Frequency test on the bitstream."""
    ones_count = np.sum(bitstream)
    zeros_count = len(bitstream) - ones_count
    return ones_count, zeros_count

def runs_test(bitstream):
    """Perform the Runs test on the bitstream."""
    runs = np.diff(np.where(np.diff(bitstream) != 0)[0]) + 1
    runs_mean = np.mean(runs)
    runs_variance = np.var(runs)
    return runs_mean, runs_variance

def autocorrelation_test(bitstream, lag=1):
    """Perform the Autocorrelation test on the bitstream."""
    n = len(bitstream)
    mean = np.mean(bitstream)
    autocorrelation = np.correlate(bitstream - mean, bitstream - mean, mode='full')[n-1:] / (n * np.var(bitstream))
    return autocorrelation[lag]

def compression_test(bitstream):
    """Perform the Compression test on the bitstream."""
    compressed_data = zlib.compress(bytes(bitstream))
    compression_ratio = len(compressed_data) / len(bitstream)
    return compression_ratio

# Example string
input_string = "qwt4yuey-jhbsdjhs-jshdjhs787"
bitstream = string_to_bitstream(input_string)

# Perform tests
entropy = entropy_test(bitstream)
chi_square_stat, chi_square_p_value = chi_square_test(bitstream)
ones_count, zeros_count = frequency_test(bitstream)
runs_mean, runs_variance = runs_test(bitstream)
autocorrelation = autocorrelation_test(bitstream)
compression_ratio = compression_test(bitstream)

# Print results
print(f"Entropy: {entropy}")
print(f"Chi-Square Test: Statistic={chi_square_stat}, p-value={chi_square_p_value}")
print(f"Frequency Test: Ones={ones_count}, Zeros={zeros_count}")
print(f"Runs Test: Mean={runs_mean}, Variance={runs_variance}")
print(f"Autocorrelation Test: {autocorrelation}")
print(f"Compression Test: Compression Ratio={compression_ratio}")
