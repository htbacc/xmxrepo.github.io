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

def scale_and_classify(value, test_name):
    """Scale the value to a range of 1 to 10 and classify the strength."""
    
    scaled_value = 0
    classification = "Weak"
    
    if test_name == "entropy":
        scaled_value = value / 8 * 10  # Entropy ranges from 0 to 8 for byte values
        if scaled_value < 3:
            classification = "Weak"
        elif scaled_value < 6:
            classification = "Medium"
        elif scaled_value < 8:
            classification = "Strong"
        else:
            classification = "Secure"
    
    elif test_name == "chi_square":
        scaled_value = min(value / 100 * 10, 10)  # Chi-Square values can be very large
        if scaled_value < 3:
            classification = "Weak"
        elif scaled_value < 6:
            classification = "Medium"
        elif scaled_value < 8:
            classification = "Strong"
        else:
            classification = "Secure"
    
    elif test_name == "frequency":
        ones_count, zeros_count = value
        total_bits = ones_count + zeros_count
        balance_ratio = min(ones_count, zeros_count) / total_bits * 10
        scaled_value = balance_ratio
        if balance_ratio < 3:
            classification = "Weak"
        elif balance_ratio < 6:
            classification = "Medium"
        elif balance_ratio < 8:
            classification = "Strong"
        else:
            classification = "Secure"
    
    elif test_name == "runs":
        runs_mean, runs_variance = value
        scaled_value = min(runs_mean / 10 * 10, 10)  # Arbitrary scaling for demonstration
        if scaled_value < 3:
            classification = "Weak"
        elif scaled_value < 6:
            classification = "Medium"
        elif scaled_value < 8:
            classification = "Strong"
        else:
            classification = "Secure"
    
    elif test_name == "autocorrelation":
        scaled_value = abs(value) * 10
        if scaled_value < 3:
            classification = "Weak"
        elif scaled_value < 6:
            classification = "Medium"
        elif scaled_value < 8:
            classification = "Strong"
        else:
            classification = "Secure"
    
    elif test_name == "compression":
        scaled_value = (1 - value) * 10
        if scaled_value < 3:
            classification = "Weak"
        elif scaled_value < 6:
            classification = "Medium"
        elif scaled_value < 8:
            classification = "Strong"
        else:
            classification = "Secure"

    return scaled_value, classification

# Example string
input_string = "qwt4yuey-jhbsdjhs-jshdjhs787"
bitstream = string_to_bitstream(input_string)

# Perform tests
entropy_val = entropy_test(bitstream)
chi_square_stat, chi_square_p_value = chi_square_test(bitstream)
frequency_val = frequency_test(bitstream)
runs_mean, runs_variance = runs_test(bitstream)
autocorrelation_val = autocorrelation_test(bitstream)
compression_val = compression_test(bitstream)

# Scale and classify results
entropy_scaled, entropy_classification = scale_and_classify(entropy_val, "entropy")
chi_square_scaled, chi_square_classification = scale_and_classify(chi_square_stat, "chi_square")
frequency_scaled, frequency_classification = scale_and_classify(frequency_val, "frequency")
runs_scaled, runs_classification = scale_and_classify((runs_mean, runs_variance), "runs")
autocorrelation_scaled, autocorrelation_classification = scale_and_classify(autocorrelation_val, "autocorrelation")
compression_scaled, compression_classification = scale_and_classify(compression_val, "compression")

# Print results
print(f"Entropy: {entropy_val} (Scaled: {entropy_scaled}, Classification: {entropy_classification})")
print(f"Chi-Square Test: Statistic={chi_square_stat}, p-value={chi_square_p_value} (Scaled: {chi_square_scaled}, Classification: {chi_square_classification})")
print(f"Frequency Test: Ones={frequency_val[0]}, Zeros={frequency_val[1]} (Scaled: {frequency_scaled}, Classification: {frequency_classification})")
print(f"Runs Test: Mean={runs_mean}, Variance={runs_variance} (Scaled: {runs_scaled}, Classification: {runs_classification})")
print(f"Autocorrelation Test: {autocorrelation_val} (Scaled: {autocorrelation_scaled}, Classification: {autocorrelation_classification})")
print(f"Compression Test: Compression Ratio={compression_val} (Scaled: {compression_scaled}, Classification: {compression_classification})")
