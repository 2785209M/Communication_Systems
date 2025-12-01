import random
import numpy as np

def gen_rand_vector():
    # 1. Generate random 100-bit sequence
    bit_sequence = [random.randint(0,1) for _ in range(100)]
    samples_per_bit = 10
    # 2. Expand each bit into samples, convert to float
    vector = np.array([float(bit) for bit in bit_sequence for _ in range(samples_per_bit)])
    return vector

# 3. Compute RMS voltage of the waveform
def vrms(waveform):
    v_rms_carrier = np.sqrt(np.mean(waveform**2))
    print(f"Signal RMS voltage = {v_rms_carrier:.5f}")
    return v_rms_carrier

# Function to add AWGN at given SNR_dB and rescale the signal
def add_awgn_and_rescale(carrier, snr_db, v_rms_carrier):
    # compute noise RMS
    v_rms_noise = v_rms_carrier / (10**(snr_db / 20.0))
    print(f"For SNR={snr_db} dB → Noise RMS = {v_rms_noise:.5f}")
    # generate noise (Gaussian with zero mean)
    noise = np.random.randn(len(carrier)) * v_rms_noise
    # add noise to signal
    noisy = carrier + noise
    # rescale so max(|value|) = 1
    max_abs = np.max(np.abs(noisy))
    if max_abs == 0:
        rescaled = noisy
    else:
        rescaled = noisy / max_abs
    return rescaled

# Function to just add AWGN at given SNR_dB
def add_awgn(carrier, snr_db, v_rms_carrier):
    v_rms_noise = v_rms_carrier / (10**(snr_db / 20.0))
    print(f"For SNR={snr_db} dB → Noise RMS = {v_rms_noise:.5f}")
    noise = np.random.randn(len(carrier)) * v_rms_noise
    noisy = carrier + noise
    return noisy  # no rescale here


# 5. Generate noisy waveforms & save CSVs
def snr(snr_values, vector):
    if(not snr_values):
        raise FileNotFoundError("SNR Values not found. Abort.")

    # Compute RMS voltage of the waveform
    v_rms_carrier = vrms(vector)

    noisy_waves = {}
    for snr in snr_values:
        noisy_wave = add_awgn(vector, snr, v_rms_carrier)
        # fname = f"waveform_SNR_{snr}dB.csv"
        # with open(fname, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     for v in noisy_wave:
        #         writer.writerow([v])
        # print(f"Saved {fname}")
        noisy_waves[snr] = noisy_wave
    return noisy_waves


# Define SNR values
snr_values = [24, 18, 12, 8, 6]
# Define vector
vector = gen_rand_vector()
# Generate noisy waveforms & save CSVs
noisy_waves = snr(snr_values, vector)