import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


def one_sided_fft(t, x):
    full_amplitude_spectrum = np.abs(np.fft.fft(x))/x.size
    full_freqs = np.fft.fftfreq(x.size, np.mean(np.ediff1d(t)))
    oneinds = np.where(full_freqs >= 0.0)
    one_sided_freqs = full_freqs[oneinds]
    one_sided_amplitude_spectrum = 2*full_amplitude_spectrum[oneinds]
    return one_sided_freqs, one_sided_amplitude_spectrum


def power_spectrum(t, x):
    onef, oneamps = one_sided_fft(t, x)
    return onef, oneamps**2


def lomb_scargle_pspec(t, x):
    tstep = np.mean(np.ediff1d(t))
    freqs = np.fft.fftfreq(x.size, tstep)
    idxx = np.argsort(freqs)
    one_sided_freqs = freqs[idxx]
    one_sided_freqs = one_sided_freqs[one_sided_freqs > 0]
    # KLUDGE TO KEEP PERIODOGRAM FROM CRASHING
    one_sided_freqs = one_sided_freqs+0.00001 * \
        np.random.random(one_sided_freqs.size)
    # THE FOLLOWING LINE CRASHES WITHOUT THE KLUDGE
    pgram = sp.lombscargle(t, x, one_sided_freqs*2*np.pi)
    return one_sided_freqs, (pgram/(t.size/4))


# Sample data
# fs = 50
# fund_freq = 5
# ampl = 0.4
# t = np.arange(0, 1, 1/fs)
# x = ampl*np.cos(2*np.pi*fund_freq*t)

# Sample data
fs = 1
fund_freq = 1/365
ampl = 0.4
t = np.arange(0, 3650, 1/fs)
x = ampl*np.cos(2*np.pi*fund_freq*t)

# power spectrum calculations
# powerf, powerspec = power_spectrum(t, x)
full_amplitude_spectrum = np.abs(np.fft.fft(x))/x.size
full_freqs = np.fft.fftfreq(x.size, np.mean(np.ediff1d(t)))

oneinds = np.where(full_freqs >= 0.0)
one_sided_freqs = full_freqs[oneinds]
one_sided_amplitude_spectrum = 2*full_amplitude_spectrum[oneinds]
powerf = one_sided_freqs
powerspec = one_sided_amplitude_spectrum**2


# power spectrum calculations
# lsf, lspspec = lomb_scargle_pspec(t, x)
tstep = np.mean(np.ediff1d(t))
freqs = np.fft.fftfreq(x.size, tstep)
idxx = np.argsort(freqs)
one_sided_freqs = freqs[idxx]
one_sided_freqs = one_sided_freqs[one_sided_freqs > 0]
pgram = sp.lombscargle(t, x, one_sided_freqs*2*np.pi)
lsf = one_sided_freqs
lspspec = (pgram/(t.size/4))

# plotting
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
fig.tight_layout()
ax0.plot(t, x)
ax0.set_title('Input Data, '+str(fund_freq)+' Hz, ' +
              'Amplitude: '+str(ampl) +
              ' Fs = '+str(fs)+' Hz')
ax0.set_ylabel('Volts')
ax0.set_xlabel('Time[s]')

ax1.plot(powerf, powerspec,'-*')
ax1.set_title('FFT-based Power Spectrum')
ax1.set_ylabel('Volts**2')
ax1.set_xlabel('Freq[Hz]')

ax2.plot(lsf, lspspec,'-*')
ax2.set_title('Lomb-Scargle Power Spectrum')
ax2.set_ylabel('Volts**2')
ax2.set_xlabel('Freq[Hz]')

plt.show()
