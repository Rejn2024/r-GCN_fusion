# Additional radar parameters for synthetic ESM observations

The current observation schema captures centre frequency, bandwidth, PRF/PRI,
pulse width, duty cycle, coherent processing interval, dwell time, waveform, and
scan type. The following additions would make observations more discriminative
without copying truth-only radar or aircraft labels into model inputs. Values
should remain representative synthetic experiment data, not claims about real
systems.

## Recommended priorities

| Priority | Observation field | Suggested representation | Why it helps |
| --- | --- | --- | --- |
| 1 | `measured_received_power_dbm` | Measurement interval plus sensor noise floor | Adds amplitude evidence for detection quality and supports realistic missingness; it must not be treated as transmitter power without propagation and geometry. |
| 1 | `measured_snr_db` | Measurement interval | Lets scoring distinguish a weak/noisy observation from a precise mismatch and can control parameter error widths. |
| 1 | `measured_angle_of_arrival_deg` | Circular estimate, error, and confidence | Supports geolocation and association across collectors. Circular uncertainty must handle the 0/360-degree boundary. |
| 1 | `observed_pri_modulation` | Category (`stable`, `staggered`, `jittered`, `sliding`, or `unknown`) with confidence | PRI patterns are often more identifying than a mean PRF and preserve behavior hidden by a single interval. |
| 1 | `observed_intrapulse_modulation` | Category (`unmodulated`, `linear_fm`, `phase_coded`, `frequency_coded`, or `unknown`) with confidence | Captures pulse structure that waveform alone does not describe. |
| 1 | `measured_scan_period_s` | Measurement interval | Adds temporal antenna-pattern behavior and complements the existing scan type and dwell time. |
| 2 | `measured_frequency_agility_mhz` | Span/rate interval plus `observed_frequency_pattern` | Separates fixed, hopping, and agile emissions that can share a centre-frequency band. |
| 2 | `measured_pulse_amplitude_db` | Per-burst summary: mean, standard deviation, min, max | Supports scan-pattern and sidelobe analysis without storing every raw pulse. |
| 2 | `measured_time_of_arrival_us` | Timestamp or inter-pulse deltas with clock error | Enables pulse deinterleaving and multi-sensor TDOA when clock provenance is retained. |
| 2 | `observed_polarization` | Category (`horizontal`, `vertical`, `circular`, `mixed`, or `unknown`) with confidence | Provides antenna/emission discrimination when the collector can measure it. |
| 2 | `measured_pulse_count` | Integer plus observation-window duration | Makes the evidential basis explicit and helps distinguish a reliable burst estimate from a sparse intercept. |
| 3 | `measured_angle_rate_deg_s` | Measurement interval | Helps associate a scanning emitter over a series, but is generally meaningful only with adequate temporal coverage. |
| 3 | `observed_lobe` | Category (`main`, `side`, or `unknown`) with confidence | Explains amplitude and parameter-quality changes, but should be treated as an inference rather than a direct measurement. |

Priority 1 is the smallest useful extension: signal quality, bearing, modulation,
and scan periodicity add information not already represented. Priority 2 is most
valuable for pulse-level or multi-collector simulations. Priority 3 should wait
until the generator models sensor geometry and antenna behavior.

The canonical graph and synthetic generators currently implement the
emitter-intrinsic subset of these recommendations: PRI modulation, intrapulse
modulation, frequency pattern and agility, scan period, and polarization. The
remaining received-power, SNR, angle/time-of-arrival, pulse-count, and lobe fields
are collector- or geometry-dependent and should be added only alongside the
sensor/propagation model described below; they must not be stored as fixed radar
or radar-mode characteristics.

## Provenance and uncertainty

Every new value should declare how it was obtained. A compatible measurement
shape is:

```json
{
  "value": 37.2,
  "error": 2.5,
  "min": 34.7,
  "max": 39.7,
  "units": "deg",
  "confidence": 0.86,
  "method": "single_sensor_aoa",
  "quality_flags": ["low_snr"]
}
```

Keep the existing `value`/`error`/`min`/`max` convention, while adding units,
confidence, estimation method, and quality flags. Categorical observations
should similarly carry `confidence`, `method`, and `unknown` as an explicit
state. Do not encode an unavailable measurement as zero.

At sensor or intercept level, also record:

- collector identifier and position/position error;
- antenna band, polarization capability, sensitivity, and calibration age;
- receiver bandwidth, sampling rate, dynamic range, and noise floor;
- observation-window start/end, clock error, and saturation/dropout flags.

These fields explain whether a parameter could have been observed and allow
models to separate emitter behavior from collector limitations.

## Derived quantities and leakage controls

Keep direct measurements distinct from derived estimates. For example,
equivalent isotropically radiated power, range, emitter location, scan/lobe
state, and geolocation quality depend on propagation, sensor geometry, or an
estimator. Put them under a separate `derived_radar_parameters` object with
`method`, input references, and confidence rather than presenting them as
directly measured ESM values.

Do not add detection range, instrumented range, track capacity, probability of
detection, false-alarm rate, transmitter peak/average power, or target
resolution directly from the knowledge graph. A passive receiver does not
directly measure these properties, and copying them from the selected radar mode
would leak the ground-truth label. They are appropriate candidate properties to
compare with independently derived evidence only.

## Simulation guidance

1. Generate latent emitter behavior first, then simulate what the collector can
   observe as a function of geometry, receiver capability, SNR, and duration.
2. Correlate error width and missingness with SNR instead of perturbing every
   parameter independently with fixed relative noise.
3. Preserve temporal correlation within an observation series: scan period,
   PRI pattern, frequency hopping, and amplitude should evolve coherently.
4. Include censored, saturated, ambiguous, and `unknown` observations so a model
   cannot equate missing data with a radar class.
5. Version the schema and add new fields incrementally. Extend candidate scoring
   only after the corresponding knowledge-graph ranges or categorical values
   exist and can be evaluated without truth-label access.
