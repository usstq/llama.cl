#pragma once

#include <omp.h>
#include <stdint.h>

//=================================================================================
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename F>
inline void parallel_nt(int64_t begin,
                        int64_t end,
                        int64_t grain_size,
                        const F& f) {
#pragma omp parallel
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See
    // #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      f(begin_tid, std::min(end, chunk_size + begin_tid));
    }
  }
}

//=================================================================================
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

typedef short float16;
float16 to_fp16(float v) {
  return static_cast<float16>(_mm_cvtsi128_si32(
      _mm256_cvtps_ph(_mm256_set1_ps(v), _MM_FROUND_CUR_DIRECTION)));
}
float to_fp32(float16 v) {
  float x[8];
  _mm256_storeu_ps(x, _mm256_cvtph_ps(_mm_set1_epi16(v)));
  return x[0];
}

static inline float _mm256_reduce_add_ps(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

inline __m256i get_mask(int N7) {
  static __m256i mask[] = {
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
      _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
      _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
      _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
      _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
      _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
      _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
  };
  return _mm256_loadu_si256(&mask[N7]);
}

inline void hsum(__m256& x) {
  __m256 y;                             // x:  0 1 2 3   4 5 6 7
  y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
  x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
  y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
  x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
  y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
  x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
}

static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

static inline float hmax_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_max_ps(res, _mm256_castps256_ps128(x));
  res = _mm_max_ps(res, _mm_movehl_ps(res, res));
  res = _mm_max_ps(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

static inline float get_amax(float* x, int len) {
  int i = 0;
  float amax = 0;
#if __AVX2__
  // https://stackoverflow.com/questions/63599391/find-absolute-in-avx
  auto sign_bit = _mm256_set1_ps(-0.0f);
  auto v_max_abs = _mm256_setzero_ps();
  for (; i + 8 <= len; i += 8) {
    auto v = _mm256_loadu_ps(x + i);
    v = _mm256_andnot_ps(sign_bit, v);
    v_max_abs = _mm256_max_ps(v_max_abs, v);
  }
  amax = hmax_float_8(v_max_abs);
#endif
  for (; i < len; i++) {
    auto a = std::abs(x[i]);
    if (amax < a)
      amax = a;
  }
  return amax;
}

// round(x * id)
static inline void quant_row_q8_0(float* x, int8_t* qx, int len, float id) {
  int i = 0;
#if __AVX2__
  auto v_id = _mm256_set1_ps(id);
  for (; i + 8 <= len; i += 8) {
    auto v = _mm256_loadu_ps(x + i);
    v = _mm256_mul_ps(v, v_id);
    v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
    auto v_i32 = _mm256_cvtps_epi32(v);

    auto high4 = _mm256_extractf128_si256(v_i32, 1);
    auto low4 = _mm256_castsi256_si128(v_i32);
    auto packed = _mm_packs_epi32(low4, high4);
    packed = _mm_packs_epi16(packed, packed);
    _mm_storeu_si64(qx + i, packed);
  }
#endif
  for (; i < len; i++) {
    qx[i] = std::round(x[i] * id);
  }
}

struct VNNI_Sequence {
  __m256i operator()(__m256i acc, const __m256i x_s8, const __m256i y_u8) {
#if __AVXVNNI__
    return _mm256_dpbusd_epi32(acc, y_u8, x_s8);
#elif __AVX2__
    const __m256i ones = _mm256_set1_epi16(1);
    // u8 x s8
    const __m256i dot = _mm256_maddubs_epi16(y_u8, x_s8);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(dot, ones));
#else
#error "at least AVX2 is required!"
#endif
  }
};

// intrinsic helpers
//  VNNI has long latency, high throughput, thus requires more independent data
//  to fill the port & hide the latency, which means more independent
//  accumulator regs in brgemm
struct vnni_inst {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i zero = _mm256_setzero_si256();

  __m256i operator()(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return summed_pairs;
#elif __AVXVNNI__
    // Get absolute values of x vectors (x becomes u8 : 0~128)
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors (negative sign of x is combined with y)
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return summed_pairs;
#else
    // Get absolute values of x vectors (x becomes u8 : 0~128)
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors (negative sign of x is combined with y)
    const __m256i sy = _mm256_sign_epi8(y, x);

    // u8 x s8
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_madd_epi16(dot, ones);
#endif
  }
};

struct VNNI_INT8_Sequence {
  __m256i operator()(__m256i acc, const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    return _mm256_dpbssd_epi32(acc, x, y);
#elif __AVXVNNI__
    // Get absolute values of x vectors (x becomes u8 : 0~128)
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors (negative sign of x is combined with y)
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_dpbusd_epi32(acc, ax, sy);
#elif __AVX2__
    // Get absolute values of x vectors (x becomes u8 : 0~128)
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors (negative sign of x is combined with y)
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i ones = _mm256_set1_epi16(1);

    // u8 x s8
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(dot, ones));
#else
#error "at least AVX2 is required!"
#endif
  }
};

inline __m256 exp_ps_avx2(__m256 src) {
  static __m256 exp_ln_flt_min_f =
      _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));  // log(FLT_MIN)
  static __m256 exp_ln_flt_max_f =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));  // log(FLT_MAX)
  static __m256 exp_log2ef =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));  // log2(e)
  static __m256 half =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000));  // 0.5f
  static __m256 ln2f =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));  // ln(2)
  static __m256 one =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000));        // 1.0f
  static __m256i exponent_bias = _mm256_set1_epi32(0x0000007f);  // 127
  static constexpr int n_mantissa_bits = 23;
  static __m256 exp_pol1 =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb));  // p1 = 0.999999701f
  static __m256 exp_pol2 =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3));  // p2 = 0.499991506f
  static __m256 exp_pol3 =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40));  // p3 = 0.166676521f
  static __m256 exp_pol4 =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d));  // p4 = 0.0418978221f
  static __m256 exp_pol5 = _mm256_castsi256_ps(
      _mm256_set1_epi32(0x3c07cfce));  // p5 = 0.00828929059f
  static __m256 two = _mm256_castsi256_ps(_mm256_set1_epi32(0x40000000));  // 2
  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

  // get mask of values lower than log(FLT_MIN) to zero them in the output
  auto zero_mask = _mm256_cmp_ps(src, exp_ln_flt_min_f, _CMP_LT_OS);

  // clip src
  src = _mm256_min_ps(src, exp_ln_flt_max_f);
  src = _mm256_max_ps(src, exp_ln_flt_min_f);

  // aux1 : r
  auto aux1 = src;

  // calculate exp(x)
  // fx = x * log2(e) + 0.5
  src = _mm256_mul_ps(src, exp_log2ef);
  src = _mm256_add_ps(src, half);

  // tmp = floorf(fx)
  src = _mm256_floor_ps(src);

  // aux1 = x - fx * ln2
  aux1 = _mm256_fnmadd_ps(src, ln2f, aux1);

  // We do not count 2^n here, because n can reach 128 and 2^128 is not
  // representable by fp32, so to get around this problem, instead of computing
  // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
  // and 2 are numbers representable in fp32.

  // compute 2^(n-1)
  src = _mm256_sub_ps(src, one);
  auto aux2_i = _mm256_cvtps_epi32(src);
  aux2_i = _mm256_add_epi32(aux2_i, exponent_bias);
  aux2_i = _mm256_slli_epi32(aux2_i, n_mantissa_bits);

  // set zeroes at those points which were < log(FLT_MIN)
  auto zero = _mm256_setzero_ps();
  auto aux2 = _mm256_blendv_ps(_mm256_castsi256_ps(aux2_i), zero, zero_mask);

  // compute polynomial
  src = exp_pol5;
  src = _mm256_fmadd_ps(src, aux1, exp_pol4);
  src = _mm256_fmadd_ps(src, aux1, exp_pol3);
  src = _mm256_fmadd_ps(src, aux1, exp_pol2);
  src = _mm256_fmadd_ps(src, aux1, exp_pol1);
  src = _mm256_fmadd_ps(src, aux1, one);

  // y = y * 2^n
  src = _mm256_mul_ps(src, aux2);
  src = _mm256_mul_ps(src, two);
  return src;
}

inline __m256 sigmoid_avx2(__m256 x) {
  // 1/(1+exp(-x))
  static __m256 one = _mm256_set1_ps(1.0f);
  auto neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
  return _mm256_rcp_ps(_mm256_add_ps(one, exp_ps_avx2(neg_x)));
}

// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);

inline __m256 mm256_uni_loadu_ps(float* a) {
  return _mm256_loadu_ps(a);
}
inline void mm256_uni_storeu_ps(float* a, __m256 v) {
  _mm256_storeu_ps(a, v);
}
inline __m256 mm256_uni_loadu_ps(float16* a) {
  auto vec_f16 = _mm_loadu_si128(reinterpret_cast<__m128i*>(a));
  return _mm256_cvtph_ps(vec_f16);
}
inline void mm256_uni_storeu_ps(float16* addr, __m256 xps) {
  return _mm_storeu_si128(reinterpret_cast<__m128i*>(addr),
                          _mm256_cvtps_ph(xps, _MM_FROUND_CUR_DIRECTION));
}

template <typename T>
float dot_product(T* a, T* b, size_t n) {
  size_t i = 0;
  float sum = 0.0f;
  auto vsum = _mm256_set1_ps(0.0f);
  for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
    auto va = mm256_uni_loadu_ps(a + i);
    auto vb = mm256_uni_loadu_ps(b + i);
    vsum = _mm256_fmadd_ps(va, vb, vsum);
  }
  sum = _mm256_reduce_add_ps(vsum);
  for (; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline float get_max(float* a, size_t size) {
  auto v_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
  auto v_a = v_max;
  size_t i = 0;
  // process vector body
  while (i + vec_len_f32_avx2 <= size) {
    v_a = _mm256_loadu_ps(a + i);
    v_max = _mm256_max_ps(v_max, v_a);
    i += vec_len_f32_avx2;
  }

  // process tails
  if (i < size) {
    auto mask = get_mask(size - i);
    v_a = _mm256_maskload_ps(a + i, mask);
    v_a = _mm256_blendv_ps(v_max, v_a, _mm256_castsi256_ps(mask));
    v_max = _mm256_max_ps(v_max, v_a);
  }
  return hmax_float_8(v_max);
}

inline float exp_reduce_sum(float* a, const float max, const size_t size) {
  size_t i = 0;
  __m256 v_a;
  auto v_max = _mm256_set1_ps(max);
  auto v_sum = _mm256_set1_ps(0.0f);
  while (i + vec_len_f32_avx2 <= size) {
    v_a = _mm256_loadu_ps(a + i);
    v_a = _mm256_sub_ps(v_a, v_max);
    v_a = exp_ps_avx2(v_a);
    v_sum = _mm256_add_ps(v_sum, v_a);
    _mm256_storeu_ps(a + i, v_a);
    i += vec_len_f32_avx2;
  }

  if (i < size) {
    auto mask = get_mask(size - i);
    v_a = _mm256_maskload_ps(a + i, mask);
    v_a = _mm256_sub_ps(v_a, v_max);
    v_a = exp_ps_avx2(v_a);
    v_a = _mm256_blendv_ps(_mm256_setzero_ps(), v_a, _mm256_castsi256_ps(mask));
    v_sum = _mm256_add_ps(v_a, v_sum);
    _mm256_maskstore_ps(a + i, mask, v_a);
  }
  return _mm256_reduce_add_ps(v_sum);
}

inline void multiply_scale(float* a, const float val, const size_t size) {
  auto v_scale = _mm256_set1_ps(val);
  __m256 v_a;
  size_t i = 0;
  while (i + vec_len_f32_avx2 <= size) {
    v_a = _mm256_loadu_ps(a + i);
    v_a = _mm256_mul_ps(v_a, v_scale);
    _mm256_storeu_ps(a + i, v_a);
    i += vec_len_f32_avx2;
  }
  if (i < size) {
    auto mask = get_mask(size - i);
    v_a = _mm256_maskload_ps(a + i, mask);
    v_a = _mm256_mul_ps(v_a, v_scale);
    _mm256_maskstore_ps(a + i, mask, v_a);
  }
}

inline void _softmax(float* a, const size_t len) {
  // exp sum
  float max = get_max(a, len);
  float sum = exp_reduce_sum(a, max, len);
  float scale = 1.0f / sum;
  multiply_scale(a, scale, len);
}

template <typename TO, typename TI>
void accumulate_weighted_v(TO* out, float weight, TI* v, size_t S) {
  size_t i = 0;
  auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
  for (; i <= S - 8; i += 8) {
    auto v_value = mm256_uni_loadu_ps(v + i);
    auto v_out = mm256_uni_loadu_ps(out + i);
    v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
    mm256_uni_storeu_ps(out + i, v_out);
  }
  for (; i < S; i++) {
    out[i] += weight * v[i];
  }
}

template <typename T>
void reduce_v(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
  size_t i = 0;
  for (; i <= S - 8; i += 8) {
    auto* src = temp + i;
    auto result_vec_fp32 = _mm256_set1_ps(0.0f);
    for (size_t m = 0; m < M; m++) {
      auto o_vec_fp32 = mm256_uni_loadu_ps(src);
      result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
      src += temp_stride;
    }
    mm256_uni_storeu_ps(dst + i, result_vec_fp32);
  }
  for (; i < S; i++) {
    auto* src = temp + i;
    float sum = 0.0f;
    // sum result from all threads partition
    for (size_t m = 0; m < M; m++) {
      sum += src[0];
      src += temp_stride;
    }
    dst[i] = sum;
  }
}