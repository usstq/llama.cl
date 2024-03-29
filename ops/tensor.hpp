#pragma once

#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <typeinfo>
#include <stdint.h>
#include <map>
#include <thread>

#ifdef _WIN32
#include <cstdlib>
#endif

#if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
#include <sycl/sycl.hpp>
#endif

template <typename... Args>
static inline void throw_rt_error(Args&&... args) {
  std::stringstream ss;
  int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
  (void)(dummy);
  throw std::runtime_error(ss.str());
}

static inline std::string file_line_no(const char * file, int line_no) {
    return std::string(file) + ":" + std::to_string(line_no);
}

#define ASSERT(condition)                                                      \
  if (!(condition)) {                                                          \
    throw_rt_error("assert", #condition, "failed at", file_line_no(__FILE__, __LINE__), \
                   "  ");                                                      \
  }

// pool:
//      cache all small size (< 1MB) buffers for future reallocation
//      big buffers are not cached to avoid waste of memory (since
//      big buffer means time-consuming computation, allocation
//      overhead is relatively small)
//
// NN-topology often has repeated sub-graphs, this simple pool reuses buffers between adjacent layers
// and to avoid waste of memory, pipeline is supposed to clear the pool after each model inference
//
struct PoolAllocator {
    int policy = 0;

#if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    static sycl::queue& default_sycl_queue() {
        static thread_local sycl::queue q{sycl::property::queue::in_order()};
        static auto initialized = [](sycl::queue& q) {
            auto dev_name = q.get_device().get_info<sycl::info::device::name>();
            std::cout << "sycl device used for tensor :" << dev_name << std::endl;
            return true;
        }(q);
        return q;
    }
#endif

    void * _alloc(size_t size, size_t alignment = 64) {
#if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    if (policy == 1)
        return sycl::aligned_alloc_shared(alignment, size, default_sycl_queue());
#endif
    // fallback to normal alloc
#ifdef _WIN32
        return _aligned_malloc(size, alignment);
#else
        void* ptr;
        ::posix_memalign(&ptr, alignment, size);
        return ptr;
#endif
    }

    void _free(void * ptr) {
#if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    if (policy == 1) {
        sycl::free(ptr, PoolAllocator::default_sycl_queue());
        return;
    }
#endif
    // fallback to normal alloc
#ifdef _WIN32
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }

    std::multimap<size_t, void*> pool;
    std::thread::id this_id;
    int to_be_recycle = 0;
    size_t alloc_size_req = 0;
    size_t alloc_size_real = 0;
    size_t alloc_times_req = 0;
    size_t alloc_times_real = 0;

    PoolAllocator() {
        this_id = std::this_thread::get_id();
        std::cout << "PoolAllocator #" << this_id << std::endl;
    }

    ~PoolAllocator() {
        summary(1);
        clear();
    }

    void set_policy(int pol) {
        clear();
        policy = pol;
    }

    std::string summary(int verbose) {
        // print all buffers
        size_t total_sz = 0;
        size_t total_cnt = 0;
        std::map<size_t, int> hist;
        for (auto it = pool.begin(); it != pool.end(); ++it) {
            auto sz = it->first;
            total_sz += sz;
            total_cnt ++;
            if (hist.count(sz) == 0)
                hist[sz] = 1;
            else
                hist[sz]++;
        }
        std::stringstream ss;
        ss << "PoolAllocator #" << this_id << " : ";
        ss << " pool_buffers=" << total_cnt
           << " pool_size=" << total_sz/1024.0 << " KB"
           << " alloc_size=" << alloc_size_real*100/alloc_size_req << "%(" << alloc_size_real << "/" << alloc_size_req << ")"
           << " alloc_times=" << alloc_times_real*100/alloc_times_req << "%(" << alloc_times_real << "/" << alloc_times_req << ")"
           << " to_be_recycle=" << to_be_recycle;

        if (verbose > 0) {
            ss << " buffers=[";
            for (auto& entry : hist) {
                auto sz = entry.first;
                auto cnt = entry.second;
                if (cnt == 1)
                    ss << sz;
                else
                    ss << sz << "x" << cnt;
                ss << ", ";
            }
            ss << "]";
        }
        return ss.str();
    }

    void clear() {
        for (auto& entry : pool) {
            _free(entry.second);
        }
        pool.clear();
        alloc_times_req = 0;
        alloc_times_real = 0;
        alloc_size_req = 0;
        alloc_size_real = 0;
    }

    std::shared_ptr<void> alloc(size_t size, size_t alignment = 64) {
        alloc_times_req++;
        alloc_size_req += size;
        if (size >= 1024 * 1024) {
            alloc_times_real++;
            alloc_size_real += size;
            return std::shared_ptr<void>(_alloc(size), [this](void* p) { _free(p); });
        }

        auto it = pool.find(size);
        void* buff;
        if (it == pool.end()) {
            alloc_times_real++;
            alloc_size_real += size;
            buff = _alloc(size);
            //if (first_tid != this_id)
            //    std::cout << this_id << " alloc " << size << ", new buff=" << std::hex << buff << std::dec << std::endl;
        } else {
            buff = it->second;
            pool.erase(it);
            //if (first_tid != this_id)
            //    std::cout << this_id << " alloc " << size << ", reuse buff=" << std::hex << buff << std::dec << std::endl;
        }

        to_be_recycle++;
        return std::shared_ptr<void>(buff, [this, size](void* p) {
            std::thread::id this_id = std::this_thread::get_id();
            //if (first_tid != this_id)
            //    std::cout << this_id << " return " << size << ", buff=" << std::hex << p << std::dec << std::endl;
            to_be_recycle--;
            pool.insert({size, p});
        });
    }
};

struct tensor {
  // coordinates inside tensor
  struct indices {
    tensor& t;
    int64_t index[8] = {0};
    bool overflow = false;
    indices(tensor& t) : t(t) {}

    const int64_t& operator[](int64_t i) const { return index[i]; }
    int64_t& operator[](int64_t i) { return index[i]; }
    int rank() const { return t.rank(); }

    int64_t move_to(int64_t index_flatten_1d) {
      for (auto r = t.rank() - 1; r >= 0; r--) {
        auto sz = t.size(r);
        index[r] = index_flatten_1d % sz;
        index_flatten_1d /= sz;
      }
      return index_flatten_1d;
    }

    // can only move_forward
    indices& operator+=(int64_t delta) {
      int64_t carry = delta;
      int64_t offset = 0;
      for (auto r = t.rank() - 1; r >= 0; r--) {
        if (carry > 0) {
          auto cur = index[r] + carry;
          auto sz = t.size(r);
          carry = 0;
          while (cur >= sz) {
            cur -= sz;
            carry++;
          }
          index[r] = cur;
        }
        offset = index[r] * t.stride(r);
      }
      if (carry)
        overflow = true;
      return *this;
    }
  };

  static PoolAllocator& pool() {
    static thread_local PoolAllocator tpool;
    return tpool;
  }
  std::shared_ptr<void> pool_alloc(size_t size, size_t alignment = 64) {
    return pool().alloc(size, alignment);
  }

  indices get_indices(int64_t index_flatten_1d) {
    indices idx(*this);
    idx.move_to(index_flatten_1d);
    return idx;
  }

  std::shared_ptr<void> m_ptr;
  int64_t m_shape[8];
  int64_t m_strides[8];
  int64_t m_item_size = 0;
  int m_rank = 0;
  int64_t m_offset = 0;
  std::type_info* m_p_tinfo;
  const char* m_format;

  tensor() = default;

  tensor(const tensor& rhs) {
    m_ptr = rhs.m_ptr;
    m_item_size = rhs.m_item_size;
    m_rank = rhs.m_rank;
    m_offset = rhs.m_offset;
    m_p_tinfo = rhs.m_p_tinfo;
    m_format = rhs.m_format;
    memcpy(m_shape, rhs.m_shape, sizeof(m_shape));
    memcpy(m_strides, rhs.m_strides, sizeof(m_strides));
  }

  void _generate_dense_strides() {
    int64_t stride = 1;
    for (int i = m_rank - 1; i >= 0; i--) {
      m_strides[i] = stride;
      stride *= m_shape[i];
    }
  }

  template <typename I = int64_t>
  void reset(void* ptr,
             const std::type_info& tinfo,
             const std::vector<I>& dims,
             const std::vector<I>& bytes_strides = {}) {
    m_p_tinfo = const_cast<std::type_info*>(&tinfo);
    if (tinfo == typeid(float)) {
      m_item_size = sizeof(float);
      m_format = "f";
    } else if (tinfo == typeid(int)) {
      m_item_size = sizeof(int);
      m_format = "i";
    } else if (tinfo == typeid(uint8_t)) {
      m_item_size = sizeof(uint8_t);
      m_format = "B";
    } else if (tinfo == typeid(int8_t)) {
      m_item_size = sizeof(int8_t);
      m_format = "b";
    } else if (tinfo == typeid(short)) {
      m_item_size = sizeof(short);
      m_format = "h";
    } else if (tinfo == typeid(int64_t)) {
      m_item_size = sizeof(int64_t);
      m_format = "q";
    } else if (tinfo == typeid(long)) {
      m_item_size = sizeof(long);
      m_format = "l";
    } else {
      throw_rt_error("Unsupported type_info : ", tinfo.name());
      return;
    }

    m_rank = 0;
    int64_t total_cnt = 1;
    for (auto it = dims.begin(); it != dims.end(); ++it) {
      m_shape[m_rank] = *it;
      total_cnt *= (*it);
      m_rank++;
    }
    if (total_cnt == 0) {
      m_ptr.reset();
      return;
    }

    if (bytes_strides.empty()) {
      _generate_dense_strides();
    } else {
      auto* p_stride = &m_strides[0];
      for (auto it = bytes_strides.begin(); it != bytes_strides.end(); ++it) {
        *p_stride++ = (*it) / m_item_size;
      }
    }

    if (ptr) {
      m_ptr = std::shared_ptr<void>(ptr, [](void*) {});
    } else {
      auto capacity_new = total_cnt * m_item_size;
      m_ptr = pool_alloc(capacity_new);
    }
  }

  template <typename T, typename I = int64_t>
  void reset(T* ptr,
             const std::vector<I>& dims,
             const std::vector<I>& bytes_strides = {}) {
    reset(ptr, typeid(T), dims, bytes_strides);
  }

  tensor permute(const std::vector<int>& order) const {
    if (order.size() != m_rank)
      throw_rt_error("permute with inconsistent number of order.");

    tensor newtv(*this);
    uint32_t hit_mask = 0;
    for (int i = 0; i < m_rank; i++) {
      auto j = order[i];
      if (j < 0 || j > m_rank)
        throw_rt_error("permute order ", j, " out of range [0,", m_rank, ")");
      if (hit_mask & (1 << j))
        throw_rt_error("permute order has duplicate item: ", j);
      hit_mask |= (1 << j);
      newtv.m_shape[i] = m_shape[j];
      newtv.m_strides[i] = m_strides[j];
    }
    return newtv;
  }

  int dense_rank() const {
    int d_rank = 0;
    size_t stride = 1;
    for (int i = m_rank - 1; i >= 0; i--) {
      if (m_strides[i] != stride)
        break;
      stride *= m_shape[i];
      d_rank++;
    }
    return d_rank;
  }

  bool is_dense() const { return dense_rank() == m_rank; }

  // reshape current tensor so it is broadcast-able to t
  tensor reshape_like(const tensor& t) const {
    ASSERT(rank() <= t.rank());
    std::vector<size_t> new_shape(t.rank(), 1);
    for (int i = t.rank() - 1, j = rank() - 1; i >= 0; j--, i--) {
      auto dst_dim = t.size(i);
      auto src_dim = (j < 0) ? 1 : size(j);
      if (src_dim == 1 || src_dim == dst_dim) {
        // broadcast-able
        new_shape[i] = src_dim;
      } else {
        throw_rt_error("reshape_like failed : src_dim", src_dim,
                       "cannot broadcast to", dst_dim);
      }
    }
    return reshape(new_shape);
  }

  template <typename I = int64_t>
  tensor reshape(const std::vector<I>& target_shape) const {
    if (!is_dense())
      throw_rt_error("tensor reshape only support dense layout.");

    tensor newtv(*this);
    int64_t total_cnt = 1;
    for (auto s : target_shape)
      total_cnt *= s;
    if (total_cnt != numel())
      throw_rt_error("tensor reshape to inconsistent element count");
    newtv.m_rank = target_shape.size();
    for (int i = 0; i < target_shape.size(); i++)
      newtv.m_shape[i] = target_shape[i];
    newtv._generate_dense_strides();  // since current tensor is dense, we can
                                      // regenerate strides
    return newtv;
  }

  tensor slice(int axis, int start, int end, int step = 1) const {
    tensor sub_tensor(*this);
    assert(axis < m_rank);

    if (end > start) {
      // change shape
      sub_tensor.m_shape[axis] = (end - start - 1) / step + 1;
    } else {
      // squeeze if end == start
      sub_tensor.m_rank = m_rank - 1;
      size_t k = 0;
      for (size_t i = 0; i < m_rank; i++) {
        if (i != static_cast<size_t>(axis)) {
          sub_tensor.m_strides[k] = m_strides[i];
          sub_tensor.m_shape[k] = m_shape[i];
          k++;
        }
      }
    }

    auto off = start * m_strides[axis];
    sub_tensor.m_offset = off + m_offset;
    return sub_tensor;
  }

  int64_t numel() const {
    int64_t sz = 1;
    for (int i = 0; i < m_rank; i++)
      sz *= m_shape[i];
    return sz;
  }
  int64_t size(int i) const {
    if (i < 0) {
      i += m_rank;
    }
    return m_shape[i];
  }
  int64_t stride(int i) const { return m_strides[i]; }
  int rank() const { return m_rank; }
  template <typename T = void>
  T* data() const {
    return reinterpret_cast<T*>(reinterpret_cast<int8_t*>(m_ptr.get()) +
                                m_offset * m_item_size);
  }
  int64_t item_size() const { return m_item_size; }
  int64_t byte_size() const { return m_item_size * numel(); }
  std::type_info* tinfo() const { return m_p_tinfo; }

  template <typename T>
  bool is(int rank = -1) const {
    if (m_p_tinfo != &typeid(T))
      return false;
    if (rank >= 0 && rank != m_rank)
      return false;
    return true;
  }

  template <typename I = int64_t>
  bool is(std::initializer_list<I> shape) const {
    int i = 0;
    auto it = shape.begin();
    for (; it != shape.end() && i < m_rank; ++it, ++i) {
      if ((*it) != m_shape[i])
        return false;
    }
    if (it != shape.end() || i < m_rank)
      return false;
    return true;
  }

  template <typename ST = int64_t>
  std::vector<ST> shape() const {
    return std::vector<ST>(m_shape, m_shape + m_rank);
  }
  template <typename ST = int64_t>
  std::vector<ST> strides() const {
    return std::vector<ST>(m_strides, m_strides + m_rank);
  }
  template <typename ST = int64_t>
  std::vector<ST> byte_strides() const {
    std::vector<ST> bstrides(m_rank);
    for (int i = 0; i < m_rank; i++) {
      bstrides[i] = m_strides[i] * m_item_size;
    }
    return bstrides;
  }

  template <typename T>
  tensor& operator=(T v) {
    ASSERT(is_dense());
    auto* ptr = data<T>();
    auto n = numel();
    for (int64_t i = 0; i < n; i++)
      ptr[i] = v;
    return *this;
  }

  template <typename T>
  T& operator()(std::initializer_list<int64_t> index) {
    return at<T>(index);
  }

  // when allow_broadcast is true, index to size-1 dim will always access 0.
  template <typename T>
  T& at(const std::initializer_list<int64_t>& index,
        bool allow_broadcast = false) const {
    size_t off = m_offset;
    auto it = index.begin();
    for (size_t i = 0; i < m_rank; i++) {
      size_t coordinate = (it != index.end()) ? (*it++) : 0;
      if (allow_broadcast && m_shape[i] == 1) {
        // allow_broadcast only works when the dimension is really 1
        coordinate = 0;
      } else {
        assert(coordinate < m_shape[i]);
      }
      off += m_strides[i] * coordinate;
    }

    return *reinterpret_cast<T*>(reinterpret_cast<int8_t*>(m_ptr.get()) +
                                 off * m_item_size);
  }
  template <typename T>
  T& at(const indices& idx, bool allow_broadcast = false) const {
    size_t off = m_offset;
    for (int64_t i = m_rank - 1, j = idx.rank() - 1; i >= 0; i--, j--) {
      size_t coordinate = (j >= 0) ? idx[j] : 0;
      if (allow_broadcast && m_shape[i] == 1) {
        // allow_broadcast only works when the dimension is really 1
        coordinate = 0;
      } else {
        assert(coordinate < m_shape[i]);
      }
      off += m_strides[i] * coordinate;
    }
    return *reinterpret_cast<T*>(reinterpret_cast<int8_t*>(m_ptr.get()) +
                                 off * m_item_size);
  }

  template <typename T>
  void print_subtensor(std::stringstream& os,
                       const T* ptr,
                       const int64_t* shape,
                       const int64_t* strides,
                       int rank,
                       int depth) const {
    if (rank == 1) {
      std::stringstream ss;
      os << "[";
      const int maxsize = 120;
      const char* sep = "";
      auto n = shape[0];
      auto stride = strides[0];
      for (int64_t i = 0; i < n; i++) {
        ss << sep;
        sep = ",";
        if (std::is_integral<T>::value && sizeof(T) < 8) {
          ss << static_cast<int64_t>(ptr[i * stride]);
        } else {
          ss << std::setw(14) << ptr[i * stride];
        }
        if (ss.tellp() > maxsize) {
          ss << "...";
          break;
        }
      }
      os << ss.str() << "]";
      return;
    }
    os << "[";
    for (int64_t i = 0; i < shape[0]; i++) {
      print_subtensor(os, ptr, shape + 1, strides + 1, rank - 1, depth + 1);
      ptr += strides[0];
      if (i + 1 < shape[0]) {
        os << "\n";
        for (int r = 0; r < depth + 1; r++)
          os << " ";
      }
    }
    os << "]";
  }

  std::string repr(bool with_values = true) const {
    if (!m_ptr) {
      return "tensor{empty}";
    }
    std::stringstream ss;
    ss << "tensor " << m_p_tinfo->name() << "(" << m_item_size << ") shape=[";
    const char* sep = "";
    for (size_t i = 0; i < m_rank; i++) {
      ss << sep << m_shape[i];
      sep = ",";
    }
    ss << "] strides=[";
    sep = "";
    for (size_t i = 0; i < m_rank; i++) {
      ss << sep << m_strides[i];
      sep = ",";
    }
    ss << "]";

    if (with_values) {
      ss << "\n";
      if (m_p_tinfo == &typeid(float))
        print_subtensor<float>(ss, data<float>(), &m_shape[0], &m_strides[0],
                               m_rank, 0);
      if (m_p_tinfo == &typeid(int8_t))
        print_subtensor<int8_t>(ss, data<int8_t>(), &m_shape[0], &m_strides[0],
                                m_rank, 0);

      if (m_p_tinfo == &typeid(long))
        print_subtensor<long>(ss, data<long>(), &m_shape[0], &m_strides[0],
                              m_rank, 0);
    }
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const tensor& T);
};

std::ostream& operator<<(std::ostream& os, const tensor& T) {
  os << T.repr();
  return os;
}
