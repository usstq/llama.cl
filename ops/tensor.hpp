#pragma once

#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <typeinfo>

#ifdef _WIN32
#include <cstdlib>
#endif

template <typename... Args>
static inline void throw_rt_error(Args&&... args) {
  std::stringstream ss;
  int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
  (void)(dummy);
  throw std::runtime_error(ss.str());
}

struct tensor {
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

  template <typename I>
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
#ifdef _WIN32
      m_ptr = std::shared_ptr<void>(_aligned_malloc(capacity_new, 64),
                                    [](void* p) { _aligned_free(p); });
#else
      ::posix_memalign(&ptr, 64, capacity_new) m_ptr =
          std::shared_ptr<void>(ptr, [](void* p) { ::free(m_ptr); });
#endif
    }
  }

  template <typename T, typename I>
  void reset(T* ptr,
             const std::vector<I>& dims,
             const std::vector<I>& bytes_strides = {}) {
    reset(ptr, typeid(T), dims, bytes_strides);
  }

  tensor clone() const {
    tensor newt;
    // allocate dense memory
    newt.reset(nullptr, *m_p_tinfo, shape());
    // copy data
    auto d_rank = dense_rank();
    if (d_rank == m_rank) {
      memcpy(newt.data(), data(), size() * m_item_size);
    } else {
      throw_rt_error("clone only supprt dense tensor for now.");
      // reduce dense rank into one
      // int64_t dense_block_size = m_item_size *((d_rank == 0) ? 1 :
      // m_strides[m_rank - 1 - d_rank]);
    }
    return newt;
  }

  tensor permute(const std::vector<size_t>& order) const {
    if (order.size() != m_rank)
      throw_rt_error("permute with inconsistent number of order.");

    tensor newtv(*this);
    uint32_t hit_mask = 0;
    for (size_t i = 0; i < m_rank; i++) {
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

  tensor reshape(const std::vector<size_t>& target_shape) const {
    if (!is_dense())
      throw_rt_error("tensor reshape only support dense layout.");

    tensor newtv(*this);
    int64_t total_cnt = 1;
    for (auto s : target_shape)
      total_cnt *= s;
    if (total_cnt != size())
      throw_rt_error("tensor reshape to inconsistent element count");
    newtv.m_rank = target_shape.size();
    for (int i = 0; i < target_shape.size(); i++)
      newtv.m_shape[i] = target_shape[i];
    newtv._generate_dense_strides();
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

  int64_t size() const {
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
  std::type_info* tinfo() const { return m_p_tinfo; }
  template <typename T>
  bool is() const {
    return m_p_tinfo == &typeid(T);
  }

  std::vector<int64_t> shape() const {
    return std::vector<int64_t>(m_shape, m_shape + m_rank);
  }
  std::vector<int64_t> strides() const {
    return std::vector<int64_t>(m_strides, m_strides + m_rank);
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

    return reinterpret_cast<T*>(m_ptr.get())[off];
  }

#if 0
    std::string repr(int max_total_lines = 16, int lines_per_row = 1) const {
        if (!m_ptr) {
            return "{empty}";
        }
        std::stringstream ss;
        ss << data_type_name<T>::value << " shape=[";
        const char* sep = "";
        size_t sz = 1;
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_shape[i];
            sz *= m_shape[i];
            sep = ",";
        }
        ss << "] strides=[";
        sep = "";
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << "] {";
        if (m_rank > 1)
            ss << "\n";
        auto last_dim_size = m_shape[m_rank - 1];
        int row_id = 0;
        int cur_row_lines_left = lines_per_row;
        size_t cur_line_elecnt = 0;
        size_t cur_row_elecnt = 0;
        size_t i;
        auto* p = reinterpret_cast<T*>(m_ptr);
        for (i = 0; i < sz && max_total_lines > 0; i++) {
            if ((i % last_dim_size) == 0) {
                ss << row_id << ":\t\t";
                row_id++;
                cur_row_lines_left = lines_per_row;
            }

            // display current element if we still have buget
            if (cur_row_lines_left > 0) {
                if (std::is_integral<T>::value)
                    ss << static_cast<int64_t>(p[i]) << ",";
                else
                    ss << p[i] << ",";
                cur_line_elecnt++;
                cur_row_elecnt++;
                if ((cur_line_elecnt % 16) == 15 || (cur_row_elecnt == last_dim_size)) {
                    max_total_lines--;
                    cur_row_lines_left--;
                    if (cur_row_lines_left == 0) {
                        if (cur_row_elecnt == last_dim_size)
                            ss << ",\n";
                        else
                            ss << "...\n";
                        cur_row_elecnt = 0;
                    } else {
                        ss << "\n\t\t";
                    }
                    cur_line_elecnt = 0;
                }
            }
        }
        if (i < sz) {
            ss << "... ... ... ... \n";
        }
        ss << "}";
        return ss.str();
    }
#endif

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

  std::string repr(int max_total_lines = 16, int lines_per_row = 1) const {
    if (!m_ptr) {
      return "tensor{empty}";
    }
    std::stringstream ss;
    ss << "tensor shape=[";
    const char* sep = "";
    size_t sz = 1;
    for (size_t i = 0; i < m_rank; i++) {
      ss << sep << m_shape[i];
      sz *= m_shape[i];
      sep = ",";
    }
    ss << "] strides=[";
    sep = "";
    for (size_t i = 0; i < m_rank; i++) {
      ss << sep << m_strides[i];
      sep = ",";
    }
    ss << "] " << m_p_tinfo->name() << " item_size=" << m_item_size << "\n";
    if (m_p_tinfo == &typeid(float))
      print_subtensor<float>(
          ss, reinterpret_cast<const float*>(m_ptr.get()) + m_offset,
          &m_shape[0], &m_strides[0], m_rank, 0);
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const tensor& T);
};

std::ostream& operator<<(std::ostream& os, const tensor& T) {
  os << T.repr();
  return os;
}
