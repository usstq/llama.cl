

template<typename T>
struct Ktensor {
    T * m_ptr;
    int64_t m_dims[8];
    int64_t m_strides[8];
    int m_rank;

    Ktensor(void* ptr, std::initializer_list<int64_t> dims) {
        m_rank = 0;
        for(auto it = dims.begin(); it != dims.end(); ++it) {
            m_dims[m_rank] = *it;
            m_rank++;
        }
        int64_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            m_strides[i] = stride;
            stride *= m_dims[i];
        }
        m_ptr = reinterpret_cast<T*>(ptr);
    }

    int64_t size(int i) {
        return m_dims[i];
    }
    int64_t stride(int i) {
        return m_strides[i];
    }
    T& operator()(std::initializer_list<int64_t> index) {
        return at(index);
    }
    // when allow_broadcast is true, index to size-1 dim will always access 0.
    T& at(const std::initializer_list<int64_t>& index, bool allow_broadcast = false) const {
        size_t off = 0;
        auto it = index.begin();
        for (size_t i = 0; i < m_rank; i++) {
            size_t coordinate = (it != index.end()) ? (*it++) : 0;
            if (allow_broadcast && m_dims[i] == 1) {
                // allow_broadcast only works when the dimension is really 1
                coordinate = 0;
            } else {
                assert(coordinate < m_dims[i]);
            }
            off += m_strides[i] * coordinate;
        }
        return m_ptr[off];
    }
};
