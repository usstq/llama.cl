#pragma once

#include <torch/extension.h>

template <typename T>
struct Ktensor
{
    T *m_ptr;
    int64_t m_dims[8];
    int64_t m_strides[8];
    int m_rank;

    Ktensor(at::Tensor x)
    {
        auto rank = x.dim();
        if (rank == 1)
            reset(x.data_ptr(), {x.size(0)});
        else if (rank == 2)
            reset(x.data_ptr(), {x.size(0), x.size(1)});
        else if (rank == 3)
            reset(x.data_ptr(), {x.size(0), x.size(1), x.size(2)});
        else if (rank == 4)
            reset(x.data_ptr(), {x.size(0), x.size(1), x.size(2), x.size(3)});
        else if (rank == 5)
            reset(x.data_ptr(), {x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)});
        else
        {
            std::cerr << "Ktensor : Unexpected rank " << rank << std::endl;
            std::abort();
        }
    }

    Ktensor(void *ptr, std::initializer_list<int64_t> dims)
    {
        reset(ptr, dims);
    }

    void reset(void *ptr, std::initializer_list<int64_t> dims)
    {
        m_rank = 0;
        for (auto it = dims.begin(); it != dims.end(); ++it)
        {
            m_dims[m_rank] = *it;
            m_rank++;
        }
        int64_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--)
        {
            m_strides[i] = stride;
            stride *= m_dims[i];
        }
        m_ptr = reinterpret_cast<T *>(ptr);
    }

    int64_t size(int i)
    {
        return m_dims[i];
    }
    int64_t stride(int i)
    {
        return m_strides[i];
    }
    T &operator()(std::initializer_list<int64_t> index)
    {
        return at(index);
    }
    // when allow_broadcast is true, index to size-1 dim will always access 0.
    T &at(const std::initializer_list<int64_t> &index, bool allow_broadcast = false) const
    {
        size_t off = 0;
        auto it = index.begin();
        for (size_t i = 0; i < m_rank; i++)
        {
            size_t coordinate = (it != index.end()) ? (*it++) : 0;
            if (allow_broadcast && m_dims[i] == 1)
            {
                // allow_broadcast only works when the dimension is really 1
                coordinate = 0;
            }
            else
            {
                assert(coordinate < m_dims[i]);
            }
            off += m_strides[i] * coordinate;
        }
        return m_ptr[off];
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
            ss << sep << m_dims[i];
            sz *= m_dims[i];
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
        auto last_dim_size = m_dims[m_rank - 1];
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
    std::string repr(int max_total_lines = 16, int lines_per_row = 1) const
    {
        if (!m_ptr)
        {
            return "Ktensor{empty}";
        }
        std::stringstream ss;
        ss << "Ktensor shape=[";
        const char *sep = "";
        size_t sz = 1;
        for (size_t i = 0; i < m_rank; i++)
        {
            ss << sep << m_dims[i];
            sz *= m_dims[i];
            sep = ",";
        }
        ss << "] strides=[";
        sep = "";
        for (size_t i = 0; i < m_rank; i++)
        {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << "]";
        return ss.str();
    }
    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const Ktensor<U> &T);
};

template <typename U>
std::ostream &operator<<(std::ostream &os, const Ktensor<U> &T)
{
    os << T.repr();
    return os;
}
