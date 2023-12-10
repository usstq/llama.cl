
#pragma once

#include <stdint.h>

#include "oneapi/dnnl/dnnl.hpp"
#include "tensor.hpp"
#include "utils.hpp"

struct onednn_mm {
    dnnl::memory::desc q_md;
    dnnl::memory::desc k_md;
    dnnl::memory::desc attn_md;
    dnnl::matmul qk_prim;
    dnnl::engine engine;
    dnnl::stream strm;

    onednn_mm() : engine(dnnl::engine::kind::cpu, 0), strm(engine) {}

    unsigned long m_B = 0;
    unsigned long m_H = 0;
    unsigned long m_S = 0;
    unsigned long m_qL = 0;
    unsigned long m_kvLen = 0;

    void update(tensor& q, tensor& kcache) {
        unsigned long B = q.size(0);
        unsigned long qL = q.size(1);
        ASSERT(kcache.size(-4) == B);
        unsigned long H = kcache.size(-3);
        unsigned long kvLen = kcache.size(-2);
        unsigned long S = kcache.size(-1);

        if (m_B == B && m_H == H && m_S == S && m_qL == qL && m_kvLen == kvLen)
            return;
        m_B = B;
        m_H = H;
        m_S = S;
        m_qL = qL;
        m_kvLen = kvLen;
        auto make_dnnl_md = [](const std::vector<size_t>& dims,
                               dnnl::memory::data_type dtype = dnnl::memory::data_type::f32) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl::memory::desc(dnnl_dims, dtype, dnnl::memory::format_tag::abcd);
        };

        q_md = make_dnnl_md({B, qL, H, S}).permute_axes({0, 2, 1, 3});
        k_md = make_dnnl_md({B, H, kvLen, S}).permute_axes({0, 1, 3, 2});
        //q_md = make_dnnl_md({B, H, qL, S});
        //k_md = make_dnnl_md({B, H, S, kvLen});
        attn_md = make_dnnl_md({B, H, qL, kvLen});
        dnnl::primitive_attr attr;
        dnnl::post_ops po;
        float d_scale = 1.0f / std::sqrt(S);
        po.append_eltwise(dnnl::algorithm::eltwise_linear, d_scale, 0.f);        
        attr.set_post_ops(po);
        auto qk_pd = dnnl::matmul::primitive_desc(strm.get_engine(), q_md, k_md, attn_md, attr);
        qk_prim = dnnl::matmul(qk_pd);
        //std::cout << "onednn_mm B=" << B << ", H=" << H << ", S=" << S << ", qL=" << qL << ", kvLen=" << kvLen << std::endl;
    }

    void exec(tensor& q, tensor& k, tensor& attn) {
        dnnl::memory query(q_md, strm.get_engine(), q.data<float>());
        dnnl::memory key(k_md, strm.get_engine(), k.data<float>());
        dnnl::memory attn_score(attn_md, strm.get_engine(), attn.data<float>());

        qk_prim.execute(strm, {{DNNL_ARG_SRC, query}, {DNNL_ARG_WEIGHTS, key}, {DNNL_ARG_DST, attn_score}});
        strm.wait();
    }
};

tensor onednn_qk(tensor q,      // [B, qL, H*S]
                 tensor kcache  // [B, H, kvLen, S]
) {
    static onednn_mm mm;
    mm.update(q, kcache);

    auto B = q.size(0);
    auto qL = q.size(1);
    ASSERT(kcache.size(-4) == B);
    auto H = kcache.size(-3);
    auto kvLen = kcache.size(-2);
    auto S = kcache.size(-1);

    // q = q.reshape({B, qL, H, S}).permute({0, 2, 1, 3});  // B,H,qL,S

    tensor attn_w;
    attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});
    mm.exec(q, kcache, attn_w);
    return attn_w;
}
