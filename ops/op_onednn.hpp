
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
    dnnl::stream strm;
    onednn_mm(tensor q, tensor kcache) {
        unsigned long B = q.size(0);
        unsigned long qL = q.size(1);
        ASSERT(kcache.size(-4) == B);
        unsigned long H = kcache.size(-3);
        unsigned long kvLen = kcache.size(-2);
        unsigned long S = kcache.size(-1);

        auto make_dnnl_dims = [](const std::vector<size_t>& dims) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl_dims;
        };
        auto default_dt = dnnl::memory::data_type::f32;
        q_md = dnnl::memory::desc(make_dnnl_dims({B, qL, H, S}), default_dt, dnnl::memory::format_tag::abcd)
                   .permute_axes({0, 2, 1, 3});
        k_md = dnnl::memory::desc(make_dnnl_dims({B, H, kvLen, S}), default_dt, dnnl::memory::format_tag::abcd)
                   .permute_axes({0, 1, 3, 2});
        attn_md = dnnl::memory::desc(make_dnnl_dims({B, H, qL, kvLen}), default_dt, dnnl::memory::format_tag::abcd);
        auto qk_pd = dnnl::matmul::primitive_desc(strm.get_engine(), q_md, k_md, attn_md);
        qk_prim = dnnl::matmul(qk_pd);
    }
    void exec(tensor& q, tensor& kcache, tensor& attn) {
        dnnl::memory query(q_md, strm.get_engine(), q.data<float>());
        dnnl::memory key(k_md, strm.get_engine(), kcache.data<float>());
        dnnl::memory attn_score(attn_md, strm.get_engine(), attn.data<float>());
        qk_prim.execute(strm, {{DNNL_ARG_SRC, query}, {DNNL_ARG_WEIGHTS, key}, {DNNL_ARG_DST, attn_score}});
    }
};

tensor onednn_qk(tensor q,      // [B, qL, H*S]
                 tensor kcache  // [B, H, kvLen, S]
) {
    static onednn_mm mm(q, kcache);
    auto B = q.size(0);
    auto qL = q.size(1);
    ASSERT(kcache.size(-4) == B);
    auto H = kcache.size(-3);
    auto kvLen = kcache.size(-2);
    auto S = kcache.size(-1);

    q = q.reshape({B, qL, H, S}).permute({0, 2, 1, 3});  // B,H,qL,S

    auto d_scale = 1.0f / std::sqrt(S);
    tensor attn_w;
    attn_w.reset(static_cast<float*>(nullptr), {B, H, qL, kvLen});
    mm.exec(q, kcache, attn_w);
    return attn_w;
}
