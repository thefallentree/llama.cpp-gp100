// speculative sampling cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

// a draft token from a distribution q is accepted with probability min(1, p/q), else a token is drawn from norm(max(0, p - q))
// the first token the verifier emits must follow p whatever q is, and the accepted share is sum_x min(p(x), q(x))
// q has mass outside the support of p
struct case_spec_reject : ctx_test {
    const char * name() const override { return "reject"; }

    const char * description() const override {
        return "rejection sampling of a draft keeps the target distribution and accepts sum min(p, q)";
    }

    std::string skip_reason(model_run & mr) const override {
        if (!mr.has_vocab()) {
            return "no vocab";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        ctx_opts opts;
        opts.n_ctx     = 64;
        opts.n_batch   = 64;
        opts.n_ubatch  = 64;
        opts.n_seq_max = 1;

        llama_context_ptr ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_tokens prompt;
        for (llama_pos pos = 0; pos < 16; ++pos) {
            prompt.push_back(gen_token(n_vocab, 0, pos));
        }
        CTX_REQUIRE(decode_seq(ctx.get(), prompt, 0, (llama_pos) prompt.size(), 0, true), "the prompt decode failed");

        common_params_sampling sparams;
        sparams.seed     = 1234;
        sparams.top_k    = 8;
        sparams.temp     = 0.8f;
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };

        common_sampler_ptr smpl(common_sampler_init(mr.model, sparams));
        CTX_REQUIRE(smpl != nullptr, "the sampler failed to init");

        // the target distribution p: what the sampler leaves in its candidate array
        common_sampler_sample(smpl.get(), ctx.get(), -1, true);
        const llama_token_data_array * cur_p = common_sampler_get_candidates(smpl.get(), true);

        std::map<llama_token, double> p;
        double p_sum = 0.0;
        for (size_t k = 0; k < cur_p->size; ++k) {
            p[cur_p->data[k].id] = cur_p->data[k].p;
            p_sum += cur_p->data[k].p;
        }
        CTX_REQUIRE(p.size() >= 4 && std::fabs(p_sum - 1.0) < 1e-3, "the target distribution has " + std::to_string(p.size()) + " candidates, sum " + std::to_string(p_sum));

        // the draft distribution q: three tokens of p (not its heaviest) and two that p does not have
        std::vector<llama_token_data> q;
        {
            std::vector<llama_token> in_p;
            for (size_t k = 0; k < cur_p->size; ++k) {
                in_p.push_back(cur_p->data[k].id);
            }
            const float w[5] = { 0.30f, 0.25f, 0.20f, 0.15f, 0.10f };
            const llama_token ids[3] = { in_p[1], in_p[3], in_p[2] };
            for (int i = 0; i < 3; ++i) {
                q.push_back({ ids[i], 0.0f, w[i] });
            }
            int n_out = 0;
            for (llama_token t = 0; t < n_vocab && n_out < 2; ++t) {
                if (p.count(t) == 0) {
                    q.push_back({ t, 0.0f, w[3 + n_out] });
                    n_out++;
                }
            }
            CTX_REQUIRE(n_out == 2, "no token outside the candidates of p");
        }

        double accept_expected = 0.0;
        for (const auto & e : q) {
            const auto it = p.find(e.id);
            accept_expected += std::min<double>(e.p, it != p.end() ? it->second : 0.0);
        }

        constexpr int n_trials = 20000;

        std::mt19937 rng(7);
        std::discrete_distribution<int> draw({ q[0].p, q[1].p, q[2].p, q[3].p, q[4].p });

        std::map<llama_token, int> counts;
        int n_accept = 0;
        for (int t = 0; t < n_trials; ++t) {
            const llama_tokens draft = { q[draw(rng)].id };

            const std::vector<llama_token> res = common_sampler_sample_and_accept_n_rejection(smpl.get(), ctx.get(), { -1, -1 }, draft, { q });

            CTX_REQUIRE(!res.empty() && res.size() <= 2, "the verifier returned " + std::to_string(res.size()) + " tokens");
            counts[res[0]]++;
            n_accept += res.size() == 2 ? 1 : 0;

            // an accepted draft is the first token, a rejected one leaves a token of the residual
            CTX_REQUIRE(res.size() == 1 || res[0] == draft[0], "an accepted draft changed");
        }

        // the first token against p: chi-square over its support, nothing outside it
        double chi2 = 0.0;
        for (const auto & e : p) {
            const double exp = n_trials*e.second;
            const double obs = counts.count(e.first) ? counts[e.first] : 0;
            chi2 += (obs - exp)*(obs - exp)/exp;
        }
        for (const auto & e : counts) {
            CTX_REQUIRE(p.count(e.first) > 0, "token " + std::to_string(e.first) + " is outside the support of p, " + std::to_string(e.second) + " times");
        }
        const double dof = (double) p.size() - 1.0;
        CTX_REQUIRE(chi2 < dof + 6.0*std::sqrt(2.0*dof),
                "the first token does not follow p: chi2 " + std::to_string(chi2) + " over " + std::to_string((int) dof) + " degrees of freedom");

        const double accept_rate = (double) n_accept/n_trials;
        const double sigma       = std::sqrt(accept_expected*(1.0 - accept_expected)/n_trials);
        CTX_REQUIRE(std::fabs(accept_rate - accept_expected) < 5.0*sigma,
                "accepted share " + std::to_string(accept_rate) + " against sum min(p, q) " + std::to_string(accept_expected));

        TLOG_INF("%s: %d trials, chi2 %.2f over %d dof, accepted %.4f (sum min(p, q) %.4f)\n", __func__, n_trials, chi2, (int) dof, accept_rate, accept_expected);

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_spec_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_spec_reject>());

    return cases;
}
