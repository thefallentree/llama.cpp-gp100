// DSpark cumulative-confidence cut (common_speculative_dspark_keep): the block ends where the running product of the
// draft confidences drops below p_cum; the floor p_min on each draft's own confidence still applies; both 0: no cut
#include "speculative.h"

#include <cstdio>
#include <vector>

static int n_fail = 0;
static void check(bool ok, const char * what) {
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    n_fail += ok ? 0 : 1;
}

int main() {
    // rows of a decode batch are n_embd apart: only the first float of a row is the confidence
    const size_t stride = 5;
    auto rows = [&](const std::vector<float> & c) {
        std::vector<float> v(c.size() * stride, -7.0f);
        for (size_t i = 0; i < c.size(); ++i) {
            v[i * stride] = c[i];
        }
        return v;
    };

    {
        const auto v = rows({0.9f, 0.9f, 0.9f, 0.9f, 0.9f});
        // products 0.9 0.81 0.729 0.656 0.590
        check(common_speculative_dspark_keep(v.data(), stride, 5, 0.0f, 0.0f) == 5, "no floor, no cut keeps the block");
        check(common_speculative_dspark_keep(v.data(), stride, 5, 0.0f, 0.5f) == 5, "product stays above 0.5 on 5 x 0.9");
        check(common_speculative_dspark_keep(v.data(), stride, 5, 0.0f, 0.6f) == 4, "product 0.590 < 0.6 cuts the fifth");
        check(common_speculative_dspark_keep(v.data(), stride, 5, 0.0f, 0.8f) == 2, "product 0.81 >= 0.8 keeps two, 0.729 cuts the third");
        check(common_speculative_dspark_keep(v.data(), stride, 5, 0.0f, 0.95f) == 0, "the first draft alone is below 0.95");
        check(common_speculative_dspark_keep(v.data(), stride, 3, 0.0f, 0.6f) == 3, "n bounds the block");
    }
    {
        // each draft passes a floor of 0.6 but the product does not hold up
        const auto v = rows({0.7f, 0.7f, 0.7f, 0.7f});
        // 0.7 0.49 0.343 0.240
        check(common_speculative_dspark_keep(v.data(), stride, 4, 0.6f, 0.0f) == 4, "the floor alone keeps all four");
        check(common_speculative_dspark_keep(v.data(), stride, 4, 0.6f, 0.5f) == 1, "the cut ends the block at 0.49 < 0.5");
    }
    {
        // a draft below the floor ends the block even when the product would allow it
        const auto v = rows({1.0f, 0.4f, 1.0f});
        check(common_speculative_dspark_keep(v.data(), stride, 3, 0.5f, 0.1f) == 1, "the floor wins over a loose cut");
        check(common_speculative_dspark_keep(v.data(), stride, 3, 0.0f, 0.1f) == 3, "a loose cut and no floor keep all");
        check(common_speculative_dspark_keep(v.data(), stride, 0, 0.5f, 0.5f) == 0, "an empty block");
    }
    {
        // the product is monotone: once cut, a later higher confidence does not bring the block back
        const auto v = rows({0.6f, 0.6f, 1.0f, 1.0f});
        check(common_speculative_dspark_keep(v.data(), stride, 4, 0.0f, 0.5f) == 1, "0.36 < 0.5 ends the block at the second draft");
    }

    std::printf("%s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
