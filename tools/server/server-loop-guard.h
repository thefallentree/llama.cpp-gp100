#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// Detects a degenerate repetition in a generated token stream: the last tokens are periodic with a period p of at
// most period_max, over at least `span` tokens and at least `reps` repetitions. run[p] counts the consecutive latest
// tokens equal to the token p positions before them; the periodic stretch is run[p] + p tokens long.
struct server_loop_guard {
    int32_t span       = 512;
    int32_t reps       = 8;
    int32_t period_max = 512;

    std::vector<llama_token> ring; // the last period_max tokens
    std::vector<int32_t>     run;  // run[p], p = 1..period_max
    uint64_t                 n = 0;

    void init(int32_t span_, int32_t reps_, int32_t period_max_) {
        span       = span_;
        reps       = reps_;
        period_max = period_max_;
        ring.assign(period_max, LLAMA_TOKEN_NULL);
        run.assign(period_max + 1, 0);
        n = 0;
    }

    // forget the history (a phase change: the forced close breaks the period anyway)
    void reset() {
        std::fill(run.begin(), run.end(), 0);
        n = 0;
    }

    // feeds one token; returns the smallest period of a loop that ends at it, 0 = none
    int32_t push(llama_token t) {
        if (period_max <= 0) {
            return 0;
        }
        int32_t found = 0;
        for (int32_t p = 1; p <= period_max; ++p) {
            if (n >= (uint64_t) p && ring[(n - p) % period_max] == t) {
                run[p]++;
                if (found == 0 && run[p] >= span && run[p] >= (int64_t) (reps - 1) * p) {
                    found = p;
                }
            } else {
                run[p] = 0;
            }
        }
        ring[n % period_max] = t;
        n++;
        return found;
    }
};
