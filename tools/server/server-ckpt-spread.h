#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

// Victim choice for a full per-slot context checkpoint list ("spread" policy, after Strata a7d4b754).
//
// Dropping the oldest checkpoint keeps the newest ones, and every turn adds one, so after a few short turns all of
// them sit within the last 1-2K tokens: a client that rewrites its history a few thousand tokens back then matches
// none of them and re-reads the conversation from the root. Here the victim is the checkpoint whose removal leaves
// the smallest gap for its distance from the newest, so the kept points thin out geometrically towards the root
// (an edit d tokens before the end re-reads about d tokens).
//
// n_tokens: the checkpoints' token counts (any order); keep: index never evicted (the restore point with the
// lowest pos_min), or -1; n_new: token count of the checkpoint about to be added (the newest, never evicted).
// The oldest checkpoint (the root) is never a candidate. Returns an index into n_tokens, or -1 without candidate
// (the caller then falls back to its plain policy).
static inline int server_ckpt_spread_victim(const std::vector<int64_t> & n_tokens, int keep, int64_t n_new) {
    const int m = (int) n_tokens.size();
    if (m < 2) {
        return -1;
    }

    std::vector<int> ord(m);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return n_tokens[a] < n_tokens[b]; });

    const int64_t newest = std::max(n_new, n_tokens[ord[m - 1]]);

    int    victim = -1;
    double best   = 0.0;
    for (int j = 1; j < m; ++j) {
        const int i = ord[j];
        if (i == keep) {
            continue;
        }
        const int64_t prev = n_tokens[ord[j - 1]];
        const int64_t next = j + 1 < m ? n_tokens[ord[j + 1]] : newest;
        const int64_t span = newest - prev;
        const double  gap  = span > 0 ? (double) (next - prev) / (double) span : 0.0;
        if (victim < 0 || gap < best) {
            victim = i;
            best   = gap;
        }
    }
    return victim;
}
