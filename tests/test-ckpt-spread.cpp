// server checkpoint eviction: the spread policy keeps points spread back from the newest (tools/server/server-ckpt-spread.h)
#include "../tools/server/server-ckpt-spread.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;
static void check(bool ok, const char * what) {
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    n_fail += ok ? 0 : 1;
}

// a conversation of n_turns turns of `turn` tokens after a root of `root` tokens, one checkpoint per turn start, at most
// `cap` kept; returns the tokens a request whose prompt differs `back` tokens before the end re-reads
static int64_t reread_after_edit(bool spread, int64_t root, int64_t turn, int n_turns, size_t cap, int64_t back) {
    std::vector<int64_t> cks = {root};
    int64_t end = root;
    for (int t = 1; t <= n_turns; ++t) {
        const int64_t start = root + (int64_t) (t - 1) * turn + (t == 1 ? 0 : 0);
        end = root + (int64_t) t * turn;
        if (t > 1) {
            while (cks.size() >= cap) {
                int v = -1;
                if (spread) {
                    v = server_ckpt_spread_victim(cks, 0, start);
                }
                if (v < 0) {
                    v = 1;   // oldest after the root (the plain policy)
                }
                cks.erase(cks.begin() + v);
            }
            cks.push_back(start);
        }
    }
    const int64_t edit = end - back;
    int64_t reuse = 0;
    for (int64_t c : cks) {
        if (c <= edit) {
            reuse = std::max(reuse, c);
        }
    }
    return end - reuse;
}

int main() {
    std::printf("test-ckpt-spread\n");
    // 200 turns of 1000 tokens after a 9525-token root, 8 checkpoints
    const int64_t lru6 = reread_after_edit(false, 9525, 1000, 200, 8, 10500);
    const int64_t sp6  = reread_after_edit(true,  9525, 1000, 200, 8, 10500);
    std::printf("  edit 10.5K back: oldest-first re-reads %lld, spread %lld\n", (long long) lru6, (long long) sp6);
    check(lru6 > 150000, "oldest-first: an edit past the last 7 turns falls through to the root");
    check(sp6 < 4 * 10500, "spread: an edit 10.5K back re-reads less than 42K");
    check(reread_after_edit(true, 9525, 1000, 200, 8, 42000) < 4 * 42000, "spread: an edit 42K back re-reads less than 168K");
    check(reread_after_edit(true, 9525, 1000, 200, 8, 500) <= 1500, "spread: the turn before is kept");

    // never the root, never the keep index
    {
        const std::vector<int64_t> v = {100, 200, 300, 400};
        const int r = server_ckpt_spread_victim(v, -1, 500);
        check(r != 0, "the root is never the victim");
        check(server_ckpt_spread_victim(v, 1, 500) != 1, "the keep index is never the victim");
        check(server_ckpt_spread_victim({100}, -1, 200) == -1, "one checkpoint: no candidate");
        check(server_ckpt_spread_victim({100, 200}, 1, 300) == -1, "two checkpoints, the second kept: no candidate");
        // unsorted input: the oldest (index 2, 50 tokens) stays
        const std::vector<int64_t> u = {400, 200, 50, 300};
        check(server_ckpt_spread_victim(u, -1, 500) != 2, "unsorted input: the oldest is never the victim");
    }
    return n_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
