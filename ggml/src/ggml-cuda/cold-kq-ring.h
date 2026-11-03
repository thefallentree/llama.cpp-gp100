#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// The staging buffer's size against the card (all sizes in bytes; a CPU test runs it): how many slots an op of n cold experts may take,
// and whether the buffer that is there goes back to the card. The buffer counts as free (it is reused or given back); the margin
// `keep` (plus the caller's `reserve`) stays free; budget is the cap of GGML_CUDA_COLD_PREFETCH_MB, at least two slots of it.
struct cold_kq_vram_in {
    size_t n, nb02, budget, reserve, keep;
    bool   have_mem;   // free_b is known (else: no limit from the card)
    size_t free_b, cap;
};

struct cold_kq_vram_out {
    size_t fit_slots;  // the most slots the op may take
    bool   refuse;     // not even two (or the op's n < 2) fit: the cold experts are read in place
    bool   cut;        // fewer than the op and the budget would take: the card is the limit
    bool   release;    // the buffer goes back to the card (sized again by the op, or not at all when refused)
};

static inline cold_kq_vram_out cold_kq_vram(const cold_kq_vram_in & in) {
    cold_kq_vram_out out = {};
    const size_t margin     = in.keep + in.reserve;
    const size_t usable     = !in.have_mem ? SIZE_MAX : in.free_b + in.cap > margin ? in.free_b + in.cap - margin : 0;
    const size_t vram_slots = usable == SIZE_MAX ? SIZE_MAX : usable/in.nb02;
    const size_t budget_slots = std::max<size_t>(2, in.budget > in.reserve ? (in.budget - in.reserve)/in.nb02 : 0);
    out.fit_slots = std::min(budget_slots, vram_slots);
    out.refuse    = out.fit_slots < std::min<size_t>(in.n, 2);
    out.cut       = !out.refuse && out.fit_slots < std::min(in.n, budget_slots);
    // the card is short and the buffer holds more than it can spare (the pool grew since): back to the card, sized again by the op
    out.release   = in.cap > 0 && (out.refuse || (in.have_mem && in.free_b < margin && in.cap > out.fit_slots*in.nb02 + in.cap/8));
    return out;
}

// Schedule of the cold-expert staging of cold-kq.cuh, without any CUDA type (a CPU test runs it).
// The experts of an op come in chunks of `chunk` experts, chunk k goes to buffer k % depth of a ring of `depth`
// buffers, and the consumer takes the experts in list order. Dev does the stream work:
//   copy(slot, expert, n)   n adjacent experts into the staging slots slot ... slot + n - 1 (copy stream)
//   copy_wait_freed(b)      the copy stream waits until the consumer is done with buffer b
//   copy_record_ready(b)    the copy stream marks the copies into buffer b done
//   wait_ready(b)           the consumer stream waits for them
//   record_freed(b)         the consumer stream marks its reads of buffer b as enqueued
struct cold_kq_ring {
    std::vector<int32_t> list; // the experts of the op (index in the cold tensor), ascending
    int chunk   = 1;
    int depth   = 1;
    int nchunks = 0;
    int next    = 0;           // the next entry of list to take

    // slots: how many experts the staging buffer holds
    void plan(size_t slots) {
        const size_t n = list.size();
        slots   = std::max<size_t>(1, std::min(slots, n));
        chunk   = (int) std::max<size_t>(std::min<size_t>(8, std::max<size_t>(1, slots/4)), (slots + 63)/64);
        depth   = (int) std::max<size_t>(1, slots/chunk);
        nchunks = (int) ((n + chunk - 1)/chunk);
        next    = 0;
    }

    // the staging slots the plan uses: slot numbers are below it (depth * chunk <= the slots given to plan)
    size_t n_slots() const {
        return (size_t) depth*chunk;
    }

    template <class Dev> void issue(Dev & dev, const int k) const {
        if (k >= nchunks) {
            return;
        }
        const int b     = k % depth;
        const int first = k*chunk;
        const int last  = std::min<int>(first + chunk, (int) list.size());
        if (k >= depth) {
            dev.copy_wait_freed(b);
        }
        for (int j = first; j < last;) {
            int e = j + 1;
            while (e < last && list[e] == list[e - 1] + 1) {
                e++;
            }
            dev.copy(b*chunk + (j - first), list[j], e - j);
            j = e;
        }
        dev.copy_record_ready(b);
    }

    template <class Dev> void start(Dev & dev) const {
        for (int k = 0; k < std::min(depth, nchunks); ++k) {
            issue(dev, k);
        }
    }

    // the staging slot of the next expert; the caller enqueues the work that reads it before the next call
    template <class Dev> int take(Dev & dev) {
        const int j = next++;
        const int k = j/chunk;
        if (j % chunk == 0) {
            if (k > 0) {
                dev.record_freed((k - 1) % depth);
                issue(dev, k - 1 + depth);
            }
            dev.wait_ready(k % depth);
        }
        return (k % depth)*chunk + j % chunk;
    }
};
