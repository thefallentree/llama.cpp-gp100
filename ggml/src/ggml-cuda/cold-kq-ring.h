#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

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
