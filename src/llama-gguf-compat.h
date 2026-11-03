#pragma once

#include <string>
#include <unordered_map>

struct gguf_context;

// GGUF written by another runtime for an architecture we already have, under another layout (a "dialect")
// the loader calls llama_gguf_compat_apply() after it reads the header: the metadata is rewritten in place to our layout, the tensors that go by another name are listed
// known dialects: "deepseek41-dflash-draft" (Lucebox DSpark drafts for DeepSeek-V4.1-Flash) -> "dflash"
struct llama_gguf_compat {
    std::string dialect; // arch string of the file, empty for a native file

    std::unordered_map<std::string, std::string> tensor_rename; // name in the file -> name the loader asks for
};

// returns true if a dialect was applied, throws if the file has something the mapping does not cover
bool llama_gguf_compat_apply(struct gguf_context * meta, llama_gguf_compat & compat);
