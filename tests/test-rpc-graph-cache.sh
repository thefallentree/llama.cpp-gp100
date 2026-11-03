#!/usr/bin/env bash
# usage: test-rpc-graph-cache.sh RPC_SERVER TEST_CLIENT
set -euo pipefail

server=$1
client=$2
port=$((40000 + ($$ + 5000) % 10000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

"$server" --device CPU --host 127.0.0.1 --port "$port" >"$test_dir/server.log" 2>&1 &
pid=$!
for _ in {1..600}; do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        exec 3>&-
        exec 3<&-
        break
    fi
    sleep 0.05
done

if ! "$client" "$endpoint"; then
    cat "$test_dir/server.log"
    exit 1
fi
if grep -q "not found" "$test_dir/server.log"; then
    cat "$test_dir/server.log"
    exit 1
fi
