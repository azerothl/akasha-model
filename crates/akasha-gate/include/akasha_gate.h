/* akasha-gate C ABI (Path A evaluate only — no tool execution).
 *
 * Build: cargo build -p akasha-gate --release
 * Artifact: target/release/libakasha_gate.so|.dylib|.dll
 *
 * Threading: re-entrant; no process-wide mutable gate state.
 * Free strings from akasha_gate_evaluate_json only with akasha_gate_string_free.
 */

#ifndef AKASHA_GATE_H
#define AKASHA_GATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Static ABI version string (e.g. "1"). Do not free. */
const char *akasha_gate_abi_version(void);

/** Wire CONTRACT_VERSION (currently 1). */
uint32_t akasha_gate_contract_version(void);

/**
 * Evaluate Path A from UTF-8 JSON (tools + proposal + signals).
 * Returns heap-allocated UTF-8 JSON; free with akasha_gate_string_free.
 * Failures still return JSON with "ok": false (unless allocation fails → NULL).
 */
char *akasha_gate_evaluate_json(const char *request_json);

/** Free a string returned by akasha_gate_evaluate_json. No-op on NULL. */
void akasha_gate_string_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* AKASHA_GATE_H */
