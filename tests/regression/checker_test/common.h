#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
    uint64_t src_addr;  // "hidden state" float array to monitor
    uint64_t dst_addr;  // output: sum of all src elements (for correctness check)
    uint32_t num_elems; // number of floats in src
} kernel_arg_t;

#endif
