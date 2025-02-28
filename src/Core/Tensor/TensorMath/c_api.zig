//!
//!
//!
//!
//!
//!  JUST FOR TESTING! DELETE THIS FILE AFTER 5/03/25
//!
//!
//!
//!
//!
const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator;
const lib = @import("lib.zig");

var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
    lib.setLogFunction(func);
}

export fn matmul_and_info() void {
    if (lib.matmul_and_info()) {} else |_| {}
}

export fn get_last_result_size() usize {
    return lib.last_result_size;
}

export fn get_last_result(index: usize) f32 {
    if (index < lib.last_result_size) {
        return lib.last_result[index];
    }
    return 0;
}

export fn predict(input: ?*anyopaque, output: ?*anyopaque) i16 {
    if (log_function) |log| {
        log(@constCast(@ptrCast("Predicting...\n")));
        if (input != null) {
            log(@constCast(@ptrCast("Input provided\n")));
        }
        if (output != null) {
            log(@constCast(@ptrCast("Output buffer provided\n")));
        }
    }
    return 69;
}
