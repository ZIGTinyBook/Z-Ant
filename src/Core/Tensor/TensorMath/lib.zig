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

const zant = @import("../../../zant.zig");
const tensor_m = zant.core.tensor.math_standard;
pub usingnamespace tensor_m;

const std = @import("std");
const tensor = zant.core.tensor;
const Tensor = tensor.Tensor;
const pkg_allocator = zant.utils.allocator;

pub var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

pub fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
    tensor.setLogFunction(func);
}

pub fn matmul_and_info() !void {
    if (log_function) |log| {
        log(@constCast(@ptrCast("Starting matrix multiplication...\n")));
    }

    // Create two tensors for matrix multiplication
    var shape1 = [_]usize{ 2, 3 };
    var shape2 = [_]usize{ 3, 2 };

    if (log_function) |log| {
        log(@constCast(@ptrCast("Creating tensors...\n")));
    }

    var t1 = try Tensor(f32).fromShape(&pkg_allocator, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromShape(&pkg_allocator, &shape2);
    defer t2.deinit();

    if (log_function) |log| {
        log(@constCast(@ptrCast("Filling tensors with values...\n")));
    }

    // Fill tensors with some values
    for (0..t1.size) |i| {
        t1.data[i] = @floatFromInt(i + 1);
    }
    for (0..t2.size) |i| {
        t2.data[i] = @floatFromInt(i + 1);
    }

    if (log_function) |log| {
        log(@constCast(@ptrCast("Matrix 1:\n")));
        t1.info_metal();
        log(@constCast(@ptrCast("Matrix 2:\n")));
        t2.info_metal();
        log(@constCast(@ptrCast("Performing matrix multiplication...\n")));
    }

    // Perform matrix multiplication
    var result = try tensor_m.dot_product_tensor(f32, f32, &t1, &t2);
    defer result.deinit();

    if (log_function) |log| {
        log(@constCast(@ptrCast("Storing result...\n")));
    }

    // Store the result in a global variable that can be accessed from C
    for (0..result.size) |i| {
        last_result[i] = result.data[i];
    }
    last_result_size = result.size;

    if (log_function) |log| {
        log(@constCast(@ptrCast("Result matrix:\n")));
        result.info_metal();
        log(@constCast(@ptrCast("Matrix multiplication completed.\n")));
    }
}

// Global variables to store the last result
pub var last_result: [100]f32 = undefined;
pub var last_result_size: usize = 0;
