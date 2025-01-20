const std = @import("std");

pub fn writeAllocator(writer: std.fs.File.Writer, name: []const u8, size: usize) !void {
    try writer.print(
        \\const {0s}_static = struct {{ var buffer: [{1}]u8 = undefined; }};
        \\var {0s}_fba = std.heap.FixedBufferAllocator.init(&{0s}_static.buffer);
        \\const {0s} = {0s}_fba.allocator();
    , .{ name, size });
}
