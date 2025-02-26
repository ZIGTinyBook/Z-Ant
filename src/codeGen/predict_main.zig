const std = @import("std");
const zant = @import("zant");
const predict_lib = @import("predict_lib");
const Tensor = zant.tensor.Tensor;

pub fn main() !void {

    // input for the debug model
    //const shape: [2]usize = [_]usize{5};
    //const array: [5]f32 = [_]f32{ 1, 1, 1, 1, 1 };
    predict_lib.ciaoCiao();
}
