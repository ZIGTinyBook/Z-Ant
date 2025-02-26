const std = @import("std");
const os = std.os;

const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const allocator = zant.utils.allocator.allocator;

// --- proto libs
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeType = onnx.AttributeType;

// --- codeGen libs
const ReadyNode = @import("globals.zig").ReadyNode;
const ReadyTensor = @import("globals.zig").ReadyTensor;
const utils = @import("codeGen_utils.zig");
const codegen_options = @import("codegen_options");

// ----------------------------------- MATH -----------------------------------

/// This method map and write the ONNX operations with the Zant LeanTensorMath mathods
/// Follow the link for details: https://onnx.ai/onnx/operators/?utm_source=chatgpt.com
pub fn write_math_op(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    if (!codegen_options.noComm) {
        try write_op_info(writer, node);
    }
    if (codegen_options.log) {
        try writer.print(
            \\ 
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running {s} operation...\n")));
            \\    }}
        , .{node.*.nodeProto.*.op_type});
    }

    if (std.mem.eql(u8, node.nodeProto.op_type, "Add")) {
        try write_add(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "AveragePool")) {
        try writer.writeAll("// Handle AveragePool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "BatchNormalization")) {
        try writer.writeAll("// Handle BatchNormalization\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Concat")) {
        try writer.writeAll("// Handle Concat\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Constant")) {
        try writer.writeAll("// Handle Constant\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Conv")) {
        try write_conv(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Div")) {
        try write_div(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Flatten")) {
        try writer.writeAll("// Handle Flatten\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gather")) {
        try write_gather(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gemm")) {
        try write_gemm(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LeakyRelu")) {
        try writer.writeAll("// Handle LeakyRelu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LogSoftmax")) {
        try writer.writeAll("// Handle LogSoftmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MatMul")) {
        try write_matmul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try write_maxPool(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try write_mul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "ReduceMean")) {
        try write_reduceMean(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try write_ReLU(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try write_reshape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try writer.writeAll("// Handle Resize\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sigmoid")) {
        try write_sigmoid(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Softmax")) {
        try write_softmax(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Slice")) {
        try write_slice(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Split")) {
        try writer.writeAll("// Handle Split\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sub")) {
        try writer.writeAll("// Handle Sub\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sum")) {
        try write_sum(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Transpose")) {
        try write_transpose(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Shape")) {
        try write_shape(writer, node);
    } else {
        return error.OperationNotSupported;
    }

    try writer.writeAll(" catch return;");
}

fn write_op_info(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    try writer.print(
        \\
        \\
        \\   //forwarding operation : {s}
        \\   //parameters:
        \\   //   inputs: 
    , .{node.*.nodeProto.*.op_type});

    //write the inputs
    for (node.inputs.items) |input| {
        try writer.print(
            \\
            \\   //      -> {s} 
        , .{input.name});
    }
    try writer.print(
        \\
        \\   //    outputs: 
    , .{});

    //write the outputs
    for (node.outputs.items) |output| {
        try writer.print(
            \\
            \\   //      <- {s} 
        , .{output.name});
    }
}

inline fn write_add(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Add.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensors_lean(T, T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_conv(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Conv.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    //      - W (heterogeneous) - T: The weight tensor
    //      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
    // OUTPUT:
    //      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
    //      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
    //      - group - INT (default is '1'): number of groups input channels and output channels are divided into
    //      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";
    var dilations: ?[]i64 = null;
    var group: i64 = 1;
    var kernel_shape: ?[]i64 = null;
    var pads: ?[]i64 = null;
    var strides: ?[]i64 = null; //mandatory

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
            if (attr.type == AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
        }
    }

    //----create ?bias string
    var bias_string: []u8 = undefined;
    // Bias Tensor B is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const B_name = try utils.getSanitizedName(node.inputs.items[2].name);
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", B_name, ")" });
    } else {
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
    }

    //----create stride string (mandatory)
    // TODO: implement default stride, see docs above
    if (strides == null) return error.StrideNotFound;
    const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(strides.?);

    //----create ?pads string
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        pads_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    //----create ?dilatations string
    var dilat_string: []const u8 = undefined;
    if (dilations != null) {
        dilat_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilat_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    // pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void
    _ = try writer.print(
        \\    
        \\    tensMath.conv_lean(
        \\        T, //type
        \\        &tensor_{s}, //input
        \\        @constCast(&tensor_{s}), //kernel
        \\        &tensor_{s}, //output
        \\        {s}, //bias
        \\        {s}, //stride
        \\        {s}, //pads
        \\        {s}, //dilatations
        \\        {}, //group
        \\        "{s}", //auto_pad
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), //Input
        try utils.getSanitizedName(node.inputs.items[1].name), //Kernel
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        bias_string, //Bias
        stride_string, //Strides
        pads_string, //Pads
        dilat_string, //Dilatations
        group, //Group
        auto_pad,
    });
}

inline fn write_div(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Div.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.div_lean(T, &tensor_{s}, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_gather(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gather.html
    // INPUTS:
    //      - data (heterogeneous) - T: Tensor of rank r >= 1.
    //      - indices (heterogeneous) - tensor(int64): Tensor of int64 indices, of any rank q.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Tensor of rank q + r - 1.
    // ATTRIBUTES:
    //      - axis (int, default is 0): Which axis to gather on. Negative value means counting dimensions from the back.

    var axis: i64 = 0;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        }
    }

    const indices_name = try utils.getSanitizedName(node.inputs.items[1].name);

    _ = try writer.print(
        \\    
        \\    //creating the indices Tensor(usize)
        \\    
        \\    var usize_slice_{s} = utils.i64SliceToUsizeSlice(tensor_{s}.data);
        \\    var usize_tensor_{s} = Tensor(usize).fromConstBuffer(&allocator, usize_slice_{s}, &tensor_{s}.shape);
        \\    defer usize_tensor_{s}.deinit();
        \\    
    , .{
        indices_name, //usize_slice_
        indices_name, //i64SliceToUsizeSlice
        indices_name, //usize_tensor_
        indices_name, //usize_slice_
        indices_name, //tensor_.sahpe
        indices_name, //usize_tensor_.deinit
    });

    _ = try writer.print(
        \\
        \\    tensMath.gather_lean(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //data tensor
        \\        &usize_tensor_{s}, //indices tensor
        \\        {}, //axis
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input data tensor
        indices_name, // Input indices tensor
        axis, // Selected axis
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_gemm(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gemm.html
    // INPUTS:
    //      - Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
    //      - Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
    //      - Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
    //OUTPUTS:
    //      - Output tensor of shape (M, N).
    // ATTRIBUTES:
    //      - alpha. FLOAT (default is '1.0'): Scalar multiplier for the product of input tensors A * B.
    //      - beta - FLOAT (default is '1.0'): Scalar multiplier for input tensor C.
    //      - transA - INT (default is '0'): Whether A should be transposed
    //      - transB - INT (default is '0'): Whether B should be transposed

    var alpha: f32 = 1.0;
    var beta: f32 = 1.0;
    var transA: bool = false;
    var transB: bool = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
            if (attr.type == AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
            if (attr.type == AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
            if (attr.type == AttributeType.INT) transA = if (attr.i != 0) true else false else return error.GemmTransANotINT;
        } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
            if (attr.type == AttributeType.INT) transB = if (attr.i != 0) true else false else return error.GemmTransBNotINT;
        }
    }

    var c_tensor_string: []u8 = undefined;
    // Input Tensor C is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const C_name = try utils.getSanitizedName(node.inputs.items[2].name);
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", C_name, ")" });
    } else {
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    _ = try writer.print(
        \\
        \\    tensMath.gemm_lean(T, &tensor_{s}, @constCast(&tensor_{s}), {s}, {}, {}, {s}, {s}, &tensor_{s} )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        c_tensor_string,
        alpha,
        beta,
        if (transA) "true" else "false",
        if (transB) "true" else "false",
        try utils.getSanitizedName(node.outputs.items[0].name), // Output
    });
}

inline fn write_matmul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__MatMul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.mat_mul_lean(T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_maxPool(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //https://onnx.ai/onnx/operators/onnx__MaxPool.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output data tensor from average or max pooling across the input tensor.
    //      - (NOT IMPLEMENTED) Indices (optional, heterogeneous) - I: Indices tensor from max pooling across the input tensor.
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
    //      - ceil_mode - INT (default is '0'): Whether to use ceil or floor (default) to compute the output shape
    //      - dilations - INTS : Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis
    //      - kernel_shape - INTS (required) : The size of the kernel along each axis.
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - storage_order - INT (default is '0'): The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";

    var ceil_mode: i64 = 0;

    var dilations: ?[]i64 = null;

    var kernel_shape: ?[]i64 = null; //mandatory

    var pads: ?[]i64 = null;

    var storage_order: i64 = 0;

    var strides: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.MaxPoolAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
            if (attr.type == AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.MaxPoolDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.MaxPoolKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.MaxPoolPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "storage_order")) |_| {
            if (attr.type == AttributeType.INT) storage_order = attr.i else return error.MaxPoolStorage_orderNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.MaxPoolStridesNotINTS;
        }
    }

    //----create kernel_shape string
    var kernel_shape_string: []const u8 = undefined;
    if (kernel_shape != null) {
        kernel_shape_string = try utils.i64SliceToUsizeArrayString(kernel_shape.?);
    } else {
        return error.Kernel_shapeNotFound;
    }

    //----create strides string
    var strides_string: []const u8 = undefined;
    if (strides != null) {
        strides_string = try utils.i64SliceToUsizeArrayString(strides.?);
    } else {
        return error.StridesNotFound;
    }

    //----create dilations string
    var dilations_string: []const u8 = undefined;
    if (dilations != null) {
        dilations_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilations_string = try utils.i64SliceToUsizeArrayString(&[_]i64{ 1, 1, 1, 1 }); // TODO: It is hardcoded in 4D, not the most elegant solution
    }

    //----create pads string
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        return error.PadsNotFound;
    }

    // pub fn lean_onnx_maxpool(
    //     comptime T: type,
    //     input: *Tensor(T),
    //     output: *Tensor(T),
    //     kernel_shape: []const usize,
    //     strides: []const usize,
    //     dilations: []const usize,
    //     pads: []const usize,
    //     auto_pad: AutoPadType,
    // ) !void

    _ = try writer.print(
        \\
        \\    tensMath.onnx_maxpool_lean(
        \\        T,
        \\        &tensor_{s}, //Input
        \\        &tensor_{s}, //Output
        \\        {s}, //kernel_shape
        \\        {s}, //strides
        \\        {s}, //dilations
        \\        {s}, //pads
        \\        tensMath.AutoPadType.{s}, //auto_pad
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), //Input
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        kernel_shape_string, //kernel_shape
        strides_string, //strides
        dilations_string, //dilatations
        pads_string, //pads
        auto_pad, //auto_pad
    });
}

inline fn write_mul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Mul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.mul_lean(T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_reduceMean(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    var keepdims: bool = true;
    var noop_with_empty_axes: bool = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
            if (attr.type == AttributeType.INT) noop_with_empty_axes = attr.i != 0;
        }
    }

    var axes_str: []const u8 = "null";
    if (node.inputs.items.len > 1) {
        axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{try utils.getSanitizedName(node.inputs.items[1].name)});
    }

    _ = try writer.print(
        \\
        \\    tensMath.reduce_mean_lean(T, &tensor_{s}, &tensor_{s}, {s}, {s}, {s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
        axes_str,
        if (keepdims) "true" else "false",
        if (noop_with_empty_axes) "true" else "false",
    });
}

inline fn write_ReLU(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    tensMath.ReLU_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_reshape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    //      - shape (heterogeneous) - tensor(int64): Specified shape for output.
    // OUTPUTS:
    //      - reshaped (heterogeneous) - T: Reshaped data.
    // ATTRIBUTES:
    //      - allowzero - INT (default is '0'): Whether to allow zeros in shape tensor

    var allowzer0: bool = false;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "allowzero")) |_| {
            if (attr.type == AttributeType.INT) allowzer0 = attr.i != 0;
        }
    }
    // const newShape_tensor_parameter193_reshape1_shapee: []usize = utils.i64SliceToUsizeSlice(tensor_pooling160_output_0_reshape0_shape.data) catch return;
    // defer allocator.free(newShape_tensor_parameter193_reshape1_shapee);
    _ = try writer.print(
        \\
        \\    const newShape_tensor_{s}: []usize = utils.i64SliceToUsizeSlice(tensor_{s}.data) catch return;
        \\    defer allocator.free(newShape_tensor_{s});
    , .{
        try utils.getSanitizedName(node.inputs.items[1].name),
        try utils.getSanitizedName(node.inputs.items[1].name),
        try utils.getSanitizedName(node.inputs.items[1].name),
    });

    _ = try writer.print(
        \\
        \\    tensMath.reshape_lean(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //Input tensor
        \\        newShape_tensor_{s}, //New shape
        \\        {s}, //allowzero
        \\        &tensor_{s}, //Output tensor
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor
        try utils.getSanitizedName(node.inputs.items[1].name), // Input shape tensor
        if (allowzer0) "true" else "false", //allowzer0
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_sigmoid(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    tensMath.sigmoid_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_slice(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Slice.html
    // INPUTS:
    //      - input (heterogeneous) - T: Tensor of data to extract slices from.
    //      - starts (heterogeneous) - T1: 1-D tensor of starting indices of corresponding axis in `axes`.
    //      - ends (heterogeneous) - T1: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`.
    //      - axes (heterogeneous) - T1: 1-D tensor of axes that `starts` and `ends` apply to.
    //      - steps (heterogeneous) - T1: 1-D tensor of slice step of corresponding axis in `axes`.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Sliced data tensor.

    // First, get the sanitized names for all tensors
    const input_name = try utils.getSanitizedName(node.inputs.items[0].name);
    const starts_name = try utils.getSanitizedName(node.inputs.items[1].name);
    const ends_name = try utils.getSanitizedName(node.inputs.items[2].name);
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Handle optional axes and steps inputs
    var axes_str: []const u8 = "null";
    var steps_str: []const u8 = "null";

    if (node.inputs.items.len > 3) {
        const axes_name = try utils.getSanitizedName(node.inputs.items[3].name);
        axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{axes_name});
    }

    if (node.inputs.items.len > 4) {
        const steps_name = try utils.getSanitizedName(node.inputs.items[4].name);
        steps_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{steps_name});
    }

    _ = try writer.print(
        \\
        \\    tensMath.lean_slice_onnx(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        &tensor_{s}.data, //starts
        \\        &tensor_{s}.data, //ends
        \\        {s}, //axes
        \\        {s}, //steps
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        input_name,
        starts_name,
        ends_name,
        axes_str,
        steps_str,
        output_name,
    });

    if (axes_str.len > 4) allocator.free(axes_str);
    if (steps_str.len > 4) allocator.free(steps_str);
}

inline fn write_softmax(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    tensMath.softmax_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_sum(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Sum.html
    // INPUTS:
    //      - list of tensors
    // OUTPUTS:
    //      - sum (heterogeneous) - T: Output tensor.

    //Writing the tensor list with all the inputs
    _ = try writer.print(
        \\
        \\    const my_tensor_list = [_]*Tensor(T){{
    , .{});

    for (node.inputs.items, 0..) |tens, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }
        _ = try writer.print(
            \\tensor_{s}
        , .{try utils.getSanitizedName(tens.name)});
    }

    _ = try writer.print("}}", .{});

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensor_list_lean(T, T, &my_tensor_list, &tensor_{s})
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});
}

inline fn write_transpose(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Transpose.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    // OUTPUTS:
    //      - transposed (heterogeneous) - T: Transposed output.
    // ATTRIBUTES:
    //      - perm - INTS: A list of integers. By default, reverse the dimensions,
    //        otherwise permute the axes according to the values given.

    // Get the perm attribute if it exists
    var perm_str: []const u8 = "null";
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "perm")) {
            if (attr.type == AttributeType.INTS) {
                perm_str = try utils.i64SliceToUsizeArrayString(attr.ints);
            }
        }
    }

    _ = try writer.print(
        \\
        \\    tensMath.transpose_onnx_lean(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        {s}, //perm
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor
        perm_str, // Permutation array
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn compute_transpose_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_transpose_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get the perm attribute if it exists
    var perm: ?[]i64 = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "perm")) {
            if (attr.type == AttributeType.INTS) {
                perm = attr.ints;
            }
        }
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n perm: []i64 = {any}", .{perm});

    // If no perm is provided, reverse the dimensions
    var output_shape = try allocator.alloc(i64, input_shape.len);
    if (perm) |p| {
        // Validate perm length
        if (p.len != input_shape.len) {
            return error.InvalidPermutation;
        }
        // Apply permutation
        for (p, 0..) |axis, i| {
            if (axis < 0 or axis >= @as(i64, @intCast(input_shape.len))) {
                return error.InvalidPermutation;
            }
            output_shape[i] = input_shape[@intCast(axis)];
        }
    } else {
        // Reverse dimensions
        for (input_shape, 0..) |_, i| {
            output_shape[i] = input_shape[input_shape.len - 1 - i];
        }
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

// ----------------------------------- SHAPE inference -----------------------------------

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        //https://onnx.ai/onnx/operators/onnx__Add.html
        readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Constant")) {
        //https://onnx.ai/onnx/operators/onnx__Constant.html
        try compute_constant_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Conv")) {
        //https://onnx.ai/onnx/operators/onnx__Conv.html
        try compute_conv_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Div")) {
        //https://onnx.ai/onnx/operators/onnx__Div.html
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Flatten")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        try compute_gather_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        //https://onnx.ai/onnx/operators/onnx__Gemm.html
        try compute_gemm_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        try compute_maxPool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        //https://onnx.ai/onnx/operators/onnx__Mul.html
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "ReduceMean")) {
        try compute_reduceMean_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        //https://onnx.ai/onnx/operators/onnx__Relu.html
        try compute_ReLU_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // https://onnx.ai/onnx/operators/onnx__Reshape.html
        try compute_reshape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        try compute_shape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        try compute_sigmoid_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        //https://onnx.ai/onnx/operators/onnx__Softmax.html
        try compute_softmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        try compute_slice_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        try compute_transpose_output_shape(readyNode);
    } else {
        std.debug.print("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------

inline fn compute_constant_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_constant_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const shape = try utils.getConstantTensorDims(readyNode.nodeProto);
    std.debug.print("\n output_shape: []i64 = {any}", .{shape});
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_ReLU_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_ReLU_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_reshape_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_reshape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    std.debug.print("\n new shape: []i64 = {any}", .{readyNode.inputs.items[1].tensorProto.?.int64_data.?});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    std.debug.print("\n output_shape shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_softmax_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_softmax_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_gemm_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_gemm_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    std.debug.print("\n weight_shape: []i64 = {any}", .{readyNode.inputs.items[1].shape});
    std.debug.print("\n bias_shape: []i64 = {any}", .{readyNode.inputs.items[2].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[2].shape;
}

fn compute_mul_output_shape(readyNode: *ReadyNode) !void {
    const input_a = readyNode.inputs.items[0];
    const input_b = readyNode.inputs.items[1];

    std.debug.print("\n====== compute_mul_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_a_shape: []i64 = {any}", .{input_a.shape});
    std.debug.print("\n input_b_shape: []i64 = {any}", .{input_b.shape});

    if (input_a.shape.len < 2 or input_b.shape.len < 2) {
        return error.InvalidShape;
    }

    const a_rows = input_a.shape[input_a.shape.len - 2];
    const a_cols = input_a.shape[input_a.shape.len - 1];
    const b_rows = input_b.shape[input_b.shape.len - 2];
    const b_cols = input_b.shape[input_b.shape.len - 1];

    if (a_cols != b_rows) {
        return error.ShapeMismatch;
    }

    var output_shape: std.ArrayList(i64) = std.ArrayList(i64).init(allocator);
    defer output_shape.deinit();

    // Broadcast the batch dimensions
    const a_batch = input_a.shape.len - 2;
    const b_batch = input_b.shape.len - 2;
    const batch_dims = if (a_batch < b_batch) a_batch else b_batch;

    for (0..batch_dims) |i| {
        try output_shape.append(if (input_a.shape[i] > input_b.shape[i]) input_a.shape[i] else input_b.shape[i]);
    }

    // Append output matrix dimensions
    try output_shape.append(a_rows);
    try output_shape.append(b_cols);

    // Update the shape of the output tensor
    readyNode.outputs.items[0].shape = try output_shape.toOwnedSlice();
}

inline fn compute_conv_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_conv_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape: []const i64 = readyNode.inputs.items[1].shape;
    const stride = readyNode.nodeProto.attribute[1].ints;
    const dilation = readyNode.nodeProto.attribute[4].ints;
    const auto_pad = readyNode.nodeProto.attribute[2].s;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape});
    std.debug.print("\n stride: []i64 = {any}", .{stride});

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_convolution_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                try utils.i64SliceToUsizeSlice(kernel_shape),
                try utils.i64SliceToUsizeSlice(stride),
                null,
                try utils.i64SliceToUsizeSlice(dilation),
                auto_pad,
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_maxPool_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_maxPool_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape = readyNode.nodeProto.attribute[0].ints;
    const stride = readyNode.nodeProto.attribute[1].ints;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape});
    std.debug.print("\n stride: []i64 = {any}", .{stride});

    const kernel_2d = [2]usize{ @intCast(kernel_shape[0]), @intCast(kernel_shape[1]) };
    const stride_2d = [2]usize{ @intCast(stride[0]), @intCast(stride[1]) };

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_pooling_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                kernel_2d,
                stride_2d,
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_reduceMean_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_reduceMean_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;

    var keepdims: bool = true;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        }
    }

    // If we have axes input tensor
    if (readyNode.inputs.items.len > 1) {
        const axes = readyNode.inputs.items[1].shape;
        std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
        std.debug.print("\n axes: []i64 = {any}", .{axes});

        // If keepdims is true, the reduced dimensions are retained with length 1
        if (keepdims) {
            var newShape = try allocator.alloc(i64, input_shape.len);
            @memcpy(newShape, input_shape);
            for (axes) |axis| {
                newShape[@as(usize, @intCast(axis))] = 1;
            }
            readyNode.outputs.items[0].shape = newShape;
        } else {
            // If keepdims is false, the reduced dimensions are removed
            var newShape = try allocator.alloc(i64, input_shape.len - axes.len);
            var j: usize = 0;
            outer: for (input_shape, 0..) |dim, i| {
                for (axes) |axis| {
                    if (i == @as(usize, @intCast(axis))) continue :outer;
                }
                newShape[j] = dim;
                j += 1;
            }
            readyNode.outputs.items[0].shape = newShape;
        }
    } else {
        // If no axes specified, reduce all dimensions to 1 if keepdims is true, or to a scalar if false
        if (keepdims) {
            var newShape = try allocator.alloc(i64, input_shape.len);
            for (newShape, 0..) |_, i| {
                newShape[i] = 1;
            }
            readyNode.outputs.items[0].shape = newShape;
        } else {
            readyNode.outputs.items[0].shape = &[_]i64{1};
        }
    }
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_slice_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_slice_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    const starts = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    const ends = readyNode.inputs.items[2].tensorProto.?.int64_data.?;

    var axes: ?[]i64 = null;
    var steps: ?[]i64 = null;

    // Get axes if provided (input 3)
    if (readyNode.inputs.items.len > 3) {
        axes = readyNode.inputs.items[3].tensorProto.?.int64_data.?;
    }

    // Get steps if provided (input 4)
    if (readyNode.inputs.items.len > 4) {
        steps = readyNode.inputs.items[4].tensorProto.?.int64_data.?;
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n starts: []i64 = {any}", .{starts});
    std.debug.print("\n ends: []i64 = {any}", .{ends});
    std.debug.print("\n axes: []i64 = {any}", .{axes});
    std.debug.print("\n steps: []i64 = {any}", .{steps});

    const output_shape = try tensorMath.get_slice_output_shape(
        try utils.i64SliceToUsizeSlice(input_shape),
        starts,
        ends,
        axes,
        steps,
    );

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_shape_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_shape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get start and end attributes if they exist
    var start: ?i64 = null;
    var end: ?i64 = null;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "start")) {
            if (attr.type == AttributeType.INT) start = attr.i;
        } else if (std.mem.eql(u8, attr.name, "end")) {
            if (attr.type == AttributeType.INT) end = attr.i;
        }
    }

    // Calculate output size
    const rank = input_shape.len;
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank - 1))));

    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    const output_size = @max(0, end_axis - start_axis);

    // Shape operator always outputs a 1D tensor
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{@intCast(output_size)});
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_gather_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_gather_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const data_shape = readyNode.inputs.items[0].shape;
    const indices_shape = readyNode.inputs.items[1].shape;

    // Get axis attribute, default is 0
    var axis: i64 = 0;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        }
    }

    // Handle negative axis
    if (axis < 0) {
        axis += @as(i64, @intCast(data_shape.len));
    }

    std.debug.print("\n data_shape: []i64 = {any}", .{data_shape});
    std.debug.print("\n indices_shape: []i64 = {any}", .{indices_shape});
    std.debug.print("\n axis: {}", .{axis});

    // Calculate output shape:
    // - Take all dimensions from data before axis
    // - Add all dimensions from indices
    // - Take all dimensions from data after axis
    const output_rank = data_shape.len + indices_shape.len - 1;
    var output_shape = try allocator.alloc(i64, output_rank);

    // Copy dimensions from data before axis
    var out_idx: usize = 0;
    for (0..@as(usize, @intCast(axis))) |i| {
        output_shape[out_idx] = data_shape[i];
        out_idx += 1;
    }

    // Copy dimensions from indices
    for (indices_shape) |dim| {
        output_shape[out_idx] = dim;
        out_idx += 1;
    }

    // Copy remaining dimensions from data after axis
    for (@as(usize, @intCast(axis + 1))..data_shape.len) |i| {
        output_shape[out_idx] = data_shape[i];
        out_idx += 1;
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_sigmoid_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_sigmoid_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Sigmoid is element-wise, output shape is identical to input shape
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn write_shape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Shape.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    // OUTPUTS:
    //      - shape (heterogeneous) - T1: Shape of the input tensor
    // ATTRIBUTES:
    //      - start - INT: First dimension to take
    //      - end - INT: Last dimension to take

    var start: ?i64 = null;
    var end: ?i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "start")) {
            if (attr.type == AttributeType.INT) start = attr.i;
        } else if (std.mem.eql(u8, attr.name, "end")) {
            if (attr.type == AttributeType.INT) end = attr.i;
        }
    }

    _ = try writer.print(
        \\
        \\    tensMath.shape_onnx(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        {s}, //start
        \\        {s}, //end
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        if (start) |s| try std.fmt.allocPrint(allocator, "{}", .{s}) else "null",
        if (end) |e| try std.fmt.allocPrint(allocator, "{}", .{e}) else "null",
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}
