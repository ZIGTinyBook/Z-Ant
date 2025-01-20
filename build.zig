const std = @import("std");

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {
    const build_options = b.addOptions();
    build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);
    build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse "raw_c_allocator"));

    // Set target options, such as architecture and OS.
    const target = b.standardTargetOptions(.{});
    //const target = b.standardTargetOptions(.{ .default_target = .{
    //.cpu_arch = .thumb,
    //.os_tag = .freestanding,
    //.cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m0plus },
    //} });

    // Set optimization level (debug, release, etc.).
    const optimize = b.standardOptimizeOption(.{});

    //************************************************MODULE CREATION************************************************

    // Create modules from the source files in the `src/Core/Tensor/` directory.
    const tensor_mod = b.addModule("tensor", .{ .root_source_file = b.path("src/Core/Tensor/tensor.zig") });
    const tensor_math_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/tensor_math.zig") });
    const architectures_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/architectures.zig") });

    // Create modules from the source files in the `src/Model/` directory.
    const loss_mod = b.createModule(.{ .root_source_file = b.path("src/Model/lossFunction.zig") });
    const activation_mod = b.createModule(.{ .root_source_file = b.path("src/Model/activation_function.zig") });
    const model_mod = b.createModule(.{ .root_source_file = b.path("src/Model/model.zig") });
    const layer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/layer.zig") });
    const optim_mod = b.createModule(.{ .root_source_file = b.path("src/Model/optim.zig") });

    // Create modules from the source files in the `src/Model/Layers` directory.
    const denseLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/denseLayer.zig") });
    const activationLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/activationLayer.zig") });
    const convLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/convLayer.zig") });
    const flattenLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/flattenLayer.zig") });
    const poolingLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/poolingLayer.zig") });

    // Create modules from the source files in the `src/DataHandler/` directory.
    const dataloader_mod = b.createModule(.{ .root_source_file = b.path("src/DataHandler/dataLoader.zig") });
    const dataProcessor_mod = b.createModule(.{ .root_source_file = b.path("src/DataHandler/dataProcessor.zig") });
    const trainer_mod = b.createModule(.{ .root_source_file = b.path("src/DataHandler/trainer.zig") });

    // Create modules from the source files in the `src/utils/` directory.
    const typeConv_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/typeConverter.zig") });
    const errorHandler_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/errorHandler.zig") });
    const modelImportExport_mod = b.addModule("model_import_export", .{ .root_source_file = b.path("src/Utils/model_import_export.zig") });
    const allocator_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/allocator.zig") });
    allocator_mod.addOptions("build_options", build_options);

    // Create modules from the source files in the `src/Codegen` directory.
    const codegen_mod = b.createModule(.{ .root_source_file = b.path("src/Codegen/codegen.zig") });

    //************************************************MODEL DEPENDENCIES************************************************

    // Add necessary imports for the model module.
    model_mod.addImport("tensor", tensor_mod);
    model_mod.addImport("layer", layer_mod);
    model_mod.addImport("optim", optim_mod); // Do not remove duplicate
    model_mod.addImport("loss", loss_mod);
    model_mod.addImport("typeC", typeConv_mod);
    model_mod.addImport("dataloader", dataloader_mod);
    model_mod.addImport("tensor_m", tensor_math_mod);
    model_mod.addImport("dataprocessor", dataProcessor_mod);
    model_mod.addImport("activation_function", activation_mod);

    //************************************************LAYER DEPENDENCIES************************************************

    // Add necessary imports for the layers module.
    layer_mod.addImport("tensor", tensor_mod);
    layer_mod.addImport("activation_function", activation_mod);
    layer_mod.addImport("tensor_m", tensor_math_mod);
    layer_mod.addImport("architectures", architectures_mod);
    layer_mod.addImport("errorHandler", errorHandler_mod);
    layer_mod.addImport("pkgAllocator", allocator_mod);

    //************************************************DENSELAYER DEPENDENCIES************************************************

    // Add necessary imports for the denselayers module.
    denseLayer_mod.addImport("tensor", tensor_mod);
    denseLayer_mod.addImport("tensor_m", tensor_math_mod);
    denseLayer_mod.addImport("Layer", layer_mod);
    denseLayer_mod.addImport("architectures", architectures_mod);
    denseLayer_mod.addImport("errorHandler", errorHandler_mod);
    denseLayer_mod.addImport("codegen", codegen_mod);

    //************************************************CONVLAYER DEPENDENCIES************************************************
    convLayer_mod.addImport("Tensor", tensor_mod);
    convLayer_mod.addImport("tensor_m", tensor_math_mod);
    convLayer_mod.addImport("Layer", layer_mod);
    convLayer_mod.addImport("architectures", architectures_mod);
    convLayer_mod.addImport("errorHandler", errorHandler_mod);

    //************************************************FLATTENLAYER DEPENDENCIES************************************************

    flattenLayer_mod.addImport("Tensor", tensor_mod);
    flattenLayer_mod.addImport("tensor_m", tensor_math_mod);
    flattenLayer_mod.addImport("Layer", layer_mod);
    flattenLayer_mod.addImport("architectures", architectures_mod);
    flattenLayer_mod.addImport("errorHandler", errorHandler_mod);

    //************************************************POOLINGLAYER DEPENDENCIES************************************************
    poolingLayer_mod.addImport("Tensor", tensor_mod);
    poolingLayer_mod.addImport("tensor_m", tensor_math_mod);
    poolingLayer_mod.addImport("Layer", layer_mod);
    poolingLayer_mod.addImport("architectures", architectures_mod);
    poolingLayer_mod.addImport("errorHandler", errorHandler_mod);

    //************************************************ACTIVATIONLAYER DEPENDENCIES************************************************

    // Add necessary imports for the activationlayers module.
    activationLayer_mod.addImport("tensor", tensor_mod);
    activationLayer_mod.addImport("tensor_m", tensor_math_mod);
    activationLayer_mod.addImport("Layer", layer_mod);
    activationLayer_mod.addImport("architectures", architectures_mod);
    activationLayer_mod.addImport("activation_function", activation_mod);
    activationLayer_mod.addImport("errorHandler", errorHandler_mod);
    activationLayer_mod.addImport("codegen", codegen_mod);

    //************************************************DATA LOADER DEPENDENCIES************************************************

    // Add necessary imports for the data loader module.
    dataloader_mod.addImport("tensor", tensor_mod);

    //************************************************DATA PROCESSOR DEPENDENCIES************************************************

    // Add necessary imports for the data processor module.
    dataProcessor_mod.addImport("tensor", tensor_mod);

    //************************************************TRAINER DEPENDENCIES************************************************

    // Add necessary imports for the trainer module.
    trainer_mod.addImport("tensor", tensor_mod);
    trainer_mod.addImport("tensor_m", tensor_math_mod);
    trainer_mod.addImport("model", model_mod);
    trainer_mod.addImport("loss", loss_mod);
    trainer_mod.addImport("optim", optim_mod);
    trainer_mod.addImport("dataloader", dataloader_mod);
    trainer_mod.addImport("dataprocessor", dataProcessor_mod);

    //************************************************TENSOR DEPENDENCIES************************************************

    // Add necessary imports for the tensor module.
    tensor_mod.addImport("tensor_m", tensor_math_mod);
    tensor_mod.addImport("architectures", architectures_mod);
    tensor_mod.addImport("errorHandler", errorHandler_mod);

    //************************************************TENSOR MATH DEPENDENCIES************************************************

    // Add necessary imports for the tensor math module.
    tensor_math_mod.addImport("tensor", tensor_mod);
    tensor_math_mod.addImport("typeC", typeConv_mod);
    tensor_math_mod.addImport("architectures", architectures_mod);
    tensor_math_mod.addImport("errorHandler", errorHandler_mod);
    tensor_math_mod.addImport("Layer", layer_mod);
    tensor_math_mod.addImport("pkgAllocator", allocator_mod);
    tensor_math_mod.addImport("poolingLayer", poolingLayer_mod);

    //************************************************ACTIVATION DEPENDENCIES************************************************

    // Add necessary imports for the activation module.
    activation_mod.addImport("tensor", tensor_mod);
    activation_mod.addImport("errorHandler", errorHandler_mod);
    activation_mod.addImport("pkgAllocator", allocator_mod);

    //************************************************LOSS DEPENDENCIES************************************************

    // Add necessary imports for the loss function module.
    loss_mod.addImport("tensor", tensor_mod);
    loss_mod.addImport("tensor_m", tensor_math_mod);
    loss_mod.addImport("typeC", typeConv_mod);
    loss_mod.addImport("errorHandler", errorHandler_mod);
    loss_mod.addImport("pkgAllocator", allocator_mod);

    //************************************************OPTIMIZER DEPENDENCIES************************************************

    // Add necessary imports for the optimizer module.
    optim_mod.addImport("tensor", tensor_mod);
    optim_mod.addImport("model", model_mod);
    optim_mod.addImport("layer", layer_mod);
    optim_mod.addImport("errorHandler", errorHandler_mod);
    optim_mod.addImport("denselayer", denseLayer_mod);

    //************************************************IMPORT/EXPORT DEPENDENCIES************************************************

    // Add necessary imports for the import/export module.
    modelImportExport_mod.addImport("tensor", tensor_mod);
    modelImportExport_mod.addImport("layer", layer_mod);
    modelImportExport_mod.addImport("activationlayer", activationLayer_mod);
    modelImportExport_mod.addImport("denselayer", denseLayer_mod);
    modelImportExport_mod.addImport("model", model_mod);
    modelImportExport_mod.addImport("errorHandler", errorHandler_mod);
    modelImportExport_mod.addImport("activation_function", activation_mod);

    //************************************************MAIN EXECUTABLE************************************************

    // Define the main executable with target architecture and optimization settings.
    const exe = b.addExecutable(.{
        .name = "Main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibC();

    //************************************************EXE DEPENDENCIES************************************************

    // Add necessary imports for the main executable.
    exe.root_module.addImport("tensor", tensor_mod);
    exe.root_module.addImport("codegen", codegen_mod);
    exe.root_module.addImport("model", model_mod);
    exe.root_module.addImport("layer", layer_mod);
    exe.root_module.addImport("dataloader", dataloader_mod);
    exe.root_module.addImport("dataprocessor", dataProcessor_mod);
    exe.root_module.addImport("activation_function", activation_mod);
    exe.root_module.addImport("loss", loss_mod);
    exe.root_module.addImport("trainer", trainer_mod);
    exe.root_module.addImport("denselayer", denseLayer_mod);
    exe.root_module.addImport("activationlayer", activationLayer_mod);
    exe.root_module.addImport("convLayer", convLayer_mod);
    exe.root_module.addImport("flattenLayer", flattenLayer_mod);
    exe.root_module.addImport("poolingLayer", poolingLayer_mod);
    exe.root_module.addImport("pkgAllocator", allocator_mod);
    exe.root_module.addImport("model_import_export", modelImportExport_mod);

    // Install the executable.
    b.installArtifact(exe);

    // Define the run command for the main executable.
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    //************************************************UNIT TESTS************************************************

    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "lib_test",
        .root_source_file = b.path("src/tests/lib_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    //************************************************UNIT TEST DEPENDENCIES************************************************

    // Add necessary imports for the unit test module.
    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("model", model_mod);
    unit_tests.root_module.addImport("layer", layer_mod);
    unit_tests.root_module.addImport("optim", optim_mod);
    unit_tests.root_module.addImport("loss", loss_mod);
    unit_tests.root_module.addImport("tensor_m", tensor_math_mod);
    unit_tests.root_module.addImport("activation_function", activation_mod);
    unit_tests.root_module.addImport("dataloader", dataloader_mod);
    unit_tests.root_module.addImport("dataprocessor", dataProcessor_mod);
    unit_tests.root_module.addImport("architectures", architectures_mod);
    unit_tests.root_module.addImport("trainer", trainer_mod);
    unit_tests.root_module.addImport("typeConverter", typeConv_mod);
    unit_tests.root_module.addImport("errorHandler", errorHandler_mod);
    unit_tests.root_module.addImport("model_import_export", modelImportExport_mod);
    unit_tests.root_module.addImport("denselayer", denseLayer_mod);
    unit_tests.root_module.addImport("activationlayer", activationLayer_mod);
    unit_tests.root_module.addImport("convLayer", convLayer_mod);
    unit_tests.root_module.addImport("flattenLayer", flattenLayer_mod);
    unit_tests.root_module.addImport("poolingLayer", poolingLayer_mod);
    unit_tests.root_module.addImport("pkgAllocator", allocator_mod);

    unit_tests.linkLibC();

    // Add tests for the optimizer module.
    const optim_tests = b.addTest(.{
        .name = "optim_test",
        .root_source_file = b.path("src/tests/tests_optim.zig"),
        .target = target,
        .optimize = optimize,
    });
    optim_tests.root_module.addImport("optim", optim_mod);

    // Define the run command for optimizer tests.
    const run_optim_tests = b.addRunArtifact(optim_tests);
    const test_optim_step = b.step("test_optim", "Test for Optim");
    test_optim_step.dependOn(&run_optim_tests.step);

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test_all", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
