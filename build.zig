const std = @import("std");

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {
    const build_options = b.addOptions();
    build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);
    build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse "raw_c_allocator"));

    // Get target and CPU options from command line or use defaults
    const target_str = b.option([]const u8, "target", "Target architecture (e.g., thumb-freestanding)") orelse "native";
    const cpu_str = b.option([]const u8, "cpu", "CPU model (e.g., cortex_m33)");

    const target_query = std.zig.CrossTarget.parse(.{
        .arch_os_abi = target_str,
        .cpu_features = cpu_str,
    }) catch |err| {
        std.debug.print("Error parsing target: {}\n", .{err});
        return;
    };

    const target = b.resolveTargetQuery(target_query);
    const optimize = b.standardOptimizeOption(.{});

    // Modules creation

    const zant_mod = b.createModule(.{ .root_source_file = b.path("src/zant.zig") });
    zant_mod.addOptions("build_options", build_options);

    // static_lib module for MNIST
    const static_lib_mod = b.createModule(.{ .root_source_file = b.path("src/codeGen/static_lib.zig") });
    const static_lib_mnist_hard_mod = b.createModule(.{ .root_source_file = b.path("src/codeGen/static_lib_mnist_hard.zig") });

    static_lib_mod.addImport("zant", zant_mod);
    static_lib_mnist_hard_mod.addImport("zant", zant_mod);

    // ************************************************MAIN EXECUTABLE************************************************

    const exe = b.addExecutable(.{
        .name = "Main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibC();

    exe.root_module.addImport("zant", zant_mod);

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

    // ************************************************CODEGEN EXECUTABLE************************************************

    // Define the main executable with target architecture and optimization settings.
    const codeGen_exe = b.addExecutable(.{
        .name = "Codegen",
        .root_source_file = b.path("src/codeGen/codeGen_main.zig"),
        .target = target,
        .optimize = optimize,
    });

    codeGen_exe.linkLibC();

    // Add necessary imports for the executable.
    codeGen_exe.root_module.addImport("zant", zant_mod);

    // Define codegen options
    const codegen_options = b.addOptions();
    codegen_options.addOption(bool, "log", b.option(bool, "log", "Run with log") orelse false);
    codegen_options.addOption([]const u8, "shape", b.option([]const u8, "shape", "Input shape") orelse "0");
    codegen_options.addOption([]const u8, "type", b.option([]const u8, "type", "Input type") orelse "f32");
    codegen_options.addOption(bool, "noComm", b.option(bool, "noComm", "Run with log") orelse true);
    codeGen_exe.root_module.addOptions("codegen_options", codegen_options);

    // Install the executable.
    b.installArtifact(codeGen_exe);

    // Define the run command for the main executable.
    const codegen_cmd = b.addRunArtifact(codeGen_exe);
    if (b.args) |args| {
        codegen_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const codegen_step = b.step("codegen", "code generation");
    codegen_step.dependOn(&codegen_cmd.step);

    // ************************************************ STATIC LIBRARY CREATION ************************************************

    const static_lib = b.addStaticLibrary(.{
        .name = "static_lib",
        .root_source_file = b.path("src/codeGen/static_lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    static_lib.linkLibC();
    static_lib.root_module.addImport("zant", zant_mod);

    const install_lib_step = b.addInstallArtifact(static_lib, .{});
    const lib_step = b.step("lib", "Compile tensor_math static library");
    lib_step.dependOn(&install_lib_step.step);

    //************************************************UNIT TESTS************************************************

    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "test_lib",
        .root_source_file = b.path("tests/test_lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define test options
    const test_options = b.addOptions();
    test_options.addOption(bool, "heavy", b.option(bool, "heavy", "Run heavy tests") orelse false);
    unit_tests.root_module.addOptions("test_options", test_options);

    unit_tests.root_module.addImport("zant", zant_mod);
    unit_tests.root_module.addImport("static_lib", static_lib_mod);
    unit_tests.root_module.addImport("static_lib_mnist_hard", static_lib_mnist_hard_mod);

    unit_tests.linkLibC();

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Add test for static_lib
    // const test_static_lib = b.addTest(.{
    //     .root_source_file = b.path("tests/CodeGen/test_static_lib_sentiment.zig"),
    //     .target = target,
    //     .optimize = optimize,
    // });
    // test_static_lib.root_module.addImport("static_lib_sentiment", static_lib_sentiment_mod);
    // test_static_lib.root_module.addImport("tensor", tensor_mod);
    // test_static_lib.root_module.addImport("pkgAllocator", allocator_mod);
    // test_static_lib.linkLibC();

    // const run_test_static_lib = b.addRunArtifact(test_static_lib);
    // const test_step_static_lib = b.step("test-static-lib", "Run static library tests");
    // test_step_static_lib.dependOn(&run_test_static_lib.step);
}
