using UMBridge
using HTTP
using JSON
using Test

function testmodel_supported_operations(model)
    all([
        UMBridge.supports_evaluate(model),
        !UMBridge.supports_gradient(model),
        !UMBridge.supports_apply_jacobian(model),
        !UMBridge.supports_apply_hessian(model)
   ])
end

function testserver(models)
    # Serve model
    port = 4242
    server = UMBridge.serve_models(models, port)
    # Shut down server
    close(server)
    return true
end

function testclientserverpair(models, httpmodel)
    server = UMBridge.serve_models(models, 4242)
    input = UMBridge.model_input_sizes(httpmodel)[1] == 1
    output = UMBridge.model_output_sizes(httpmodel)[1] == 1
    close(server)
    return all([input, output])
end

function testserver_sizes(models)
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "config" => Dict()
    )
    response_input  = UMBridge.inputRequest(models)(HTTP.Request("POST", "/InputSizes", [], JSON.json(body)))
    response_output = UMBridge.outputRequest(models)(HTTP.Request("POST", "/OutputSizes", [], JSON.json(body)))
    all([response_input.status == 200, response_output.status == 200])
end

function testserver_info(models)
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "config" => Dict()
    )
    response_input  = UMBridge.infoRequest(models)(HTTP.Request("GET", "/Info", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_evaluate(models)
    UMBridge.define_evaluate(models[1], (input, config) -> (input))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "input" => [1.0],
        "config" => Dict()
    )
    response_input  = UMBridge.evaluateRequest(models)(HTTP.Request("POST", "/Evaluate", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_gradient(models)
    models[1].supportsGradient = true
    UMBridge.define_gradient(models[1], (outWrt, inWrt, input, sens, config) -> ([1]))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt" => [1],
        "outWrt" => [1],
        "sens" => [1],
        "input" => [1],
        "config" => Dict()
    )
    response_input  = UMBridge.gradientRequest(models)(HTTP.Request("POST", "/Gradient", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_jacobian(models)
    models[1].supportsJacobian = true
    UMBridge.define_applyjacobian(models[1], (outWrt, inWrt, input, vec, config) -> ([1]))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt" => [1],
        "outWrt" => [1],
        "input" => [1],
        "vec" => [1],
        "config" => Dict()
    )
    response_input  = UMBridge.applyJacobianRequest(models)(HTTP.Request("POST", "/ApplyJacobian", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_hessian(models)
    models[1].supportsHessian = true
    UMBridge.define_applyhessian(models[1], (outWrt, inWrt1, inWrt2, input, sens, vec, config) -> ([1]))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt1" => [1],
        "inWrt2" => [1],
        "outWrt" => [1],
        "input" => [1],
        "vec" => [1],
        "sens" => [1],
        "config" => Dict()
    )
    response_input  = UMBridge.applyHessianRequest(models)(HTTP.Request("POST", "/ApplyHessian", [], JSON.json(body)))
    return response_input.status == 200
end

@testset "UMBridge.jl" begin

    # Test client
    httpmodel = UMBridge.HTTPModel("forward", "https://testmodel.linusseelinger.de")
    @test UMBridge.evaluate(httpmodel, [[4]], Dict())[1][1] == 8.0
    @test UMBridge.protocol_version_supported(httpmodel)
    @test UMBridge.get_models(httpmodel)[1] == "forward"
    @test UMBridge.model_input_sizes(httpmodel)[1] == 1
    @test UMBridge.model_output_sizes(httpmodel)[1] == 1
    @test testmodel_supported_operations(httpmodel)

    # Test server
    modelinput = UMBridge.model_input_sizes(httpmodel)
    modeloutput = UMBridge.model_output_sizes(httpmodel)
    models = [UMBridge.Model(name = "forward", inputSizes = modelinput, outputSizes = modeloutput)]
    @test testserver(models)
    @test testclientserverpair(models,httpmodel)
    @test testserver_sizes(models)
    @test testserver_info(models)
    @test testserver_evaluate(models)
    @test testserver_gradient(models)
    @test testserver_jacobian(models)
    @test testserver_hessian(models)

end
