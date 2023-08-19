using UMBridge
using Test

function testmodel_supported_operations(model)
    all([
        UMBridge.supports_evaluate(model),
        !UMBridge.supports_gradient(model),
        !UMBridge.supports_apply_jacobian(model),
        !UMBridge.supports_apply_hessian(model)
   ])
end

@testset "UMBridge.jl" begin

    model = UMBridge.HTTPModel("forward", "https://testmodel.linusseelinger.de")
    @test UMBridge.evaluate(model, [[4]], Dict())[1][1] == 8.0
    @test UMBridge.protocol_version_supported(model)
    @test UMBridge.get_models(model)[1] == "forward"
    @test UMBridge.model_input_sizes(model)[1] == 1
    @test UMBridge.model_output_sizes(model)[1] == 1
    @test testmodel_supported_operations(model)

end
