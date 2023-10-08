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

@testset "Test Connection" begin
    
    model=UMBridge.HTTPModel("posterior", "https://benchmark-analytic-funnel.linusseelinger.de")
    httpValue = UMBridge.evaluate(model, [[1, 3]], Dict())[1][1]
    exactValue = -5.147502395904501;
    @test isapprox(httpValue, exactValue, rtol=1e-14)
    
end

@testset "Test Apply Jacobian" begin
    
    model=UMBridge.HTTPModel("posterior", "https://benchmark-analytic-funnel.linusseelinger.de")
    httpValue = UMBridge.apply_jacobian(model, 0, 0, [[1, 3]], [1, 4])[1]
    exactValue = -3.370206919896928;
    @test isapprox(httpValue, exactValue, rtol=1e-14)
    
end

@testset "Test Gradient" begin
    
    model=UMBridge.HTTPModel("posterior", "https://benchmark-analytic-funnel.linusseelinger.de")
    httpValue = UMBridge.gradient(model, 0, 0, [[1, 3]], [12])[1][1]
    exactValue = 12.53215648992455;
    @test isapprox(httpValue, exactValue, rtol=1e-14)
    
end
