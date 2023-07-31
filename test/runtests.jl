using UMBridge
using HTTP
using Test

function testmodel_supported_operations()
    all([
        UMBridge.supports_evaluate("https://testmodel.linusseelinger.de", "forward"),
        !UMBridge.supports_gradient("https://testmodel.linusseelinger.de", "forward"),
        !UMBridge.supports_apply_jacobian("https://testmodel.linusseelinger.de", "forward"),
        !UMBridge.supports_apply_hessian("https://testmodel.linusseelinger.de", "forward")
   ])
end

@testset "UMBridge.jl" begin

    @test HTTP.request("GET", "https://testmodel.linusseelinger.de/Info").status == 200
    @test UMBridge.evaluate("https://testmodel.linusseelinger.de", "forward", [[4]], Dict())[1][1] == 8.0
    @test UMBridge.protocol_version_supported("https://testmodel.linusseelinger.de")
    @test UMBridge.get_models("https://testmodel.linusseelinger.de")[1] == "forward"
    @test UMBridge.model_input_sizes("https://testmodel.linusseelinger.de", "forward")[1] == 1
    @test UMBridge.model_output_sizes("https://testmodel.linusseelinger.de", "forward")[1] == 1
    @test testmodel_supported_operations()

end
