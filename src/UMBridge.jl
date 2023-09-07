module UMBridge

import HTTP
import JSON

# Make HTTP request following UM-Bridge protocol

struct HTTPModel
   name::String
   url::String
end

name(model::HTTPModel) = model.name
url(model::HTTPModel) = model.url

function check_response(response, expected_code)
    if response.status != expected_code
        error("Request failed with status code " * string(response.status) * " instead of " * string(expected_code))
    end
end

function check_parsed_response(parsed)
    if haskey(parsed, "error")
        error(parsed["error"]["type"] * ": " * parsed["error"]["message"])
    end
end

function evaluate(model::HTTPModel, input, config)
    body = Dict(
        "name" =>name(model),
        "input" => input,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/Evaluate", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)

    return parsed["output"]
end

function gradient(model::HTTPModel, out_wrt, in_wrt, input, sens, config = Dict())
    body = Dict(
        "name" =>name(model),
        "outWrt" => out_wrt,
        "inWrt" => in_wrt,
        "input" => input,
        "sens" => sens,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/Gradient", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function apply_jacobian(model::HTTPModel, out_wrt, in_wrt, input, vec, config = Dict())
    body = Dict(
        "name" =>name(model),
        "outWrt" => out_wrt,
        "inWrt" => in_wrt,
        "input" => input,
        "vec" => vec,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/ApplyJacobian", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function apply_hessian(model::HTTPModel, out_wrt, in_wrt1, in_wrt2, input, vec, sens, config = Dict())
    body = Dict(
        "name" =>name(model),
        "outWrt" => out_wrt,
        "inWrt1" => in_wrt1,
        "inWrt2" => in_wrt2,
        "input" => input,
        "vec" => vec,
        "sens" => sens,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/ApplyHessian", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function protocol_version_supported(model::HTTPModel)
    response = HTTP.request("GET", url(model) * "/Info")
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["protocolVersion"] == 1.0
end

function get_models(model::HTTPModel)
    response = HTTP.request("GET", url(model) * "/Info")
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["models"]
end

function model_input_sizes(model::HTTPModel, config = Dict())
    body = Dict(
        "name" =>name(model),
        "config" => config
    )
    response = HTTP.request("POST", url(model) * "/InputSizes", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["inputSizes"]
end

function model_output_sizes(model::HTTPModel, config = Dict())
    body = Dict(
        "name" =>name(model),
        "config" => config
    )
    response = HTTP.request("POST", url(model) * "/OutputSizes", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["outputSizes"]
end

function supports_evaluate(model::HTTPModel)
    body = Dict(
        "name" =>name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["Evaluate"]
end

function supports_gradient(model::HTTPModel)
    body = Dict(
        "name" =>name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["Gradient"]
end

function supports_apply_jacobian(model::HTTPModel)
    body = Dict(
        "name" =>name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["ApplyJacobian"]
end

function supports_apply_hessian(model::HTTPModel)
    body = Dict(
        "name" =>name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["ApplyHessian"]
end

end
