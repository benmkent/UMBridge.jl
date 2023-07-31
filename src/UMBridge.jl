module UMBridge

import HTTP
import JSON

# Make HTTP request following UM-Bridge protocol

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

function evaluate(url, name, input, config)
    body = Dict(
        "name" => name,
        "input" => input,
        "config" => config
    )

    response = HTTP.request("POST", url * "/Evaluate", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)

    return parsed["output"]
end

function gradient(url, name, out_wrt, in_wrt, parameters, sens, config = Dict())
    body = Dict(
        "name" => name,
        "outWrt" => out_wrt,
        "inWrt" => in_wrt,
        "parameters" => parameters,
        "sens" => sens,
        "config" => config
    )

    response = HTTP.request("POST", url * "/Gradient", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function apply_jacobian(url, name, out_wrt, in_wrt, parameters, vec, config = Dict())
    body = Dict(
        "name" => name,
        "outWrt" => out_wrt,
        "inWrt" => in_wrt,
        "parameters" => parameters,
        "vec" => vec,
        "config" => config
    )

    response = HTTP.request("POST", url * "/ApplyJacobian", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function apply_hessian(url, name, out_wrt, in_wrt1, in_wrt2, parameters, vec, sens, config = Dict())
    body = Dict(
        "name" => name,
        "outWrt" => out_wrt,
        "inWrt1" => in_wrt1,
        "inWrt2" => in_wrt2,
        "parameters" => parameters,
        "vec" => vec,
        "sens" => sens,
        "config" => config
    )

    response = HTTP.request("POST", url * "/ApplyHessian", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function protocol_version_supported(url)
    response = HTTP.request("GET", url * "/Info")
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["protocolVersion"] == 1.0
end

function get_models(url)
    response = HTTP.request("GET", url * "/Info")
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["models"]
end

function model_input_sizes(url, name, config = Dict())
    body = Dict(
        "name" => name,
        "config" => config
    )
    response = HTTP.request("POST", url * "/InputSizes", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["inputSizes"]
end

function model_output_sizes(url, name, config = Dict())
    body = Dict(
        "name" => name,
        "config" => config
    )
    response = HTTP.request("POST", url * "/OutputSizes", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["outputSizes"]
end

function supports_evaluate(url, name)
    body = Dict(
        "name" => name
    )
    response = HTTP.request("POST", url * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["Evaluate"]
end

function supports_gradient(url, name)
    body = Dict(
        "name" => name
    )
    response = HTTP.request("POST", url * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["Gradient"]
end

function supports_apply_jacobian(url, name)
    body = Dict(
        "name" => name
    )
    response = HTTP.request("POST", url * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["ApplyJacobian"]
end

function supports_apply_hessian(url, name)
    body = Dict(
        "name" => name
    )
    response = HTTP.request("POST", url * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["ApplyHessian"]
end

end
