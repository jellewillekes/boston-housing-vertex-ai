import json


# Convert JSONL to a single JSON payload
def convert_jsonl_to_payload(jsonl_path):
    with open(jsonl_path, "r") as f:
        instances = [json.loads(line)["input"] for line in f]
    return {"instances": instances}


# Save the payload to a file for testing
payload = convert_jsonl_to_payload("../prediction_input.jsonl")
with open("../payload.json", "w") as f:
    json.dump(payload, f, indent=4)

print("Converted payload saved to payload.json")
