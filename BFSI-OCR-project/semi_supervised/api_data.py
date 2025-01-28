import requests
import json
import os
from tornado.websocket import WebSocketClosedError

# SEC API details
API_KEY = "9a5063ede94c5e26056eff4e4163b1545653507f09f8282d15e83a50f1004441"
XBRL_CONVERTER_API_ENDPOINT = "https://api.sec-api.io/xbrl-to-json"

def fetch_financial_data(filing_url):
    """
    Fetch financial data from SEC API.
    Handles timeouts and request exceptions.
    """
    try:
        final_url = f"{XBRL_CONVERTER_API_ENDPOINT}?htm-url={filing_url}&token={API_KEY}"
        response = requests.get(final_url, timeout=10)  # Timeout for API call
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

def save_json(data, file_path="financial_data.json"):
    """
    Save JSON data to a file with error handling.
    """
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"JSON data saved to {file_path}")
        return f"JSON data saved to {file_path}"
    except Exception as e:
        error_message = f"Failed to save JSON: {str(e)}"
        print(error_message)
        return {"error": error_message}

def json_to_xml(json_obj, line_padding=""):
    """
    Convert JSON to XML format.
    Handles nested JSON structures recursively.
    """
    result_list = []
    json_obj_type = type(json_obj)

    if json_obj_type is list:
        for sub_elem in json_obj:
            result_list.append(json_to_xml(sub_elem, line_padding))
        return "\n".join(result_list)

    if json_obj_type is dict:
        for tag_name in json_obj:
            sub_obj = json_obj[tag_name]
            result_list.append(
                f"{line_padding}<{tag_name}>"
                f"{json_to_xml(sub_obj, line_padding + '  ')}"
                f"</{tag_name}>"
            )
        return "\n".join(result_list)

    return f"{line_padding}{json_obj}"

def save_xml(data, file_path="financial_data.xml"):
    """
    Convert JSON to XML and save to a file with error handling.
    """
    try:
        xml_content = json_to_xml(data)
        with open(file_path, "w") as xml_file:
            xml_file.write(f"<root>\n{xml_content}\n</root>")
        print(f"XML data saved to {file_path}")
        return f"XML data saved to {file_path}"
    except Exception as e:
        error_message = f"Failed to save XML: {str(e)}"
        print(error_message)
        return {"error": error_message}

# Example WebSocket Handling Code
async def send_message_over_websocket(socket, message):
    """
    Send a message over a WebSocket with error handling for closed connections.
    """
    try:
        await socket.write_message(message)
    except WebSocketClosedError:
        print("WebSocket connection closed. Cannot send message.")
