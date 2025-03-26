from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


def request_token(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Requests an OAuth2 token from a specified API using the authorization code flow.

    Args:
        config (Dict[str, Any]): Dictionary containing API credentials and settings.

    Returns:
        Optional[Dict[str, Any]]: Token response data if successful, otherwise None.
    """
    # Prepare token request parameters
    token_params = config["token_params"]
    token_params["code"] = config["auth_code"]
    token_params["redirect_uri"] = config["redirect_uri"]

    # Determine authentication method
    if config.get("use_basic_auth", True):
        auth = HTTPBasicAuth(config["client_id"], config["client_secret"])
        response = requests.post(config["token_url"], params=token_params, auth=auth)
    else:
        token_params["client_id"] = config["client_id"]
        token_params["client_secret"] = config["client_secret"]
        response = requests.post(config["token_url"], data=token_params)

    # Attempt to parse JSON response
    try:
        data = response.json()
    except ValueError:
        print("Failed to parse JSON response:")
        print(response.text)
        return None

    # Handle successful or failed response
    if response.status_code == 200 and "access_token" in data:
        access_token = data["access_token"]
        refresh_token = data.get("refresh_token", "")
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        print("Access Token:", access_token)
        print("Refresh Token:", refresh_token)

        # Save token data to CSV
        df = pd.DataFrame(
            [
                {
                    "time": str(timestamp),
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                }
            ]
        )
        filename = f"{timestamp_str}_tokens.csv"
        df.to_csv(filename, index=False)
        print(f"Token details saved to: {filename}")

        return data
    else:
        print("Failed to retrieve tokens.")
        print("Status Code:", response.status_code)
        print("Response:", data)
        return None


def main() -> None:
    """
    Main function demonstrating example usage of the OAuth2 token requester.
    """
    example_config: Dict[str, Any] = {
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "auth_code": "your_auth_code",
        "redirect_uri": "your_redirect_uri",
        "token_url": "https://your-api.com/oauth/token",  # Replace with actual token endpoint
        "token_params": {
            "grant_type": "authorization_code",
            "code": "",  # Will be overwritten
            "redirect_uri": "",  # Will be overwritten
        },
        "use_basic_auth": True,  # Set to False if API expects creds in body
    }

    token_data = request_token(example_config)

    if token_data:
        print("Token request succeeded.")
    else:
        print("Token request failed.")


if __name__ == "__main__":
    main()
