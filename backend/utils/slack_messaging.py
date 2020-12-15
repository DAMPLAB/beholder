import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

token = 'xoxb-22974935442-1562739401110-y2hA1mCP4aZKIbQdnMV36bJ2'
client = WebClient(token=token)

andre_channel = 'C01GN8U5GTX'


def slack_message(input_text: str, channel: str = '#andre'):
    try:
        response = client.chat_postMessage(channel=channel, text=f'{input_text}')
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")
