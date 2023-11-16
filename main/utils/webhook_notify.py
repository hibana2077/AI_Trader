'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-11-16 22:49:02
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-11-16 23:00:08
FilePath: \AI_Trader\main\utils\webhook_notify.py
Description: This module contains functions for sending reports to a discord webhook.
'''
import requests

def send_text_report(webhook_url:str, time_usage:float, mertic_dict:dict) -> str:
    """
    Sends a text report to a webhook URL with the given time usage and metric dictionary.

    Args:
        webhook_url (str): The URL of the discord webhook to send the report to.
        time_usage (float): The time usage to include in the report.
        mertic_dict (dict): The metric dictionary to include in the report.

    Returns:
        str: The response text from the webhook.
    """
    template = """
=========================
    AI Trader Report
=========================

Time Usage: {time_usage:.2f} seconds
"""
    for k, v in mertic_dict.items():
        template += f"{k}: {v}\n"
    data = {
        "content": template.format(time_usage=time_usage),
        "username": "AI Trader Notification",
        "avatar_url": "https://i.imgur.com/4bY31Fb.jpg"
    }
    response = requests.post(webhook_url, json=data)

    if response.status_code != 204:
        raise ValueError(
            f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}"
        )
    return response.text

def send_image_report(webhook_url:str, image_path:str) -> str:
    """
    Sends an image report to a webhook URL.

    Args:
    webhook_url (str): The URL of the webhook to send the report to.
    image_path (str): The path of the image to include in the report.

    Returns:
    str: The response text from the webhook.
    """
    template = """
=========================
    AI Trader Report
=========================
"""
    data = {
        "content": template,
        "username": "AI Trader Notification",
        "avatar_url": "https://i.imgur.com/4bY31Fb.jpg",
    }

    files = {
        "file": (image_path, open(image_path, "rb")),
    }

    response = requests.post(webhook_url, data=data, files=files)

    if response.status_code != 204:
        raise ValueError(
            f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}"
        )
    return response.text
